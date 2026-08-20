"""Shared base config — imported by run.py and inference modules."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, model_validator


def amp_dtype_from_str(mixed_precision: str) -> torch.dtype | None:
    """Map a ``mixed_precision`` config value to a torch dtype, or ``None``.

    ``None`` means "off": every consumer treats it as *do nothing* — no
    autocast, no GradScaler, no half-precision cast anywhere. Shared by the
    trainer and the inference paths so a run cannot train in one dtype and then
    evaluate or infer in another.
    """
    if mixed_precision == "off":
        return None
    if mixed_precision not in ("fp16", "bf16"):
        raise ValueError(
            f"mixed_precision must be 'off', 'fp16' or 'bf16', got {mixed_precision!r}."
        )
    return torch.bfloat16 if mixed_precision == "bf16" else torch.float16


class RunEIBaseConfig(BaseModel):
    """Fields shared across all training and inference configs."""
    model_config = ConfigDict(extra="ignore")

    # ── Method ──────────────────────────────────────────────────────────────
    # Selects the (physics, model, losses) builder triple from registry.py.
    preset: str = "missingwedge_ei"

    # ── Data ────────────────────────────────────────────────────────────────
    input_dir: str = "./dataset/empiar-11058"
    max_train_vols: int | None = None
    max_val_vols: int = 5
    seed: int = 0
    # Select train / val volumes by tomo dir name (e.g. ["tomo_001"]).
    # Empty = fall back to the random split by max_train_vols / max_val_vols.
    train_names: list[str] = []
    val_names: list[str] = []

    # ── DataLoader (shared defaults) ─────────────────────────────────────────
    pin_memory: bool = True
    persistent_workers: bool = True

    # ── Physics ─────────────────────────────────────────────────────────────
    tilt_max: float = 60.0
    tilt_min: float = -60.0
    use_spherical_support: bool = True
    wedge_low_support: float = 0.0
    ref_wedge_support: float = 1.0

    # ── EI loss ─────────────────────────────────────────────────────────────
    eq_weight: float = 2.0

    # ── Training ────────────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    grad_clip: float | None = 1.0
    ckp_interval: int = 10
    eval_interval: int = 1
    # ReduceLROnPlateau on the training TotalLoss. Off by default — unchanged
    # behaviour (flat learning_rate for the whole run) unless enabled.
    use_lr_scheduler: bool = False
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 10

    # ── Physics ─────────────────────────────────────────────────────────────
    wedge_double_size: bool = True
    # Tomography operator backend (unrolled/tomo_ei presets only; ignored by
    # missingwedge_ei). "auto" = astra wherever its CUDA kernels can actually
    # run, else the pure-torch operator — which is also the only one that works
    # on CPU and on AMD/ROCm. "torch" reproduces astra's approximate adjoint, so
    # switching backends leaves the gradient unchanged; "torch_exact" uses the
    # true transpose instead (a real PGD gradient, ~4x slower at native size).
    tomography_backend: Literal["auto", "astra", "torch", "torch_exact"] = "auto"

    # ── Mixed precision ──────────────────────────────────────────────────────
    # One switch for the whole run — training, validation and inference alike.
    #
    #   "off"  : pure fp32. A strict no-op — no autocast, no GradScaler, no
    #            half cast in the inference sliding window.
    #   "fp16" : icecream's default. Needs a GradScaler, and overflows at
    #            native resolution (the 65504 ceiling).
    #   "bf16" : fp32's dynamic range, so no loss scaling and no overflow —
    #            the right choice for the full/unrolled presets.
    mixed_precision: Literal["off", "fp16", "bf16"] = "off"

    # ── Model ───────────────────────────────────────────────────────────────
    # torch.compile the denoiser, always *before* the distribute() tiling
    # wrapper, so the compiled region is the plain denoiser rather than the
    # wrapper's Python tiling loop.
    compile: bool = False
    model_type: str = "unet"
    unet_dropout: float = 0.1
    drunet_sigma: float = 0.0

    # ── Evaluation ──────────────────────────────────────────────────────────
    fsc_threshold: float = 0.143
    pixel_size_angstrom: float | None = None
    save_fsc_curves: bool = True   # write the full per-shell FSC curve, not just the resolution

    # ── Pretrained init ──────────────────────────────────────────────────────
    pretrained_ckpt: str | None = None

    # ── Checkpoint axis order (tomo_ei only — missingwedge_ei never needs this,
    # it stays in native order throughout) ────────────────────────────────────
    # None = auto: True when loading a checkpoint for tomo_ei *training*
    # (pretrained_ckpt is assumed to be a native-order missingwedge_ei/patch
    # checkpoint), False for tomo_ei *inference* (checkpoint_paths is assumed
    # to already be astra-order, i.e. produced by a previous tomo_ei/unrolled
    # run). Set explicitly to override either way — e.g. True to evaluate a
    # raw patch checkpoint directly under tomo_ei inference, or False to
    # resume tomo_ei training from its own (already astra-order) checkpoint.
    permute_native_to_astra: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_amp_keys(cls, data):
        """Fail loudly on the retired ``use_mixed_precision`` / ``mixed_precision_dtype``.

        ``model_config`` sets ``extra="ignore"``, so without this an old config
        would be accepted with its AMP keys silently dropped and the run would
        fall back to the ``mixed_precision`` default — flipping precision with
        nothing printed. Raising is the only safe migration.
        """
        if isinstance(data, dict):
            legacy = [k for k in ("use_mixed_precision", "mixed_precision_dtype") if k in data]
            if legacy:
                raise ValueError(
                    f"{', '.join(legacy)} has been replaced by a single field: "
                    'mixed_precision: "off" | "fp16" | "bf16". '
                    "Rewrite the config — use_mixed_precision: false becomes "
                    'mixed_precision: "off", and true + mixed_precision_dtype: bf16 '
                    'becomes mixed_precision: "bf16".'
                )
        return data

    @classmethod
    def _flat_from_yaml(cls, conf: dict, default_run_name: str) -> dict:
        """Flatten all YAML sections into a single dict and compute output_dir."""
        flat: dict = {}
        for section in conf.values():
            if isinstance(section, dict):
                flat.update(section)
        general     = conf.get("general", {})
        slurm       = conf.get("slurm", {})
        timestamp   = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name    = general.get("run_name", slurm.get("job_name", default_run_name))
        output_root = general.get("output_root", "./runs")
        flat["output_dir"] = str(Path(output_root) / f"{run_name}_{timestamp}")
        return flat
