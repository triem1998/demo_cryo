"""Shared base config — imported by run.py and inference modules."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from pydantic import BaseModel, ConfigDict


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
    loss_type: str = "icecream"

    # ── Training ────────────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    grad_clip: float | None = 1.0
    ckp_interval: int = 10
    eval_interval: int = 1

    # ── Physics ─────────────────────────────────────────────────────────────
    wedge_double_size: bool = True

    # ── Mixed precision ──────────────────────────────────────────────────────
    use_mixed_precision: bool = True
    # "fp16" (default, unchanged behaviour + GradScaler) or "bf16". bf16 has
    # fp32's dynamic range, so it needs no loss scaling and cannot overflow —
    # use it for the unrolled/full preset, whose large native-resolution
    # gradients overflow fp16's 65504 ceiling.
    mixed_precision_dtype: str = "fp16"

    # ── Model ───────────────────────────────────────────────────────────────
    model_type: str = "unet"
    unet_dropout: float = 0.1
    drunet_sigma: float = 0.0

    # ── Evaluation ──────────────────────────────────────────────────────────
    fsc_threshold: float = 0.143
    pixel_size_angstrom: float | None = None
    save_fsc_curves: bool = True   # write the full per-shell FSC curve, not just the resolution

    # ── Pretrained init ──────────────────────────────────────────────────────
    pretrained_ckpt: str | None = None

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
