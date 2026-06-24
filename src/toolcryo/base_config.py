"""Shared base config and physics builder — imported by run.py and inference modules."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .physics import MissingWedge


class RunEIBaseConfig(BaseModel):
    """Fields shared across all training and inference configs."""
    model_config = ConfigDict(extra="ignore")

    # ── Data ────────────────────────────────────────────────────────────────
    input_dir: str = "./dataset/empiar-11058"
    max_train_vols: int | None = None
    max_val_vols: int = 5
    seed: int = 0

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

    # ── Model ───────────────────────────────────────────────────────────────
    model_type: str = "unet"
    unet_dropout: float = 0.1
    drunet_sigma: float = 0.0

    # ── Evaluation ──────────────────────────────────────────────────────────
    fsc_threshold: float = 0.143
    pixel_size_angstrom: float | None = None

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


def _build_physics(cfg: RunEIBaseConfig, crop_size: int, device) -> MissingWedge:
    return MissingWedge(
        tilt_max=float(cfg.tilt_max), tilt_min=float(cfg.tilt_min),
        crop_size=crop_size,
        use_spherical_support=bool(cfg.use_spherical_support),
        wedge_double_size=bool(cfg.wedge_double_size),
        wedge_low_support=float(cfg.wedge_low_support),
        ref_wedge_support=float(cfg.ref_wedge_support),
        device=str(device),
    ).to(device)
