"""Physics operator construction from a run config."""
from __future__ import annotations

from ..base_config import RunEIBaseConfig
from ..physics import MissingWedge


def build_physics(cfg: RunEIBaseConfig, crop_size: int, device) -> MissingWedge:
    return MissingWedge(
        tilt_max=float(cfg.tilt_max), tilt_min=float(cfg.tilt_min),
        crop_size=crop_size,
        use_spherical_support=bool(cfg.use_spherical_support),
        wedge_double_size=bool(cfg.wedge_double_size),
        wedge_low_support=float(cfg.wedge_low_support),
        ref_wedge_support=float(cfg.ref_wedge_support),
        device=str(device),
    ).to(device)
