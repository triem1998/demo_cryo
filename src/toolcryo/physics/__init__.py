"""Physics operator construction from a run config.

Heavy classes live in the sibling modules — ``missingwedge.py``
(``MissingWedge``, for the ``missingwedge_ei`` preset) and ``tomography.py``
(``TomographyEM``/``TomographyEMPair``, for the ``unrolled``/tomography-domain
presets). This ``__init__`` is the thin construction layer: one builder per
preset, plus re-exports so callers can do ``from ..physics import X`` without
knowing which sibling module ``X`` actually lives in.
"""
from __future__ import annotations

from pathlib import Path

from deepinv.distributed import DistributedContext

from ..base_config import RunEIBaseConfig
from .missingwedge import MissingWedge
from .tomography import TomographyEM, TomographyEMPair, build_one_tomography_em

__all__ = [
    "MissingWedge", "TomographyEM", "TomographyEMPair",
    "build_missingwedge_physics", "build_tomography_physics",
]


def build_missingwedge_physics(cfg: RunEIBaseConfig, crop_size: int, device) -> MissingWedge:
    return MissingWedge(
        tilt_max=float(cfg.tilt_max), tilt_min=float(cfg.tilt_min),
        crop_size=crop_size,
        use_spherical_support=bool(cfg.use_spherical_support),
        wedge_double_size=bool(cfg.wedge_double_size),
        wedge_low_support=float(cfg.wedge_low_support),
        ref_wedge_support=float(cfg.ref_wedge_support),
        device=str(device),
    ).to(device)


def build_tomography_physics(
    cfg: RunEIBaseConfig, evn_paths: list[Path], odd_paths: list[Path], device, ctx: DistributedContext,
) -> TomographyEMPair:
    """Build a tomogram's EVN/ODD TomographyEM operators and their FBP inits.

    The operators are built with ``normalize=True``, so each has unit spectral
    norm and maps a z-normalized volume onto the same scale as the z-normalized
    sinogram — no separate operator-norm bookkeeping or stepsize rescaling is
    needed.

    Each rank runs the full operator locally — the tilt angles are not sharded
    across ranks; only the denoiser is tiled. ``ctx`` is accepted for a uniform
    ``preset["physics"]`` signature but no longer used here.

    ``evn_paths``/``odd_paths`` are the *combined* (train, then val)
    discovered FBP volume paths — ``train_ds.evn_paths + val_ds.evn_paths``.
    Only volume 0 (the first training volume) is built eagerly here; the rest
    are built lazily by ``TomographyEMPair.update()`` as each tomogram is
    encountered — its ``tomo_idx`` (from ``CryoEIFullDataset.index_offset``)
    indexes into these same lists. If ``cfg.target_shape`` is set, both the
    init volumes here and the sinograms loaded by
    ``CryoEIFullDataset._load_measurement`` are resampled to match (local
    testing only — not used for a real training run).
    """
    target_shape = getattr(cfg, "target_shape", None)
    evn_path, odd_path = evn_paths[0], odd_paths[0]
    physics_evn, init_evn = build_one_tomography_em(
        evn_path.parent, "split1", evn_path, device, target_shape)
    physics_odd, init_odd = build_one_tomography_em(
        odd_path.parent, "split2", odd_path, device, target_shape)

    return TomographyEMPair(
        physics_evn=physics_evn, physics_odd=physics_odd,
        init_evn=init_evn, init_odd=init_odd,
        evn_paths=evn_paths, odd_paths=odd_paths, device=device, target_shape=target_shape,
    )
