"""Physics operator construction from a run config.

Heavy code lives in the sibling modules — ``missingwedge.py`` (``MissingWedge``,
for the ``missingwedge_ei`` preset), the two interchangeable tomography
operators ``tomography.py`` (astra) / ``tomography_torch.py`` (pure torch), and
``tomography_build.py``, the backend-neutral layer that builds and pairs either
of them. This ``__init__`` is the thin construction layer: one builder per
preset, plus re-exports so callers can do ``from ..physics import X`` without
knowing which sibling module ``X`` actually lives in.

Reading ``cfg`` stops here: this is the only module that touches
``cfg.tomography_backend``; everything below it takes a resolved backend name.
"""
from __future__ import annotations

from pathlib import Path

from deepinv.distributed import DistributedContext

from ..base_config import RunEIBaseConfig
from .missingwedge import MissingWedge
from .tomography import TomographyEM
from .tomography_build import (
    TOMOGRAPHY_BACKENDS, TomographyEMPair, build_one_tomography_em,
    normalize_sharded, resolve_tomography_backend, split_sinogram,
)
from .tomography_torch import TomographyEMTorch

__all__ = [
    "MissingWedge", "TomographyEM", "TomographyEMPair", "TomographyEMTorch",
    "TOMOGRAPHY_BACKENDS", "resolve_tomography_backend",
    "build_missingwedge_physics", "build_tomography_physics", "split_sinogram",
]


def build_missingwedge_physics(
    cfg: RunEIBaseConfig, crop_size: int | tuple[int, int, int], device,
) -> MissingWedge:
    """``crop_size``: an int (cubic — patch preset) or a (D, H, W) tuple (full preset).

    ``MissingWedge`` ignores ``crop_size`` whenever ``volume_shape`` is given.
    """
    volume_shape = None if isinstance(crop_size, int) else tuple(int(v) for v in crop_size)
    return MissingWedge(
        tilt_max=float(cfg.tilt_max), tilt_min=float(cfg.tilt_min),
        crop_size=crop_size,
        volume_shape=volume_shape,
        use_spherical_support=bool(cfg.use_spherical_support),
        wedge_double_size=bool(cfg.wedge_double_size),
        wedge_low_support=float(cfg.wedge_low_support),
        ref_wedge_support=float(cfg.ref_wedge_support),
        device=str(device),
    ).to(device)


def build_tomography_physics(
    cfg: RunEIBaseConfig, evn_paths: list[Path], odd_paths: list[Path], device, ctx: DistributedContext,
) -> TomographyEMPair:
    """Build a tomogram's EVN/ODD tomography operators and their FBP inits.

    The operator ends up unit spectral norm either way — ``normalize=True``
    unsharded, ``normalize_sharded`` when the angles are split — so a
    z-normalized volume maps onto the z-normalized sinogram's scale and the PGD
    stepsize needs no rescaling.

    ``cfg.num_operators`` shards the tilt angles across ranks (null = one full
    operator per rank, only the denoiser tiled); ``cfg.tomography_backend``
    picks the astra or the pure-torch operator.

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

    # cfg.num_operators: None = one full operator per rank (no physics
    # collective, today's behaviour); "auto" = one operator per rank; an int =
    # that many, distributed round-robin. Resolved here because this is the
    # first place ctx.world_size is known.
    n_ops = getattr(cfg, "num_operators", None)
    if n_ops == "auto":
        n_ops = int(ctx.world_size)
    elif n_ops is not None:
        n_ops = int(n_ops)

    backend = resolve_tomography_backend(
        getattr(cfg, "tomography_backend", "auto"), device)
    if ctx.rank == 0:
        print(f"[physics] tomography backend: {backend} "
              f"({TOMOGRAPHY_BACKENDS[backend].__name__})", flush=True)

    evn_path, odd_path = evn_paths[0], odd_paths[0]
    physics_evn, init_evn = build_one_tomography_em(
        evn_path.parent, "split1", evn_path, device, target_shape, n_ops, ctx, backend)
    physics_odd, init_odd = build_one_tomography_em(
        odd_path.parent, "split2", odd_path, device, target_shape, n_ops, ctx, backend)

    # Shards are built unnormalised (a shard's own norm is not the operator's),
    # then all rescaled by the measured global norm — leaving the assembled
    # operator unit-norm, exactly as normalize=True leaves the unsharded one.
    if n_ops is not None:
        # build_one_tomography_em caps the shard count at the tilt count, so read
        # back what was actually built: TomographyEMPair.num_operators drives
        # split_sinogram (forward.py), which must match the operator exactly.
        requested, n_ops = n_ops, int(physics_evn.num_operators)
        sq_evn = normalize_sharded(physics_evn, init_evn)
        normalize_sharded(physics_odd, init_odd)
        if ctx.rank == 0:
            capped = f" (capped from {requested} by the tilt count)" if n_ops != requested else ""
            print(f"[physics] sharded into {n_ops} operator(s) over {ctx.world_size} rank(s)"
                  f"{capped}  ||A^T A||_2={sq_evn:.4g} -> normalised", flush=True)

    return TomographyEMPair(
        physics_evn=physics_evn, physics_odd=physics_odd,
        init_evn=init_evn, init_odd=init_odd,
        evn_paths=evn_paths, odd_paths=odd_paths, device=device, target_shape=target_shape,
        num_operators=n_ops, backend=backend, ctx=ctx,
    )
