"""Backend-neutral tomography construction — the layer above the two operators.

``tomography.py`` (astra) and ``tomography_torch.py`` (pure torch) each define
one operator and nothing else. Everything *about* those operators that belongs
to neither — picking between them, building one from a tomogram's ``.tlt`` file,
pairing the EVN/ODD halves, sharding the angles, normalising a sharded
operator — lives here, so neither backend has to import the other.

The two operators share a constructor signature by design, which is what makes
this whole module backend-agnostic: the backend is a dict lookup
(``TOMOGRAPHY_BACKENDS``) and everything downstream is identical.

``physics/__init__.py`` sits one level up and does the cfg -> objects step; it
is the only place ``cfg.tomography_backend`` is read.
"""
from __future__ import annotations

import fcntl
import importlib.util
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from deepinv.distributed.framework import DistributedStackedLinearPhysics
from deepinv.distributed.framework.distributed_utils import DistributedGradientSync
from deepinv.utils.tensorlist import TensorList

from ..utils.utils import _read_mrc_vol_shape
from .tomography import TomographyEM
from .tomography_torch import TomographyEMTorch

#: The interchangeable operators, keyed by the ``tomography_backend`` config
#: value. They share a constructor signature, so the backend is a pure lookup.
TOMOGRAPHY_BACKENDS = {
    "astra": TomographyEM,
    # astra's back-projector is not a true transpose, so reproducing it keeps the
    # gradient unchanged when ``auto`` flips astra -> torch; ~4x quicker too.
    "torch": partial(TomographyEMTorch, adjoint_mode="fast"),
    "torch_exact": TomographyEMTorch,   # true transpose: a real PGD gradient
}


def resolve_tomography_backend(backend: str, device) -> str:
    """Turn ``cfg.tomography_backend`` into a concrete key of ``TOMOGRAPHY_BACKENDS``.

    ``"astra"``/``"torch"`` are taken as given (so a run can be forced onto
    either). ``"auto"`` picks astra only where it can actually run — astra ships
    CUDA kernels, so a ROCm build of torch or a CPU device must fall back to the
    pure-torch operator, which is numerically equivalent (see
    ``tests/test_tomography_torch.py``) at ~4-5x the cost.
    """
    if backend not in ("auto", *TOMOGRAPHY_BACKENDS):
        raise ValueError(
            f"tomography_backend must be 'auto', 'astra' or 'torch', got {backend!r}.")
    if backend != "auto":
        return backend
    astra_usable = (
        torch.device(device).type == "cuda"
        and torch.version.hip is None
        and importlib.util.find_spec("astra") is not None
    )
    return "astra" if astra_usable else "torch"


def projection_splits(num_angles: int, num_operators: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` angle ranges, one per operator (demo_tomo's split)."""
    base, rem = divmod(int(num_angles), int(num_operators))
    sizes = [base + (1 if i < rem else 0) for i in range(num_operators)]
    edges = [0]
    for s in sizes:
        edges.append(edges[-1] + s)
    return [(edges[i], edges[i + 1]) for i in range(num_operators)]


def split_sinogram(y: torch.Tensor, num_operators: int) -> TensorList:
    """Split a ``(B, C, V, A, N)`` sinogram along the angle axis to match the
    sharded operators — the measurement counterpart of ``projection_splits``.
    Same layout and axis as demo_tomo's ``split_sinogram``.
    """
    chunks = projection_splits(int(y.shape[3]), num_operators)
    return TensorList([y[:, :, :, s:e, :].contiguous() for (s, e) in chunks])


class ShardedTomography(DistributedStackedLinearPhysics):
    """Angle-sharded tomography + the one method deepinv's container lacks: ``fbp``.

    ``fbp`` is a ``TomographyEM`` method, not part of ``LinearPhysics``, so the
    distributed container has none. It is the same map-reduce as ``A_adjoint``
    (each shard back-projects its own angles, the volumes are summed across
    ranks) plus the two corrections that keep it identical to the unsharded
    operator: each shard divides by its *own* angle count, so it is reweighted
    by ``A_i / n_angles_total`` (attached by ``build_one_tomography_em``); and
    the DC centring uses the global sinogram mean, since centring is
    shift-idempotent and so cannot be recovered shard by shard.
    """

    def A(self, x, gather: bool = True, **kwargs):
        """``A`` with one padded ``all_gather``: every shape is known from ``projection_splits``."""
        if not (gather and self.ctx.use_dist) or kwargs:
            return super().A(x, gather=gather, **kwargs)
        if x.requires_grad:
            x = DistributedGradientSync.apply(x, self.ctx)   # same input-grad sum as deepinv
        splits = projection_splits(self.n_angles_total, self.num_operators)
        w, n_max = self.ctx.inner_world_size, splits[0][1] - splits[0][0]
        local = [F.pad(p.A(x), (0, 0, 0, n_max - p.n_angles)) for p in self.local_physics]
        # zero rows tied to x: an empty rank then picks the same (autograd) collective
        zero = (0 * x.reshape(-1)[0]).expand(
            *x.shape[:2], self.volume_shape[0], n_max, self.volume_shape[2])
        local += [zero] * (-(-self.num_operators // w) - len(local))
        g = self.ctx.all_gather(torch.stack(local))   # (w, k_max, B, C, V, n_max, N)
        return TensorList([g[i % w, i // w, ..., :e - s, :] for i, (s, e) in enumerate(splits)])

    def fbp(self, y, gather: bool = True, **kwargs):
        if len(y) != self.num_operators:
            raise ValueError(
                f"fbp needs the whole sinogram (all {self.num_operators} pieces, as "
                f"returned by A(x)), got {len(y)}: the global DC mean cannot be formed "
                f"from a subset, and centring per shard is not equivalent.")
        # Every rank holds every piece (A gathers), so the global mean is local
        # arithmetic — no collective. Summing first and dividing once is the plain
        # definition of the mean; the A_i/A reweighting below is still needed
        # because each shard's fbp_raw divides by its *own* angle count.
        count = sum(t.shape[-3] * t.shape[-2] * t.shape[-1] for t in y)
        mean = sum(t.sum(dim=(-3, -2, -1), keepdim=True) for t in y) / count
        out = sum(p.fbp_raw(y[i] - mean) * (p.n_angles / self.n_angles_total)
                  for i, p in zip(self.local_indexes, self.local_physics))
        if not torch.is_tensor(out):   # empty rank: zeros tied to y, same collective choice
            out = (0 * mean).expand(*mean.shape[:2], *self.volume_shape).contiguous()
        return self.ctx.all_reduce(out) if gather else out


# ---------------------------------------------------------------------------
# A tomogram's two half-set (EVN/ODD) TomographyEM operators, bundled with
# their FBP-init volumes. Each half is built from its own .tlt, so differing
# angle lists are supported. Method-agnostic physics: consumed by the
# unrolled and tomo_ei presets, reusable by future tomography-domain presets.
# ---------------------------------------------------------------------------

# Calibrated for this dataset's acquisition convention (see
# scripts/test_tomography_em.py) — the tilt sign matches IMOD's convention
# negated. The tilt axis is Y, first in astra order.
_TOMO_ANGLE_SIGN = -1.0


@dataclass
class TomographyEMPair:
    """A tomogram's two half-set (EVN/ODD) tomography operators + FBP inits.

    With ``num_operators=None`` each rank runs the full operator locally and only
    the denoiser is tiled; otherwise the angles are sharded and every
    ``A_adjoint``/``fbp`` costs a collective.
    """
    physics_evn: "TomographyEM"
    physics_odd: "TomographyEM"
    init_evn: torch.Tensor | None   # set per batch by update(), from the dataloader
    init_odd: torch.Tensor | None
    # Full (train + val, concatenated) path lists + build params, kept so
    # ``update()`` can lazily rebuild physics_evn/odd for whichever tomogram
    # the current batch is — see CryoEIFullDataset.index_offset.
    evn_paths: list
    odd_paths: list
    device: object
    target_shape: tuple | None
    # None = one full operator per rank (no physics collective). An int shards
    # the angles into that many operators, distributed round-robin across ranks.
    num_operators: int | None = None
    # Backend the pair was built with — carried so a tomogram switch rebuilds
    # on the same one it started on (already resolved, never "auto").
    backend: str = "astra"
    ctx: object = None
    _tomo_idx: int = 0
    psnr_ref: object = None   # this batch's PSNR reference, or None

    def update(self, tomo_idx=None, psnr_ref=None, init_evn=None, init_odd=None,
               **kwargs) -> None:
        """Store this batch's FBP inits; rebuild physics_evn/odd for a different tomogram.

        ``deepinv.Trainer`` calls ``physics.update(**params)`` before every
        forward pass (train and eval) — mirrors how ``MissingWedge`` rebuilds
        its wedge mask per volume (dataset_full.py), except a TomographyEM
        operator is calibrated to one tomogram's actual tilt angles, so a
        "rebuild" here means re-reading that tomogram's .tlt file, not just
        recomputing a mask from two floats.
        """
        self.psnr_ref = psnr_ref   # before the early returns
        if init_evn is not None:
            self.init_evn, self.init_odd = init_evn, init_odd
        if tomo_idx is None:
            return
        if hasattr(tomo_idx, "numel"):
            tomo_idx = int(tomo_idx.flatten()[0].item())
        if tomo_idx == self._tomo_idx:
            return
        evn_path, odd_path = self.evn_paths[tomo_idx], self.odd_paths[tomo_idx]
        self.physics_evn = build_one_tomography_em(
            evn_path.parent, "split1", evn_path, self.device, self.target_shape,
            self.num_operators, self.ctx, self.backend)
        self.physics_odd = build_one_tomography_em(
            odd_path.parent, "split2", odd_path, self.device, self.target_shape,
            self.num_operators, self.ctx, self.backend)
        self._tomo_idx = tomo_idx


def cached_operator_norm(vol_path: Path, volume_shape, angles, backend: str, device) -> float:
    """``||A||`` of the full operator, cached next to the FBP volume.

    Sharding does not change it, so a miss measures it on a local operator: no
    collective, so one process can measure while the rest wait on the lock.
    """
    cache = vol_path.with_suffix(f".norm_{backend}_{'x'.join(map(str, volume_shape))}.npy")
    if not cache.exists():
        with open(cache.with_suffix(".lock"), "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)   # one process measures, the rest wait
            if not cache.exists():
                # own RNG: a miss must not shift the run's random stream vs a hit
                with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
                    op = TOMOGRAPHY_BACKENDS[backend](
                        volume_shape=volume_shape, angles_deg=angles,
                        angle_sign=_TOMO_ANGLE_SIGN, normalize=True, device=str(device))
                tmp = cache.with_suffix(f".{os.getpid()}.tmp.npy")
                np.save(tmp, np.array([float(getattr(op, "xray", op).operator_norm)]))
                tmp.replace(cache)
    return float(np.load(cache)[0])


def _set_norm(ops, norm: float) -> None:
    for p in ops:
        t = getattr(p, "xray", p)   # astra holds the knobs on its wrapper
        t.operator_norm, t.normalize = norm, True


def build_one_tomography_em(
    tomo_dir: Path, split: str, vol_path: Path, device,
    target_shape: tuple[int, int, int] | None,
    num_operators: int | None = None, ctx=None,
    backend: str = "astra",
) -> TomographyEM | TomographyEMTorch | ShardedTomography:
    """``backend``: a resolved key of ``TOMOGRAPHY_BACKENDS`` — never ``"auto"``,
    which ``resolve_tomography_backend`` has already turned into one of the two.
    Everything below is backend-agnostic; only the class being instantiated moves.

    ``num_operators=None`` (default): one full operator held locally by every
    rank — no physics collective, today's behaviour.

    An int shards the tilt angles into that many operators and distributes them
    round-robin across ranks (deepinv only parallelises a *collection* of
    operators; it cannot split one, so the split is built here — same recipe as
    demo_tomo).

    Every operator is built with ``normalize=False`` and given the full operator's
    norm from ``cached_operator_norm`` — a shard's own norm would be wrong.
    """
    ang_matches = sorted(tomo_dir.glob(f"angles_*_{split}.tlt"))
    if ang_matches:
        angles = np.loadtxt(str(ang_matches[0]))
    else:
        # Some tomograms ship only the full-series tlt. The half-sets split each
        # tilt's frames (dose), not the tilt list, so both keep every angle —
        # the full series file is the right angle list for either split.
        full = [p for p in sorted(tomo_dir.glob("angles_*.tlt"))
                if not p.stem.endswith(("_split1", "_split2"))]
        if not full:
            raise FileNotFoundError(
                f"build_unrolled_physics: no angles_*_{split}.tlt in {tomo_dir}")
        angles = np.loadtxt(str(full[0]))
        print(f"[physics] {tomo_dir.name}: no angles_*_{split}.tlt, using the "
              f"{len(angles)} angles of {full[0].name}")

    if num_operators is not None:
        # The tilt count is the hard ceiling: more shards than angles would build
        # zero-angle operators, which fail in the constructor. A 64-rank job over
        # 41 angles shards into 41; the spare ranks hold no physics (deepinv
        # supports empty ranks) but still carry their denoiser tiles.
        num_operators = min(int(num_operators), len(angles))

    # canonical (Y, X, Z) -> astra (Y, Z, X); no volume is loaded here
    ty, tx, tz = target_shape if target_shape is not None else _read_mrc_vol_shape(vol_path)
    volume_shape = (int(ty), int(tz), int(tx))
    norm = cached_operator_norm(vol_path, volume_shape, angles, backend, device)
    op_cls = TOMOGRAPHY_BACKENDS[backend]

    if num_operators is None:
        physics = op_cls(
            volume_shape=volume_shape,
            angles_deg=angles,
            angle_sign=_TOMO_ANGLE_SIGN,
            normalize=False,
            device=str(device),
        )
        _set_norm([physics], norm)
        return physics

    splits = projection_splits(len(angles), int(num_operators))

    def _factory(index: int, dev, shared=None):
        start, end = splits[index]
        return op_cls(
            volume_shape=volume_shape,
            angles_deg=angles[start:end],
            angle_sign=_TOMO_ANGLE_SIGN,
            normalize=False,
            device=str(dev),
        )

    physics = ShardedTomography(ctx, int(num_operators), _factory)
    # The shards each hold a slice of the angles; carry the *global* range on the
    # container so logging reports the tomogram's real tilt range, not a shard's.
    physics._tilt_min, physics._tilt_max = float(angles.min()), float(angles.max())
    physics.n_angles_total = len(angles)   # fbp's A_i/A reweighting
    physics.volume_shape = volume_shape     # shards share it; Rotate3D reads it (run.py)
    _set_norm(physics.local_physics, norm)
    return physics
