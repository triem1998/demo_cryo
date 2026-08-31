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

import importlib.util
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from deepinv.distributed.framework import DistributedStackedLinearPhysics
from deepinv.utils.tensorlist import TensorList

from ..utils.utils import load_mrc_volume
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

    def fbp(self, y, gather: bool = True, reduce_op: str | None = "sum", **kwargs):
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
        return self._map_reduce_gather(
            [y[i] - mean for i in self.local_indexes],
            lambda p, t, **kw: p.fbp_raw(t) * (p.n_angles / self.n_angles_total),
            gather=gather, reduce_op=reduce_op, **kwargs)


# ---------------------------------------------------------------------------
# A tomogram's two half-set (EVN/ODD) TomographyEM operators — split1/split2
# use different interleaved tilt angles, so each half gets its own operator —
# bundled with their FBP-init volumes. Method-agnostic physics: consumed by the
# unrolled and tomo_ei presets, reusable by future tomography-domain presets.
# ---------------------------------------------------------------------------

# Calibrated for this dataset's acquisition convention (see
# scripts/test_tomography_em.py) — the tilt sign matches IMOD's convention
# negated. The tilt axis is Y, which load_fbp_init puts first.
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
    init_evn: torch.Tensor
    init_odd: torch.Tensor
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

    def update(self, tomo_idx=None, **kwargs) -> None:
        """Rebuild physics_evn/odd + their FBP inits for a different tomogram.

        ``deepinv.Trainer`` calls ``physics.update(**params)`` before every
        forward pass (train and eval) — mirrors how ``MissingWedge`` rebuilds
        its wedge mask per volume (dataset_full.py), except a TomographyEM
        operator is calibrated to one tomogram's actual tilt angles, so a
        "rebuild" here means re-reading that tomogram's .tlt file + FBP
        volume, not just recomputing a mask from two floats.
        """
        if tomo_idx is None:
            return
        if hasattr(tomo_idx, "numel"):
            tomo_idx = int(tomo_idx.flatten()[0].item())
        if tomo_idx == self._tomo_idx:
            return
        evn_path, odd_path = self.evn_paths[tomo_idx], self.odd_paths[tomo_idx]
        self.physics_evn, self.init_evn = build_one_tomography_em(
            evn_path.parent, "split1", evn_path, self.device, self.target_shape,
            self.num_operators, self.ctx, self.backend)
        self.physics_odd, self.init_odd = build_one_tomography_em(
            odd_path.parent, "split2", odd_path, self.device, self.target_shape,
            self.num_operators, self.ctx, self.backend)
        # A different tomogram means different angles, hence a different global
        # norm — re-normalise so the sharded operator stays unit-norm per volume.
        if self.num_operators is not None:
            normalize_sharded(self.physics_evn, self.init_evn)
            normalize_sharded(self.physics_odd, self.init_odd)
        self._tomo_idx = tomo_idx


def measure_opnorm_sq(physics, init: torch.Tensor) -> float:
    """``||A^T A||_2`` of a distributed (sharded) operator.

    Only needed when sharding: the shards are built with ``normalize=False``
    because each one's own norm is not the full operator's.

    ``local_only=False`` runs the power iteration over the *assembled* operator,
    communicating at each step. deepinv's default (``True``) only sums the
    per-shard norms, an upper bound that grows with the shard count — which
    would make the stepsize, and so the reconstruction, depend on
    ``num_operators``. Paid once at build time, not per step.
    """
    # Full (B, C, D, H, W) init, not the unbatched form deepinv's docstring
    # suggests: the power iteration feeds x0 straight into A, and astra's
    # forward unpacks five dims.
    return float(physics.compute_sqnorm(init, local_only=False, verbose=False))


def normalize_sharded(physics, init: torch.Tensor) -> float:
    """Give a sharded operator the unit spectral norm ``normalize=True`` gives
    the unsharded one, by rescaling every shard with the *global* norm.

    Scaling only the PGD stepsize by :math:`1/\\|A\\|^2` fixes the :math:`A^{T}A` term of the 
    data-fidelity gradient but leaves the :math:`A^{T}y` term off by one factor of the norm,
    so the two paths converge to different reconstructions. Normalising the
    operator itself makes the sharded and unsharded physics identical.

    :return: the measured ``||A^T A||_2`` before normalisation (diagnostic).
    """
    sqnorm = measure_opnorm_sq(physics, init)
    for p in physics.local_physics:
        # astra holds the two knobs on its wrapper, the torch operator on itself.
        target = getattr(p, "xray", p)
        target.operator_norm = sqnorm ** 0.5
        target.normalize = True
    return sqnorm


def load_fbp_init(
    path: Path, device, target_shape: tuple[int, int, int] | None,
) -> torch.Tensor:
    """Load a precomputed FBP volume as the PGD iteration's ``x_init``.

    Physics rather than dataset code on purpose: this is set up once alongside
    the operator, not fetched per batch, and the z-normalisation below is what
    puts ``A(x)`` on the same scale as the z-normalised sinogram. The generic
    "read an MRC and reorient it" step is shared — ``utils.utils.load_mrc_volume``.
    """
    # (Y, Z, X), no crop — the reorder is a real copy done once here rather than
    # as a permute inside A()/A_adjoint(); see the TomographyEM docstring.
    vol_np = load_mrc_volume(path, order="astra")
    vol = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0)  # (1, 1, Y, Z, X)
    if target_shape is not None:
        # Local-testing only — pure downsample, no crop, kept consistent with
        # CryoEIFullDataset._resample_sinogram's matching resample of the tilt
        # series' (ny, nx).
        # target_shape is given in the canonical (Y, X, Z) order that every
        # preset's config uses; permute it to this operator's astra (Y, Z, X).
        ty, tx, tz = (int(s) for s in target_shape)
        vol = torch.nn.functional.interpolate(
            vol, size=(ty, tz, tx), mode="trilinear", align_corners=False,
        )
    # Z-normalize (centre + unit std), matching CryoEIFullDataset._load_and_prepare.
    # Centring matters as much as scaling: a volume with a non-zero mean projects
    # to a constant offset in A(x) that can never match a centred sinogram.
    return ((vol - vol.mean()) / (vol.std() + 1e-8)).to(device)


def build_one_tomography_em(
    tomo_dir: Path, split: str, vol_path: Path, device,
    target_shape: tuple[int, int, int] | None,
    num_operators: int | None = None, ctx=None,
    backend: str = "astra",
) -> tuple[TomographyEM | TomographyEMTorch, torch.Tensor]:
    """``backend``: a resolved key of ``TOMOGRAPHY_BACKENDS`` — never ``"auto"``,
    which ``resolve_tomography_backend`` has already turned into one of the two.
    Everything below is backend-agnostic; only the class being instantiated moves.

    ``num_operators=None`` (default): one full operator held locally by every
    rank — no physics collective, today's behaviour.

    An int shards the tilt angles into that many operators and distributes them
    round-robin across ranks (deepinv only parallelises a *collection* of
    operators; it cannot split one, so the split is built here — same recipe as
    demo_tomo). Shards are built with ``normalize=False``: each shard's own
    spectral norm differs from the full operator's, so per-shard normalisation
    would be wrong. The caller (``build_tomography_physics``) measures the global
    norm once and rescales every shard with it (``normalize_sharded``).
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

    init = load_fbp_init(vol_path, device, target_shape)
    volume_shape = tuple(init.shape[-3:])
    op_cls = TOMOGRAPHY_BACKENDS[backend]

    if num_operators is None:
        physics = op_cls(
            volume_shape=volume_shape,
            angles_deg=angles,
            angle_sign=_TOMO_ANGLE_SIGN,
            normalize=True,
            device=str(device),
        )
        return physics, init

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
    return physics, init
