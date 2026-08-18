"""The maths behind ``ShardedTomography.fbp`` — angle-sharded filtered back-projection.

``A`` and ``A_adjoint`` shard trivially: the forward splits its *output* and the
adjoint is a plain sum of per-shard volumes, neither carrying a global constant.
``fbp`` does not, because it divides by the angle count:

    fbp_raw(y) = SUM over all A angles  bp(ramp(y_a))  *  pi / (2A)

A shard holding A_i of the A angles divides by ``A_i`` instead — the classic
"you cannot average the averages" error — so its contribution has to be
reweighted by ``A_i / A`` before the shards are summed. Its DC centring is
global for the same reason.

These tests pin that arithmetic directly on the operators, with no deepinv
distributed machinery involved: pure CPU, no GPU, no astra, no dataset, so they
also run on an AMD/ROCm box. The collective plumbing that carries these sums
across ranks is exercised by the ``num_operators`` config runs instead.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from deepinv.distributed import DistributedContext
from deepinv.distributed.framework import DistributedDataFidelity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from toolcryo.physics.tomography_build import (  # noqa: E402
    build_one_tomography_em, projection_splits, split_sinogram,
)
from toolcryo.physics.tomography_torch import TomographyEMTorch  # noqa: E402

SHAPE = (8, 6, 10)                       # (n_slices, n_rows, n_cols) = astra (Y, Z, X)
ANGLES = np.linspace(-60.0, 60.0, 7)     # 7 angles: 5 shards -> uneven 2/2/1/1/1
N_ANGLES = len(ANGLES)


def make_op(angles) -> TomographyEMTorch:
    return TomographyEMTorch(volume_shape=SHAPE, angles_deg=angles, device="cpu")


def shard_ops(n: int) -> list[TomographyEMTorch]:
    return [make_op(ANGLES[s:e]) for s, e in projection_splits(N_ANGLES, n)]


def sinogram() -> torch.Tensor:
    """A sinogram with an *angle-dependent* pedestal.

    The offset ramps with the angle index on purpose: it makes each shard's own
    mean genuinely different from the global one, which is what
    ``test_per_shard_centring_is_wrong`` needs in order to fail loudly. A
    zero-mean random sinogram would hide the bug.
    """
    torch.manual_seed(0)
    y = torch.randn(1, 1, SHAPE[0], N_ANGLES, SHAPE[2])
    return y + torch.arange(N_ANGLES, dtype=y.dtype).view(1, 1, 1, -1, 1) * 10.0


def combine(y, n, *, centre=True, weight=True) -> torch.Tensor:
    """What ``ShardedTomography.fbp`` computes, written out shard by shard."""
    ops, parts = shard_ops(n), list(split_sinogram(y, n))
    w = [(op.n_angles / N_ANGLES) if weight else 1.0 for op in ops]
    if centre:
        mean = sum(wi * t.mean(dim=(-3, -2, -1), keepdim=True) for wi, t in zip(w, parts))
        parts = [t - mean for t in parts]
    return sum(wi * op.fbp_raw(t) for wi, op, t in zip(w, ops, parts))


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("n", [2, 3])
def test_weighted_shard_sum_equals_full_fbp_raw(n):
    """The core identity: sum_i (A_i/A) * fbp_raw_i(y_i) == fbp_raw(y)."""
    y = sinogram()
    assert rel(combine(y, n, centre=False), make_op(ANGLES).fbp_raw(y)) < 1e-5


def test_global_centring_reproduces_fbp():
    """Global mean + reweighting reproduces the full operator's fbp() exactly."""
    y = sinogram()
    assert rel(combine(y, 3), make_op(ANGLES).fbp(y)) < 1e-5


def test_per_shard_centring_is_wrong():
    """Each shard centring on its own mean does *not* match — this is why the
    global mean is computed first, and why ``fbp_raw`` has to exist at all
    (subtracting a mean is shift-idempotent, so no pre-shift can fix it)."""
    y = sinogram()
    ops, parts = shard_ops(3), list(split_sinogram(y, 3))
    per_shard = sum(
        (op.n_angles / N_ANGLES) * op.fbp_raw(t - t.mean(dim=(-3, -2, -1), keepdim=True))
        for op, t in zip(ops, parts)
    )
    assert rel(per_shard, make_op(ANGLES).fbp(y)) > 1e-2


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_shard_count_invariance(n):
    """Any shard count, including uneven splits (7 angles over 5 shards), gives
    the same volume — the contract ``normalize_sharded`` already holds to."""
    y = sinogram()
    assert rel(combine(y, n), make_op(ANGLES).fbp(y)) < 1e-5


def test_missing_weight_is_detected():
    """Without A_i/A the shards are each scaled by ~A/A_i — a silent, large error."""
    y = sinogram()
    assert rel(combine(y, 3, weight=False), make_op(ANGLES).fbp(y)) > 0.5


def test_data_fidelity_reduction_is_sum():
    """L2 is ``0.5*||.||^2`` — extensive, with no 1/N — so its value *and* gradient
    shard by plain summation, unlike fbp. ``reduction="mean"`` would divide both
    by ``num_operators``, making the effective PGD stepsize depend on the shard
    count; models.py relies on the default staying ``"sum"``."""
    default = inspect.signature(DistributedDataFidelity.__init__).parameters["reduction"].default
    assert default == "sum"


# --------------------------------------------------------------------------
# Shard-count ceiling — needs the local dataset (a .tlt file to count angles),
# so it skips wherever that is absent.
# --------------------------------------------------------------------------

TOMO_DIR = Path(__file__).resolve().parent.parent / "dataset" / "empiar-11830" / "tomo_001"
FBP_VOL = TOMO_DIR / "vol_06022023_BrnoKrios_Arctis_xe_Position_70_split1_fbp_float16.mrc"


@pytest.mark.skipif(not FBP_VOL.exists(), reason="needs the local empiar-11830 dataset")
@pytest.mark.parametrize("requested,expected", [(4, 4), (64, 41), (256, 41)])
def test_shard_count_capped_by_tilt_count(requested, expected):
    """More shards than angles would build zero-angle operators, which raise in
    the constructor. The cap must also be visible on the container, because
    ``TomographyEMPair.num_operators`` drives ``split_sinogram`` (forward.py) and
    a mismatch there would hand the operator the wrong number of measurements."""
    with DistributedContext(seed=0, cleanup=True) as ctx:
        physics, _ = build_one_tomography_em(
            TOMO_DIR, "split1", FBP_VOL, "cpu", (64, 64, 32), requested, ctx, "torch")
        assert physics.num_operators == expected
        assert min(p.n_angles for p in physics.local_physics) >= 1
