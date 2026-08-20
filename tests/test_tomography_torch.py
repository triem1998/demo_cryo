"""Correctness tests for ``toolcryo.physics.TomographyEMTorch``.

Self-contained on purpose: no ``conftest.py``, no fixture data files, no helper
modules. Everything needed is in here, so the file can be dropped into any
checkout and run with::

    python -m pytest tests/ -q

Two groups:

**Group 1 — reference-free.** Checks laws the operator must obey on its own: a
0-degree projection *is* a sum down Z, a solid block *does* project to
``Z/cos(theta)``, ``<Ax, y>`` *does* equal ``<x, A^T y>``. These need no astra
and no GPU, so they are what runs on an AMD/ROCm box — where astra cannot run at
all, which is the reason this operator exists.

**Group 2 — parity against astra**, skipped automatically when astra is not
importable. Builds both operators live and compares them; no stored fixtures.
The conventions this pins down (Joseph-style sampling, astra's rotation being
``-theta``, ``object_cell_volume == 1``) were measured rather than guessed — see
the module docstring of ``src/toolcryo/physics/tomography_torch.py``.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from toolcryo.physics.tomography_torch import TomographyEMTorch  # noqa: E402

# --------------------------------------------------------------------------
# Tiny geometry — every test runs here, so the whole suite takes seconds.
# (n_slices, n_rows, n_cols) = astra order (Y, Z, X).
# --------------------------------------------------------------------------
SHAPE = (4, 6, 10)
ANGLES = [0.0, 25.0, -40.0, 62.0]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

astra_required = pytest.mark.skipif(
    importlib.util.find_spec("astra") is None or not torch.cuda.is_available(),
    reason="astra needs CUDA and is not importable on AMD/ROCm — parity is "
           "established on the CUDA box instead",
)


@pytest.fixture(params=DEVICES)
def device(request) -> str:
    return request.param


def make_op(device="cpu", shape=SHAPE, angles=None, **kwargs) -> TomographyEMTorch:
    return TomographyEMTorch(
        volume_shape=shape,
        angles_deg=ANGLES if angles is None else angles,
        normalize=kwargs.pop("normalize", False),
        device=device,
        **kwargs,
    )


def rand_volume(device="cpu", shape=SHAPE, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((1, 1, *shape), generator=g, dtype=torch.float64)
    return x.to(device=device, dtype=dtype)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean-removed normalised correlation, computed in float64 on CPU."""
    a = a.detach().flatten().double().cpu()
    b = b.detach().flatten().double().cpu()
    a, b = a - a.mean(), b - b.mean()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().flatten().double().cpu()
    b = b.detach().flatten().double().cpu()
    return float((a - b).norm() / (a.norm() + 1e-30))


# ==========================================================================
# Group 1a — geometry conventions
# ==========================================================================

def test_output_shape_and_contiguity(device):
    op, x = make_op(device), rand_volume(device)
    y = op.A(x)
    assert y.shape == (1, 1, SHAPE[0], len(ANGLES), SHAPE[2])
    assert y.is_contiguous()
    assert op.A_adjoint(y).shape == x.shape
    assert op.A_adjoint(y).is_contiguous()


def test_linearity(device):
    op = make_op(device)
    x1, x2 = rand_volume(device, seed=1), rand_volume(device, seed=2)
    a, b = 2.5, -0.75
    assert rel_l2(op.A(a * x1 + b * x2), a * op.A(x1) + b * op.A(x2)) < 1e-5


def test_zero_angle_equals_axis_sum(device):
    """At theta=0 the projection must be *exactly* the sum along Z.

    The sharpest check on the origin / half-pixel convention: an
    ``align_corners`` or centring mistake fails here immediately, while a
    correlation-based comparison would still look healthy.
    """
    op = make_op(device, angles=[0.0])
    x = rand_volume(device)
    assert rel_l2(op.A(x)[:, :, :, 0, :], x.sum(dim=-2)) < 1e-5


@pytest.mark.parametrize("theta", [0.0, 25.0, -40.0, 62.0])
def test_uniform_slab_chord_length(device, theta):
    """A constant volume projects to the chord length Z/|cos(theta)|."""
    v, z, n = 2, 8, 64          # wide in x so central rays stay inside
    op = TomographyEMTorch(volume_shape=(v, z, n), angles_deg=[theta],
                           normalize=False, device=device)
    y = op.A(torch.ones((1, 1, v, z, n), device=device))
    assert y[0, 0, 0, 0, n // 2].item() == pytest.approx(
        z / math.cos(math.radians(theta)), rel=1e-3)


# Angles span both sampling branches: |theta| <= 45 is z-dominant (step over
# rows), |theta| > 45 is x-dominant (step over columns) — two separate formulas
# in the module docstring, so both need this, the sharpest geometry check.
@pytest.mark.parametrize("theta", [0.0, 25.0, -40.0, 62.0, -75.0])
def test_marker_lands_at_predicted_detector_bin(device, theta):
    """A single voxel projects to the analytically predicted detector bin.

    Pins the rotation sign together with the origin, independently of astra, on
    *both* the z-dominant and x-dominant branches: astra's rotation is ``-theta``
    in this module's parameterisation, so the ray through voxel ``(z0, x0)``
    meets the detector at ``a = (x0-X/2)cos(t) + (z0-Z/2)sin(t)`` with
    ``t = -theta`` regardless of which axis the sweep steps over.
    """
    v, z, n = 1, 16, 32
    z0, x0 = 4, 9
    op = TomographyEMTorch(volume_shape=(v, z, n), angles_deg=[theta],
                           normalize=False, device=device)
    vol = torch.zeros((1, 1, v, z, n), device=device)
    vol[0, 0, 0, z0, x0] = 1.0
    y = op.A(vol)[0, 0, 0, 0]

    t = -math.radians(theta)
    a = (x0 + 0.5 - n / 2) * math.cos(t) + (z0 + 0.5 - z / 2) * math.sin(t)
    assert abs(y.argmax().item() - (a + n / 2 - 0.5)) <= 1.0
    # Mass stays local and roughly preserved. A delta is the worst case for
    # Joseph interpolation (astra loses the same ~9% at 40 degrees), so this is
    # a sanity bound, not a tight one.
    assert 0.8 < y.sum().item() < 1.2
    assert (y > 1e-6).sum().item() <= 2


def test_angle_sign_mirrors(device):
    """+theta and -theta projections are mirror images.

    Independent of the marker test: catches a wrong handedness even when the
    origin happens to be right.
    """
    v, z, n = 2, 8, 24
    op_p = TomographyEMTorch(volume_shape=(v, z, n), angles_deg=[35.0],
                             normalize=False, device=device)
    op_m = TomographyEMTorch(volume_shape=(v, z, n), angles_deg=[-35.0],
                             normalize=False, device=device)
    x = rand_volume(device, shape=(v, z, n))
    flipped = torch.flip(op_m.A(torch.flip(x, dims=(-1,))), dims=(-1,))
    assert rel_l2(op_p.A(x), flipped) < 1e-4


# ==========================================================================
# Group 1b — the adjoint and the two adjoint modes
# ==========================================================================

@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.float64, 1e-10)])
def test_adjoint_dot_product(device, dtype, tol):
    """<Ax, y> == <x, A^T y> — the definition of the adjoint.

    astra's own back-projector fails this by 6-12% (it is a voxel-driven
    approximation, not a transpose). This operator's default adjoint is exact
    because it *is* grid_sample's backward.
    """
    op, x = make_op(device), rand_volume(device, dtype=dtype)
    g = torch.Generator().manual_seed(7)
    y = torch.randn(op.A(x).shape, generator=g, dtype=torch.float64).to(device, dtype)
    lhs = (op.A(x) * y).sum().item()
    rhs = (x * op.A_adjoint(y)).sum().item()
    assert abs(lhs - rhs) / abs(lhs) < tol


def test_adjoint_mode_tradeoff(device):
    """``exact`` is a transpose, ``fast`` is astra's quicker non-transpose.

    Asserting that ``fast`` genuinely fails the identity keeps the two modes
    from silently collapsing into one, which would make the switch meaningless.
    """
    x = rand_volume(device)
    g = torch.Generator().manual_seed(11)
    y = torch.randn(make_op(device).A(x).shape, generator=g).to(device)

    errs = {}
    for mode in ("exact", "fast"):
        op = make_op(device, adjoint_mode=mode)
        lhs = (op.A(x) * y).sum().item()
        errs[mode] = abs(lhs - (x * op.A_adjoint(y)).sum().item()) / abs(lhs)
    assert errs["exact"] < 1e-5
    assert errs["fast"] > 0.01

    # A's autograd backward must follow the mode, so the gradient and A_adjoint
    # can never disagree inside one operator.
    for mode in ("exact", "fast"):
        op = make_op(device, adjoint_mode=mode)
        xg = rand_volume(device).requires_grad_(True)
        (op.A(xg) * y).sum().backward()
        assert rel_l2(xg.grad, op.A_adjoint(y)) < 1e-5


def test_default_adjoint_mode_is_exact(device):
    assert make_op(device).adjoint_mode == "exact"


# ==========================================================================
# Group 1c — gradients
# ==========================================================================

def test_gradcheck_tiny():
    """Exact float64 gradcheck on a minimal geometry (CPU only)."""
    op = make_op("cpu", shape=(2, 4, 6), angles=[0.0, 30.0])
    x = torch.randn(1, 1, 2, 4, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: op.A(t), (x,),
                                    eps=1e-6, atol=1e-8, rtol=1e-5)


def test_backward_matches_adjoint(device):
    """d/dx of <A(x), v> must equal A^T v — the gradient *is* the adjoint."""
    op = make_op(device)
    x = rand_volume(device).requires_grad_(True)
    v = torch.randn(op.A(x).shape, device=device)
    (op.A(x) * v).sum().backward()
    assert rel_l2(x.grad, op.A_adjoint(v)) < 1e-5


def test_gradients_flow_and_are_contiguous(device):
    """Every op stays differentiable, with contiguous grads.

    Contiguity is the NCCL precondition for the tiled denoiser — non-contiguous
    gradients are why ``TomographyEM.A`` contains no ``permute``. Checked here
    without needing a training run or a second GPU.
    """
    op = make_op(device)
    x = rand_volume(device).requires_grad_(True)
    op.A(x).sum().backward()
    assert x.grad.shape == x.shape and x.grad.is_contiguous()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0

    shape = op.A(rand_volume(device)).shape
    for fn in (op.A_adjoint, op.fbp):
        y = torch.randn(shape, device=device, requires_grad=True)
        fn(y).sum().backward()
        assert torch.isfinite(y.grad).all() and y.grad.abs().sum() > 0

    # One A^T A round trip — the exact shape of the PGD data-fidelity step.
    x2 = rand_volume(device).requires_grad_(True)
    op.A_adjoint(op.A(x2)).sum().backward()
    assert torch.isfinite(x2.grad).all() and x2.grad.is_contiguous()


# ==========================================================================
# Group 1d — operator contract
# ==========================================================================

def test_normalize_gives_unit_norm(device):
    op = make_op(device, normalize=True)
    assert op.normalize is True
    assert op.compute_norm(rand_volume(device), squared=False).item() == \
        pytest.approx(1.0, rel=0.05)


def test_fbp_recovers_smooth_phantom(device):
    """Dense angular sampling must reconstruct a phantom.

    Threshold 0.90, not 0.95: astra's own FBP scores 0.9244 on this phantom and
    this operator 0.9236, so 0.95 would fail both. Sharp edges plus the ramp
    filter's zero-padding cap what FBP can do.
    """
    v, z, n = 2, 64, 64
    zz, xx = torch.meshgrid(torch.arange(z).float(), torch.arange(n).float(),
                            indexing="ij")
    ph = (((zz - z / 2) ** 2 + (xx - n / 2) ** 2) < (z * 0.28) ** 2).float()
    x = ph[None, None, None].expand(1, 1, v, z, n).contiguous().to(device)
    op = TomographyEMTorch(volume_shape=(v, z, n),
                           angles_deg=np.linspace(-90, 90, 180, endpoint=False),
                           normalize=False, device=device)
    assert corr(op.fbp(op.A(x)), x) > 0.90


def test_chunking_is_exact(device):
    """Chunk sizes are a memory knob and must not change the result."""
    x = rand_volume(device)
    full = make_op(device)
    chunked = make_op(device, angle_chunk=1, slice_chunk=3, ray_chunk=2)
    assert rel_l2(full.A(x), chunked.A(x)) < 1e-6
    y = full.A(x)
    assert rel_l2(full.A_adjoint(y), chunked.A_adjoint(y)) < 1e-6


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_no_nan_under_amp(amp_dtype):
    """Inside autocast the projector stays fp32 and finite, in both AMP dtypes.

    Line integrals accumulate hundreds of terms; fp16 loses far too much, so the
    operator deliberately opts out of autocast's dtype. It has to do so by
    itself: autocast gives ``grid_sample`` no protection at all — the op is
    fallthrough and returns whatever dtype it is handed. Covers ``fbp`` and the
    gradient too, since those are the paths the training presets actually use.
    """
    if not torch.cuda.is_available():
        pytest.skip("autocast check needs CUDA")
    op, x = make_op("cuda"), rand_volume("cuda")
    x = x.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=amp_dtype):
        y = op.A(x)
        rec = op.A_adjoint(y)
        recon = op.fbp(y)
    for t in (y, rec, recon):
        assert t.dtype == torch.float32
        assert torch.isfinite(t).all()

    # The backward runs in whatever dtype the forward ran in, so an fp32 forward
    # is what keeps the gradient fp32 as well.
    (g,) = torch.autograd.grad(rec.square().sum(), x)
    assert g.dtype == torch.float32 and torch.isfinite(g).all()


def test_rejects_invalid_geometry_and_mode(device):
    """Unsupported settings must fail loudly, not produce wrong numbers."""
    with pytest.raises(NotImplementedError, match="detector row"):
        TomographyEMTorch(volume_shape=SHAPE, angles_deg=[0.0],
                          detector_shape=(7, 10), device=device)
    with pytest.raises(NotImplementedError, match="detector_spacing"):
        TomographyEMTorch(volume_shape=SHAPE, angles_deg=[0.0],
                          pixel_spacing=1.0, detector_spacing=2.0, device=device)
    with pytest.raises(ValueError, match="adjoint_mode"):
        TomographyEMTorch(volume_shape=SHAPE, angles_deg=[0.0],
                          adjoint_mode="approximate", device=device)
    # A 90-degree tilt is *not* an error: the dominant-axis sweep switches to
    # stepping over columns, so there is no degenerate angle.
    op = TomographyEMTorch(volume_shape=SHAPE, angles_deg=[90.0], device=device)
    assert torch.isfinite(op.A(torch.ones((1, 1, *SHAPE), device=device))).all()


# ==========================================================================
# Group 2 — parity with astra (live; skipped where astra cannot run)
# ==========================================================================

P_SHAPE = (16, 12, 16)
P_ANGLES = list(np.linspace(-60, 60, 9))


def _astra_op(normalize=False):
    from toolcryo.physics import TomographyEM
    return TomographyEM(volume_shape=P_SHAPE, angles_deg=P_ANGLES,
                        angle_sign=1.0, normalize=normalize, device="cuda")


def _torch_op(adjoint_mode="exact", normalize=False):
    return TomographyEMTorch(volume_shape=P_SHAPE, angles_deg=P_ANGLES,
                             angle_sign=1.0, normalize=normalize, device="cuda",
                             adjoint_mode=adjoint_mode)


def _parity_inputs():
    torch.manual_seed(0)
    x = torch.randn(1, 1, *P_SHAPE, device="cuda")
    y = torch.randn(1, 1, P_SHAPE[0], len(P_ANGLES), P_SHAPE[2], device="cuda")
    return x, y


@astra_required
def test_forward_matches_astra():
    x, _ = _parity_inputs()
    ref, got = _astra_op().A(x), _torch_op().A(x)
    assert corr(ref, got) > 0.9999
    assert rel_l2(ref, got) < 0.01


@astra_required
def test_normalize_scale_matches_astra():
    """``normalize=True`` sets the unrolled preset's PGD step scale, so the two
    backends must agree here or trained hyperparameters would not transfer."""
    x, _ = _parity_inputs()
    ref, got = _astra_op(normalize=True).A(x), _torch_op(normalize=True).A(x)
    assert got.std().item() == pytest.approx(ref.std().item(), rel=0.02)


@astra_required
def test_fbp_matches_astra():
    """``fbp`` uses the voxel-driven back-projector — the same algorithm astra
    uses for FBP — so this is near-exact, unlike ``A_adjoint``."""
    x, _ = _parity_inputs()
    sino = _astra_op().A(x)
    a, t = _astra_op().fbp(sino), _torch_op().fbp(sino)
    assert corr(a, t) > 0.999
    assert t.std().item() == pytest.approx(a.std().item(), rel=0.02)


@astra_required
def test_fast_adjoint_matches_astra_exact_does_not():
    """The two modes are exactly the astra-compatible / mathematically-correct
    pair; this pins both halves of that claim."""
    _, y = _parity_inputs()
    ref = _astra_op().A_adjoint(y)
    assert corr(ref, _torch_op("fast").A_adjoint(y)) > 0.999
    assert corr(ref, _torch_op("exact").A_adjoint(y)) < 0.999


@astra_required
def test_astra_adjoint_is_not_a_transpose_but_ours_is():
    """Documents *why* the exact adjoint deliberately differs from astra.

    astra's back-projector fails astra's own dot-product identity; the exact
    adjoint agrees with the transpose of astra's *forward*.
    """
    x, y = _parity_inputs()
    lhs = (_astra_op().A(x) * y).sum().item()
    astra_err = abs(lhs - (x * _astra_op().A_adjoint(y)).sum().item()) / abs(lhs)
    torch_err = abs(lhs - (x * _torch_op().A_adjoint(y)).sum().item()) / abs(lhs)
    assert astra_err > 0.02
    assert torch_err < 0.01
    assert torch_err < astra_err / 5


@astra_required
def test_pgd_gradient_matches_astra_in_fast_mode():
    """The unrolled data-fidelity gradient, d/dx of 0.5*||A(x)-y||^2.

    This is what ``L2().grad`` computes every PGD iteration, flowing through
    ``A``'s autograd backward. ``fast`` reproduces astra's, so a run stays
    comparable with an astra one; ``exact`` gives the true gradient instead.
    """
    x, y = _parity_inputs()

    def grad(op):
        xg = x.clone().requires_grad_(True)
        (0.5 * (op.A(xg) - y).pow(2).sum()).backward()
        return xg.grad

    ref = grad(_astra_op())
    assert corr(ref, grad(_torch_op("fast"))) > 0.999
    assert corr(ref, grad(_torch_op("exact"))) < 0.9999


@astra_required
def test_fbp_gradient_follows_adjoint_mode():
    """``adjoint_mode`` governs the gradient through ``fbp`` too.

    The fbp *output* matches astra in both modes; only the gradient depends on
    the mode. One flag for every backward pass is what makes the switch mean a
    single thing.
    """
    _, y = _parity_inputs()
    w = torch.randn(1, 1, *P_SHAPE, device="cuda")

    def grad(op):
        yg = y.clone().requires_grad_(True)
        (op.fbp(yg) * w).sum().backward()
        return yg.grad

    ref = grad(_astra_op())
    assert corr(ref, grad(_torch_op("fast"))) > 0.999
    assert corr(ref, grad(_torch_op("exact"))) < 0.999
