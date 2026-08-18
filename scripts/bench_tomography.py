"""Compare the astra and pure-torch tomography backends: correctness and speed.

``TomographyEM`` (deepinv + astra-toolbox) is CUDA-only — astra ships CUDA
kernels, so it cannot run on an AMD/ROCm GPU at all. ``TomographyEMTorch`` is
plain PyTorch and runs anywhere. This script is where the two are measured
against each other.

Four modes:

``--compare`` (default)
    Correctness *and* speed for every operation — ``A``, ``A_adjoint`` in both
    adjoint modes, ``fbp``, and the PGD data-fidelity **gradient** — at a range
    of volume sizes. Needs astra, so it is a CUDA-box tool.

``--selftest``
    Reference-free invariants, PASS/FAIL. Needs no astra and no GPU, so this is
    what can be run on the AMD box. It checks laws the operator must obey on its
    own, which is what pinned the conventions in the first place: a 0-degree
    projection *is* a sum down Z, a solid block *does* project to ``Z/cos(t)``,
    ``<Ax,y>`` *does* equal ``<x,A^T y>``. (``pytest tests/`` covers the same
    ground more thoroughly; this exists for when pytest is not to hand.)

``--real-data``
    The three checks from ``scripts/test_tomography_em.py`` — forward vs. the
    real tilt series, ``fbp(A(x))`` vs. the volume, ``fbp(real split1)`` vs. the
    IMOD reference — run on both backends. Needs the empiar-11830 dataset.

``--chunk-sweep``
    Memory/speed trade-off across chunk sizes.

Two facts worth knowing when reading the output:

* ``A`` and ``fbp`` match astra closely (corr ~0.999998 / ~1.0). ``A_adjoint``
  matches only in ``adjoint_mode="fast"``. In ``"exact"`` it deliberately does
  *not*: astra's back-projector is a voxel-driven approximation that fails
  astra's own ``<Ax,y> == <x,A^T y>`` identity by 6-12%, whereas the exact mode
  is a true transpose. Matching astra there would mean copying its error.
* The gradient follows whichever adjoint mode is selected, because deepinv
  defines ``A``'s autograd backward to be ``A_adjoint``.

Run with::

    python scripts/bench_tomography.py                 # correctness + speed
    python scripts/bench_tomography.py --native        # add 1024^2 x 512
    python scripts/bench_tomography.py --selftest      # no astra needed
    python scripts/bench_tomography.py --real-data     # needs the dataset
    python scripts/bench_tomography.py --chunk-sweep
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from toolcryo.physics.tomography_torch import TomographyEMTorch  # noqa: E402

DATASET_DIR = REPO_ROOT / "dataset" / "empiar-11830" / "tomo_001"
TAG = "06022023_BrnoKrios_Arctis_xe_Position_70"

SHAPES = [(128, 64, 128), (256, 128, 256), (512, 256, 512)]
NATIVE = (1024, 512, 1024)
ANGLE_SIGN = -1.0


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _free() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def _time(fn, *args, repeats: int = 3):
    """Median wall-clock over ``repeats`` runs, after one warm-up."""
    out = fn(*args)
    _sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        _sync()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().flatten().double().cpu()
    b = b.detach().flatten().double().cpu()
    a, b = a - a.mean(), b - b.mean()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().flatten().double().cpu()
    b = b.detach().flatten().double().cpu()
    return float((a - b).norm() / (a.norm() + 1e-30))


def load_angles() -> np.ndarray:
    p = DATASET_DIR / f"angles_{TAG}.tlt"
    return np.loadtxt(str(p)) if p.exists() else np.linspace(-60, 60, 41)


def torch_op(shape, angles, device="cuda", adjoint_mode="exact", **kw):
    return TomographyEMTorch(volume_shape=shape, angles_deg=angles,
                             angle_sign=ANGLE_SIGN, normalize=False,
                             device=device, adjoint_mode=adjoint_mode, **kw)


def astra_op(shape, angles, device="cuda"):
    from toolcryo.physics import TomographyEM
    return TomographyEM(volume_shape=shape, angles_deg=angles,
                        angle_sign=ANGLE_SIGN, normalize=False, device=device)


def pgd_grad(op, x, y):
    """d/dx of 0.5*||A(x) - y||^2 — what ``L2().grad`` computes each PGD step."""
    xg = x.clone().requires_grad_(True)
    (0.5 * (op.A(xg) - y).pow(2).sum()).backward()
    return xg.grad


# ---------------------------------------------------------------------------
# 1. correctness + speed vs astra
# ---------------------------------------------------------------------------

def compare(shape, angles, device: str, repeats: int) -> None:
    """One size: every op, both backends, correctness and timing side by side.

    Operators are built one at a time and freed immediately — at native
    resolution two operators plus a volume do not fit in 8 GB.
    """
    print(f"[{shape}]  {len(angles)} angles")
    torch.manual_seed(0)
    x = torch.randn((1, 1, *shape), device=device)
    y = torch.randn((1, 1, shape[0], len(angles), shape[2]), device=device)
    rows = []

    try:
        a = astra_op(shape, angles, device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_A, ref_A = _time(a.A, x, repeats=repeats)
        t_At, ref_At = _time(a.A_adjoint, y, repeats=repeats)
        t_fbp, ref_fbp = _time(a.fbp, y, repeats=repeats)
        t_g, ref_g = _time(pgd_grad, a, x, y, repeats=repeats)
        astra_peak = _peak_gb()
        ref_A, ref_At = ref_A.clone(), ref_At.clone()
        ref_fbp, ref_g = ref_fbp.clone(), ref_g.clone()
        # Does astra agree with itself? <Ax,y> vs <x,Aty>.
        lhs = (ref_A * y).sum().item()
        astra_selfadj = abs(lhs - (x * ref_At).sum().item()) / abs(lhs)
        del a
        _free()
    except Exception as exc:                      # OOM or astra missing
        print(f"    astra unavailable at this size: {type(exc).__name__}: {exc}")
        del x, y
        _free()
        return

    for mode in ("exact", "fast"):
        try:
            t = torch_op(shape, angles, device, adjoint_mode=mode)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            tt_A, got_A = _time(t.A, x, repeats=repeats)
            tt_At, got_At = _time(t.A_adjoint, y, repeats=repeats)
            tt_fbp, got_fbp = _time(t.fbp, y, repeats=repeats)
            tt_g, got_g = _time(pgd_grad, t, x, y, repeats=repeats)
            peak = _peak_gb()
            lhs = (got_A * y).sum().item()
            selfadj = abs(lhs - (x * got_At).sum().item()) / abs(lhs)
            rows.append((mode, [
                ("A", corr(ref_A, got_A), rel_l2(ref_A, got_A), tt_A, t_A),
                ("A_adjoint", corr(ref_At, got_At), rel_l2(ref_At, got_At), tt_At, t_At),
                ("fbp", corr(ref_fbp, got_fbp), rel_l2(ref_fbp, got_fbp), tt_fbp, t_fbp),
                ("grad", corr(ref_g, got_g), rel_l2(ref_g, got_g), tt_g, t_g),
            ], selfadj, peak))
            del t, got_A, got_At, got_fbp, got_g
            _free()
        except torch.cuda.OutOfMemoryError:
            print(f"    torch {mode}: OOM")
            _free()

    hdr = f"    {'op':<10}{'mode':<7}{'corr':>10}{'relL2':>9}" \
          f"{'astra s':>10}{'torch s':>10}{'ratio':>8}"
    print(hdr)
    for mode, ops, _, _ in rows:
        for name, c, l, tt, ta in ops:
            print(f"    {name:<10}{mode:<7}{c:>10.6f}{l:>9.4f}"
                  f"{ta:>10.4f}{tt:>10.4f}{tt / ta:>7.1f}x")
    print(f"    self-adjointness  <Ax,y> vs <x,Aty>:  astra={astra_selfadj:.2e}"
          + "".join(f"  torch-{m}={s:.2e}" for m, _, s, _ in rows))
    print(f"    peak memory: astra={astra_peak:.2f}GB"
          + "".join(f"  torch-{m}={p:.2f}GB" for m, _, _, p in rows))
    print()
    del x, y, ref_A, ref_At, ref_fbp, ref_g
    _free()


# ---------------------------------------------------------------------------
# 2. reference-free self-test (no astra, no GPU needed)
# ---------------------------------------------------------------------------

def selftest(device: str) -> bool:
    """Invariants the operator must satisfy with no reference to compare to."""
    print(f"[selftest] device={device}")
    checks: list[tuple[str, bool, str]] = []

    # theta=0 must be exactly a sum down Z (origin / half-pixel convention).
    op = torch_op((4, 6, 10), [0.0], device)
    x = torch.randn(1, 1, 4, 6, 10, device=device)
    e = rel_l2(op.A(x)[:, :, :, 0, :], x.sum(dim=-2))
    checks.append(("theta=0 == sum along Z", e < 1e-5, f"rel L2 {e:.2e}"))

    # A solid block projects to the chord length Z/|cos t| (scale).
    v, z, n = 2, 8, 64
    worst = 0.0
    for th in (0.0, 25.0, -40.0, 62.0):
        o = torch_op((v, z, n), [th], device)
        got = o.A(torch.ones(1, 1, v, z, n, device=device))[0, 0, 0, 0, n // 2].item()
        worst = max(worst, abs(got - z / math.cos(math.radians(th))) /
                    (z / math.cos(math.radians(th))))
    checks.append(("uniform slab == Z/cos(theta)", worst < 1e-3, f"max rel {worst:.2e}"))

    # <Ax,y> == <x,Aty> for the exact adjoint, and *not* for fast.
    op = torch_op((4, 6, 10), [0.0, 25.0, -40.0, 62.0], device)
    x = torch.randn(1, 1, 4, 6, 10, device=device)
    y = torch.randn(op.A(x).shape, device=device)
    lhs = (op.A(x) * y).sum().item()
    e_exact = abs(lhs - (x * op.A_adjoint(y)).sum().item()) / abs(lhs)
    checks.append(("adjoint identity (exact)", e_exact < 1e-5, f"err {e_exact:.2e}"))

    fast = torch_op((4, 6, 10), [0.0, 25.0, -40.0, 62.0], device, adjoint_mode="fast")
    e_fast = abs(lhs - (x * fast.A_adjoint(y)).sum().item()) / abs(lhs)
    checks.append(("fast mode is NOT a transpose", e_fast > 0.01, f"err {e_fast:.2e}"))

    # Gradient must equal the adjoint.
    xg = x.clone().requires_grad_(True)
    (op.A(xg) * y).sum().backward()
    e = rel_l2(xg.grad, op.A_adjoint(y))
    checks.append(("gradient == A_adjoint", e < 1e-5, f"rel L2 {e:.2e}"))
    checks.append(("gradient contiguous", bool(xg.grad.is_contiguous()), ""))

    # Chunking is a memory knob only.
    chunked = torch_op((4, 6, 10), [0.0, 25.0, -40.0, 62.0], device,
                       angle_chunk=1, slice_chunk=3, ray_chunk=2)
    e = rel_l2(op.A(x), chunked.A(x))
    checks.append(("chunking changes nothing", e < 1e-6, f"rel L2 {e:.2e}"))

    ok = all(c[1] for c in checks)
    for name, passed, detail in checks:
        print(f"    [{'PASS' if passed else 'FAIL'}] {name:<32} {detail}")
    print(f"    -> {'ALL PASS' if ok else 'FAILURES PRESENT'}\n")
    return ok


# ---------------------------------------------------------------------------
# 3. real data, both backends
# ---------------------------------------------------------------------------

def real_data(device: str, target_shape=None) -> None:
    """The three checks of scripts/test_tomography_em.py, on both backends."""
    import test_tomography_em as ref                      # read-only reuse
    from toolcryo.physics import TomographyEM

    angles = np.loadtxt(str(DATASET_DIR / f"angles_{TAG}.tlt"))
    ice = ref.load_volume(DATASET_DIR / f"vol_{TAG}_Icecream.mrc")
    ts = ref.load_tilt_series(DATASET_DIR / f"tilt_series_{TAG}.mrc")
    if target_shape is not None:
        ty, tx, tz = target_shape
        ice = torch.nn.functional.interpolate(
            ice[None, None], size=(ty, tz, tx), mode="trilinear",
            align_corners=False)[0, 0]
        ts = torch.nn.functional.interpolate(
            ts.permute(2, 0, 1)[None], size=(ty, tx), mode="bilinear",
            align_corners=False)[0].permute(1, 2, 0)
    ice, ts = ice.to(device), ts.to(device)
    print(f"[real-data] volume {tuple(ice.shape)}  tilt series {tuple(ts.shape)}")

    kw = dict(volume_shape=tuple(ice.shape), angles_deg=angles,
              detector_shape=(ts.shape[0], ts.shape[1]), angle_sign=ref.ANGLE_SIGN)
    builders = {"torch": lambda: TomographyEMTorch(**kw, device=device)}
    if device == "cuda":
        builders["astra"] = lambda: TomographyEM(**kw, device=device)

    real_cmp = ts.movedim(-1, 1)
    for name, build in builders.items():
        op = build()
        with torch.no_grad():
            fwd = ref.normalized_corr(op.A(ice[None, None])[0, 0], real_cmp)
            rec = op.fbp(op.A(ice[None, None]))[0, 0]
        print(f"    {name:<6} forward A(icecream) vs real tilt series : corr={fwd:.4f}")
        print(f"    {name:<6} pipeline fbp(A(x)) vs x                 : "
              f"corr={ref.normalized_corr(rec, ice):.4f}")
        del op, rec
        _free()
    del ice, ts
    _free()

    s1_ts = ref.load_tilt_series(DATASET_DIR / f"tilt_series_{TAG}_split1.mrc")
    s1_fbp = ref.load_volume(DATASET_DIR / f"vol_{TAG}_split1_fbp_float16.mrc")
    if target_shape is not None:
        ty, tx, tz = target_shape
        s1_ts = torch.nn.functional.interpolate(
            s1_ts.permute(2, 0, 1)[None], size=(ty, tx), mode="bilinear",
            align_corners=False)[0].permute(1, 2, 0)
        s1_fbp = torch.nn.functional.interpolate(
            s1_fbp[None, None], size=(ty, tz, tx), mode="trilinear",
            align_corners=False)[0, 0]
    kw1 = dict(volume_shape=tuple(s1_fbp.shape),
               angles_deg=np.loadtxt(str(DATASET_DIR / f"angles_{TAG}_split1.tlt")),
               detector_shape=(s1_ts.shape[0], s1_ts.shape[1]),
               angle_sign=ref.ANGLE_SIGN)
    y1 = s1_ts.movedim(-1, 1).contiguous()[None, None].to(device)
    for name, build in {
        "torch": lambda: TomographyEMTorch(**kw1, device=device),
        **({"astra": lambda: TomographyEM(**kw1, device=device)} if device == "cuda" else {}),
    }.items():
        op = build()
        with torch.no_grad():
            v = op.fbp(y1)[0, 0]
        print(f"    {name:<6} backward fbp(split1) vs IMOD reference  : "
              f"corr={ref.normalized_corr(v, s1_fbp):.4f}")
        del op, v
        _free()
    print()


# ---------------------------------------------------------------------------
# 4. chunk sweep
# ---------------------------------------------------------------------------

def chunk_sweep(shape, angles, device: str) -> None:
    """Memory/speed trade-off. Chunking never changes the result (selftest)."""
    x = torch.randn((1, 1, *shape), device=device)
    print(f"[chunk-sweep] {shape}")
    print(f"    {'slice':>6} {'ray':>6} {'A (s)':>9} {'peak (GB)':>10}")
    for s_c in (16, 32, 64, 128):
        for r_c in (256, 512, shape[2]):
            try:
                op = torch_op(shape, angles, device, angle_chunk=1,
                              slice_chunk=s_c, ray_chunk=r_c)
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t, _ = _time(op.A, x, repeats=1)
                print(f"    {s_c:6d} {r_c:6d} {t:9.3f} {_peak_gb():10.2f}")
                del op
            except torch.cuda.OutOfMemoryError:
                print(f"    {s_c:6d} {r_c:6d} {'OOM':>9}")
            _free()
    del x
    _free()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", action="store_true",
                   help="correctness + speed vs astra (default when no mode given)")
    p.add_argument("--selftest", action="store_true",
                   help="reference-free invariants; needs neither astra nor a GPU")
    p.add_argument("--real-data", action="store_true",
                   help="the three empiar-11830 checks, both backends")
    p.add_argument("--chunk-sweep", action="store_true")
    p.add_argument("--native", action="store_true",
                   help=f"include the native shape {NATIVE}")
    p.add_argument("--target-shape", type=int, nargs=3, default=None,
                   metavar=("Y", "X", "Z"),
                   help="downsample the real data first, e.g. 256 256 128")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    any_mode = args.compare or args.selftest or args.real_data or args.chunk_sweep
    if not any_mode:
        args.compare = True

    angles = load_angles()

    if args.selftest:
        if not selftest(args.device):
            sys.exit(1)

    if args.compare:
        for shape in SHAPES + ([NATIVE] if args.native else []):
            compare(shape, angles, args.device,
                    repeats=1 if shape == NATIVE else args.repeats)

    if args.real_data:
        real_data(args.device,
                  tuple(args.target_shape) if args.target_shape else None)

    if args.chunk_sweep:
        chunk_sweep(NATIVE if args.native else SHAPES[-1], angles, args.device)


if __name__ == "__main__":
    main()
