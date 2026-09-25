"""Generate a synthetic tomogram with a known ground truth: x_gt -> A -> +noise -> fbp.

The ground truth is a real tomogram's IceCream volume, z-normalised. Two
independent noise draws produce the split1/split2 half-sets the training
pipeline already expects, so the output directory is a drop-in dataset root:

    python scripts/make_synthetic_tomo.py \
        --src dataset/empiar-11830/tomo_002 --out dataset/synthetic/tomo_002

or, for many tomograms into one dataset root:

    python scripts/make_synthetic_tomo.py --list configs/synthetic_split.txt \
        --src-root /path/to/cryolithe --out-root /path/to/cryolithe/synthetic_split

Noise is Gaussian, per tilt angle, at the noise-to-signal ratio measured from
the *source* tomogram's real split1/split2 difference -- the same estimator as
EqLoss._noise_ratio, so the synthetic regime is the one the losses assume.

Caveat worth remembering when scoring runs on this data: the IceCream volume is
itself a reconstruction from these 41 angles, so x_gt already has nothing in the
missing wedge. This dataset measures denoising and data fidelity against a known
reference; it cannot score wedge recovery.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import mrcfile
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from toolcryo.physics import (  # noqa: E402
    TOMOGRAPHY_BACKENDS, resolve_tomography_backend,
)
from toolcryo.utils.utils import load_mrc_volume  # noqa: E402

# Same calibration as scripts/test_tomography_em.py: tilt axis is Y (which
# load_mrc_volume(order="astra") puts first), rotation sign is negated.
ANGLE_SIGN = -1.0


# ---------------------------------------------------------------------------
# Source-directory bookkeeping
# ---------------------------------------------------------------------------

def find_icecream(src: Path) -> Path:
    """The IceCream volume; EMPIAR-11058 ships it as an MRC named ``.tlt``."""
    for ext in ("mrc", "tlt"):
        matches = sorted(src.glob(f"vol_*[Ii]cecream*.{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no vol_*Icecream*.mrc (or .tlt) in {src}")


def resolve_tag(src: Path) -> str:
    return find_icecream(src).stem[len("vol_"):].rsplit("_", 1)[0]


def copy_angle_files(src: Path, out: Path) -> Path:
    """Copy the tlt/rawtlt files verbatim; the geometry is unchanged."""
    copied = []
    for pattern in ("angles_*.tlt", "*.rawtlt"):
        for path in sorted(src.glob(pattern)):
            shutil.copy2(path, out / path.name)
            copied.append(path.name)
    full = [p for p in sorted(out.glob("angles_*.tlt"))
            if not p.stem.endswith(("_split1", "_split2"))]
    if not full:
        raise FileNotFoundError(f"no full-series angles_*.tlt copied from {src}")
    print(f"[copy] {len(copied)} angle files -> {out}")
    return full[0]


def read_voxel_size(path: Path) -> float:
    with mrcfile.open(str(path), permissive=True, mode="r") as mrc:
        return float(mrc.voxel_size.x)


# ---------------------------------------------------------------------------
# MRC writing -- both layouts, since utils._save_mrc only does (D,H,W) float32
# ---------------------------------------------------------------------------

def save_volume(path: Path, vol_yzx: np.ndarray, dtype) -> None:
    """Write an astra-ordered (Y, Z, X) volume to the file's (Z, Y, X) layout.

    No voxel size, matching the real dataset, whose reconstructions all carry
    ``cella = 0``. It is not cosmetic: ``_read_pixel_sizes`` prefers the header
    and falls back to ``cfg.pixel_size_angstrom`` only when it is 0, so a
    populated header here would silently override the config on synthetic runs
    and report FSC resolutions on a different scale than the real ones.
    """
    vol_zyx = np.ascontiguousarray(np.moveaxis(vol_yzx, 1, 0)).astype(dtype)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(vol_zyx)
    print(f"[save] {path.name}  {vol_zyx.shape} {np.dtype(dtype).name}")


def save_tilt_series(path: Path, sino_van: np.ndarray, voxel_size: float) -> None:
    """Write a (V, A, N) sinogram to the file's (A, V, N) tilt-stack layout."""
    stack = np.ascontiguousarray(np.moveaxis(sino_van, 0, 1)).astype(np.float32)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(stack)
        if voxel_size > 0:
            mrc.voxel_size = voxel_size
    print(f"[save] {path.name}  {stack.shape} float32")


# ---------------------------------------------------------------------------
# Chunked forward / fbp -- one operator per contiguous angle range
# ---------------------------------------------------------------------------

def build_ops(volume_shape, angles, detector_shape, chunk, op_cls, device):
    n = len(angles)
    step = n if chunk in (0, None) else int(chunk)
    ranges = [(s, min(s + step, n)) for s in range(0, n, step)]
    ops = []
    for s, e in ranges:
        ops.append((s, e, op_cls(
            volume_shape=volume_shape,
            angles_deg=angles[s:e],
            detector_shape=detector_shape,
            angle_sign=ANGLE_SIGN,
            normalize=False,
            device=str(device),
        )))
    print(f"[physics] {len(ops)} operator(s) over {n} angles "
          f"(chunk={step}), volume {volume_shape}, detector {detector_shape}")
    return ops


def forward_chunked(ops, x_gpu, n_angles) -> np.ndarray:
    """A(x) assembled on CPU as (V, A, N)."""
    v, n = ops[0][2].detector_shape
    out = np.empty((v, n_angles, n), dtype=np.float32)
    with torch.no_grad():
        for s, e, op in ops:
            out[:, s:e, :] = op.A(x_gpu).squeeze(0).squeeze(0).cpu().numpy()
            torch.cuda.empty_cache()
    return out


def fbp_chunked(ops, sino_van: np.ndarray, volume_shape, device) -> np.ndarray:
    """fbp(y) assembled on CPU as (Y, Z, X).

    Same composition as ShardedTomography.fbp: centre once on the global
    sinogram, then reweight each shard's fbp_raw by its own angle share --
    fbp_raw divides by the shard's angle count, not the total.
    """
    n_total = sino_van.shape[1]
    mean = float(sino_van.mean(dtype=np.float64))
    acc = np.zeros(volume_shape, dtype=np.float32)
    with torch.no_grad():
        for s, e, op in ops:
            y = torch.from_numpy(sino_van[:, s:e, :]).to(device)[None, None]
            vol = op.fbp_raw(y - mean) * ((e - s) / n_total)
            acc += vol.squeeze(0).squeeze(0).cpu().numpy()
            del y, vol
            torch.cuda.empty_cache()
    return acc


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def per_angle_stats(stack_avn: np.ndarray) -> np.ndarray:
    """Variance over each tilt image, as float64."""
    flat = stack_avn.reshape(stack_avn.shape[0], -1)
    return flat.var(axis=1, dtype=np.float64)


def measure_nsr(ts1_path: Path, ts2_path: Path) -> np.ndarray:
    """Per-angle noise-to-signal ratio from the real half-set difference.

        var_n = var(ts1 - ts2) / 2      # the object cancels; 2 sigma^2 remains
        var_s = var(ts1) - var_n        # cryo-ET noise is a large fraction of
        r     = sqrt(var_n / var_s)     # the total, so the subtraction matters

    Always measured at native resolution: r is dimensionless and per angle, so
    it transfers to a resampled synthetic sinogram unchanged.
    """
    with mrcfile.open(str(ts1_path), permissive=True, mode="r") as m:
        ts1 = np.asarray(m.data, dtype=np.float32)
    with mrcfile.open(str(ts2_path), permissive=True, mode="r") as m:
        ts2 = np.asarray(m.data, dtype=np.float32)
    var_n = per_angle_stats(ts1 - ts2) / 2.0
    var_s = np.clip(per_angle_stats(ts1) - var_n, 1e-12, None)
    return np.sqrt(var_n / var_s)


def add_noise(sino_van: np.ndarray, sigma_a: np.ndarray, generator) -> np.ndarray:
    """y + eps, one independent draw, sigma set per tilt angle."""
    out = np.empty_like(sino_van)
    v, _, n = sino_van.shape
    for a in range(sino_van.shape[1]):
        eps = torch.randn(v, n, generator=generator, dtype=torch.float32).numpy()
        out[:, a, :] = sino_van[:, a, :] + sigma_a[a] * eps
    return out


# ---------------------------------------------------------------------------
# Verification -- streamed over the file's first axis, so nothing is ever
# materialised twice in float64
# ---------------------------------------------------------------------------

def corr_files(path_a: Path, path_b: Path, block: int = 32) -> float:
    with mrcfile.mmap(str(path_a), permissive=True, mode="r") as ma, \
         mrcfile.mmap(str(path_b), permissive=True, mode="r") as mb:
        a_d, b_d = ma.data, mb.data
        if a_d.shape != b_d.shape:
            raise ValueError(f"shape mismatch: {a_d.shape} vs {b_d.shape}")
        n = sa = sb = saa = sbb = sab = 0.0
        for i in range(0, a_d.shape[0], block):
            a = np.asarray(a_d[i:i + block], dtype=np.float64)
            b = np.asarray(b_d[i:i + block], dtype=np.float64)
            n += a.size
            sa += a.sum(); sb += b.sum()
            saa += (a * a).sum(); sbb += (b * b).sum(); sab += (a * b).sum()
    cov = sab / n - (sa / n) * (sb / n)
    va = max(saa / n - (sa / n) ** 2, 0.0)
    vb = max(sbb / n - (sb / n) ** 2, 0.0)
    return float(cov / (np.sqrt(va * vb) + 1e-20))


def psnr_files(path_ref: Path, path_test: Path, block: int = 32) -> float:
    """PSNR after z-scoring both volumes, with the reference's range as peak.

    The two live on unrelated intensity scales (a raw fbp against a normalised
    ground truth), so an un-normalised PSNR would measure the scale gap.
    """
    stats = []
    for path in (path_ref, path_test):
        with mrcfile.mmap(str(path), permissive=True, mode="r") as m:
            n = s = ss = 0.0
            for i in range(0, m.data.shape[0], block):
                blk = np.asarray(m.data[i:i + block], dtype=np.float64)
                n += blk.size; s += blk.sum(); ss += (blk * blk).sum()
        mean = s / n
        stats.append((mean, np.sqrt(max(ss / n - mean ** 2, 1e-24))))
    (mr, sr), (mt, st) = stats
    with mrcfile.mmap(str(path_ref), permissive=True, mode="r") as ma, \
         mrcfile.mmap(str(path_test), permissive=True, mode="r") as mb:
        n = se = 0.0
        lo, hi = np.inf, -np.inf
        for i in range(0, ma.data.shape[0], block):
            a = (np.asarray(ma.data[i:i + block], dtype=np.float64) - mr) / sr
            b = (np.asarray(mb.data[i:i + block], dtype=np.float64) - mt) / st
            n += a.size; se += ((a - b) ** 2).sum()
            lo, hi = min(lo, a.min()), max(hi, a.max())
    return float(10.0 * np.log10((hi - lo) ** 2 / (se / n + 1e-20)))


# ---------------------------------------------------------------------------

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=REPO_ROOT / "dataset/empiar-11830/tomo_002",
                   help="real tomogram directory: supplies the GT volume, angles and noise level")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "dataset/synthetic/tomo_002")
    p.add_argument("--nsr-scale", type=float, default=1.0,
                   help="multiplier on the measured noise-to-signal ratio (1.0 = match the real data)")
    p.add_argument("--seed", type=int, default=0,
                   help="with --list, each tomogram draws with seed + crc32(out name)")
    p.add_argument("--list", type=Path, default=None,
                   help="text file, one tomogram per line: '<src> [<out name>]', paths "
                        "relative to --src-root, out name defaults to the src basename, "
                        "'#' starts a comment. Replaces --src/--out; tomograms whose "
                        "GENERATION.json already exists are skipped")
    p.add_argument("--src-root", type=Path, default=REPO_ROOT / "dataset")
    p.add_argument("--out-root", type=Path, default=REPO_ROOT / "dataset/synthetic")
    p.add_argument("--angle-chunk", type=int, default=0,
                   help="angles per operator; 0 = one operator over all angles")
    p.add_argument("--target-shape", type=int, nargs=3, default=None, metavar=("Y", "X", "Z"),
                   help="downsample the GT to this (Y, X, Z) first -- for a cheap dry run")
    p.add_argument("--backend", default="auto", choices=["auto", *TOMOGRAPHY_BACKENDS])
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-clean-fbp", action="store_true",
                   help="also write fbp(A(x_gt)), the noiseless reference reconstruction. "
                        "Off by default: it is an extra 1 GB with no counterpart in the "
                        "real dataset, and --check reports its number without keeping it")
    p.add_argument("--check", action="store_true",
                   help="after writing, report correlations against the GT and the real tomogram")
    return p.parse_args()


def read_list(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line in path.read_text().splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        src = fields[0]
        entries.append((src, fields[1] if len(fields) > 1 else Path(src).name))
    names = [name for _, name in entries]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"{path}: duplicate out names {dups} -- give them distinct names")
    return entries


def main() -> None:
    args = parse_args()
    if args.list is None:
        generate_one(args.src.resolve(), args.out.resolve(), args.seed, args)
        return

    entries = read_list(args.list)
    print(f"[list] {len(entries)} tomograms from {args.list}")
    failed = []
    for i, (src_rel, name) in enumerate(entries, 1):
        src, out = (args.src_root / src_rel).resolve(), (args.out_root / name).resolve()
        print(f"\n===== [{i}/{len(entries)}] {src_rel} -> {name} =====")
        if (out / "GENERATION.json").exists():
            print(f"[skip] {out} already has GENERATION.json")
            continue
        try:
            generate_one(src, out, args.seed + zlib.crc32(name.encode()), args)
        except Exception as e:  # keep going; the rest of the list is independent
            print(f"[error] {name}: {type(e).__name__}: {e}")
            failed.append(name)
        gc.collect()
        torch.cuda.empty_cache()
    if failed:
        sys.exit(f"[list] {len(failed)} failed: {failed}")
    print(f"[list] done, {len(entries)} tomograms in {args.out_root}")


def generate_one(src: Path, out: Path, seed: int, args: argparse.Namespace) -> None:
    tag = resolve_tag(src)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[tag] {tag}\n[src] {src}\n[out] {out}")

    tlt_path = copy_angle_files(src, out)
    angles = np.loadtxt(str(tlt_path))
    print(f"[angles] {len(angles)} tilts, [{angles.min():.2f}, {angles.max():.2f}] deg")

    ts_path = src / f"tilt_series_{tag}.mrc"
    voxel_size = read_voxel_size(ts_path)

    # -- ground truth -------------------------------------------------------
    gt_src = find_icecream(src)
    gt = torch.from_numpy(load_mrc_volume(gt_src, order="astra"))
    gt = gt[None, None]                                    # (1, 1, Y, Z, X)
    if args.target_shape is not None:
        ty, tx, tz = args.target_shape                     # config order -> astra order
        gt = torch.nn.functional.interpolate(
            gt, size=(ty, tz, tx), mode="trilinear", align_corners=False)
    gt = (gt - gt.mean()) / (gt.std() + 1e-8)
    volume_shape = tuple(int(s) for s in gt.shape[-3:])
    detector_shape = (volume_shape[0], volume_shape[2])
    gt_path = out / f"vol_{tag}_gt.mrc"
    save_volume(gt_path, gt.squeeze(0).squeeze(0).numpy(), np.float32)

    # -- forward ------------------------------------------------------------
    backend = resolve_tomography_backend(args.backend, args.device)
    op_cls = TOMOGRAPHY_BACKENDS[backend]
    print(f"[backend] {backend}")
    ops = build_ops(volume_shape, angles, detector_shape, args.angle_chunk, op_cls, args.device)

    gt_gpu = gt.to(args.device)
    y_clean = forward_chunked(ops, gt_gpu, len(angles))
    del gt_gpu, gt
    torch.cuda.empty_cache()
    print(f"[forward] A(x_gt) {y_clean.shape}  std={y_clean.std():.4g}")

    # -- noise --------------------------------------------------------------
    nsr = measure_nsr(src / f"tilt_series_{tag}_split1.mrc",
                      src / f"tilt_series_{tag}_split2.mrc")
    sigma = args.nsr_scale * nsr * y_clean.std(axis=(0, 2), dtype=np.float64)
    print(f"[noise] measured NSR per angle: min={nsr.min():.3f} "
          f"median={np.median(nsr):.3f} max={nsr.max():.3f}  (x{args.nsr_scale})")

    gen = torch.Generator().manual_seed(seed)
    y1 = add_noise(y_clean, sigma, gen)
    y2 = add_noise(y_clean, sigma, gen)
    save_tilt_series(out / f"tilt_series_{tag}_split1.mrc", y1, voxel_size)
    save_tilt_series(out / f"tilt_series_{tag}_split2.mrc", y2, voxel_size)
    save_tilt_series(out / f"tilt_series_{tag}.mrc", y1 + y2, voxel_size)

    # -- reconstructions ----------------------------------------------------
    for split, y in (("split1", y1), ("split2", y2)):
        vol = fbp_chunked(ops, y, volume_shape, args.device)
        save_volume(out / f"vol_{tag}_{split}_fbp_float16.mrc", vol, np.float16)
        del vol
    if args.save_clean_fbp:
        vol = fbp_chunked(ops, y_clean, volume_shape, args.device)
        save_volume(out / f"vol_{tag}_clean_fbp_float16.mrc", vol, np.float16)
        del vol

    meta = {
        "source_dir": str(src), "tag": tag, "git_sha": git_sha(),
        "ground_truth": f"{gt_src.name}, z-normalised whole-volume",
        "volume_shape_astra_yzx": list(volume_shape), "detector_shape_vn": list(detector_shape),
        "target_shape_yxz": args.target_shape, "angle_sign": ANGLE_SIGN,
        "backend": backend, "normalize": False,
        "noise": "gaussian, per tilt angle, sigma[a] = nsr_scale * r[a] * std(A(x_gt)[a])",
        "nsr_scale": args.nsr_scale, "seed": seed,
        "nsr_per_angle": [round(float(v), 6) for v in nsr],
        "sigma_per_angle": [round(float(v), 6) for v in sigma],
        "voxel_size_angstrom": voxel_size,
    }
    (out / "GENERATION.json").write_text(json.dumps(meta, indent=2))
    print(f"[save] GENERATION.json")

    if args.check:
        run_checks(out, src, tag, y1, y2, args)


def run_checks(out: Path, src: Path, tag: str, y1, y2, args) -> None:
    print("\n=== checks ===")
    remeasured = np.sqrt(
        (per_angle_stats(np.moveaxis(y1 - y2, 0, 1)) / 2.0)
        / np.clip(per_angle_stats(np.moveaxis(y1, 0, 1))
                  - per_angle_stats(np.moveaxis(y1 - y2, 0, 1)) / 2.0, 1e-12, None))
    target = args.nsr_scale * measure_nsr(src / f"tilt_series_{tag}_split1.mrc",
                                          src / f"tilt_series_{tag}_split2.mrc")
    print(f"  NSR round-trip   target median={np.median(target):.4f}  "
          f"remeasured={np.median(remeasured):.4f}  "
          f"max|rel err|={np.abs(remeasured / target - 1).max():.3f}")

    gt_p = out / f"vol_{tag}_gt.mrc"
    s1_p = out / f"vol_{tag}_split1_fbp_float16.mrc"
    s2_p = out / f"vol_{tag}_split2_fbp_float16.mrc"
    clean_p = out / f"vol_{tag}_clean_fbp_float16.mrc"

    if clean_p.exists():
        print(f"  corr(fbp(A x_gt), gt)   = {corr_files(gt_p, clean_p):+.4f}   "
              f"psnr={psnr_files(gt_p, clean_p):.2f} dB   <- noiseless ceiling")
    print(f"  corr(fbp(y_1),    gt)   = {corr_files(gt_p, s1_p):+.4f}   "
          f"psnr={psnr_files(gt_p, s1_p):.2f} dB   <- baseline to beat")
    print(f"  corr(fbp(y_1), fbp(y_2))= {corr_files(s1_p, s2_p):+.4f}")

    real1 = src / f"vol_{tag}_split1_fbp_float16.mrc"
    real2 = src / f"vol_{tag}_split2_fbp_float16.mrc"
    if real1.exists() and real2.exists() and args.target_shape is None:
        print(f"  real half-set corr      = {corr_files(real1, real2):+.4f}   "
              f"<- synthetic difficulty matches when these two agree")


if __name__ == "__main__":
    main()
