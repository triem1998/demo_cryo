"""Correctness checks for ``toolcryo.physics.TomographyEM`` against real data.

Three checks, at native resolution (each saves a comparison figure):
  - forward:  A(icecream)          vs. real tilt_series_<tag>.mrc
  - pipeline: fbp(A(icecream))     vs. icecream itself (clean round-trip demo)
  - backward: fbp(real split1_ts)  vs. vol_<tag>_split1_fbp_float16.mrc (true IMOD reference)

Uses the axis/sign convention calibrated for this repo's volumes (tilt axis = Y,
loaded first by load_volume; rotation sign = -1, see ANGLE_SIGN).



Run with:
    python scripts/test_tomography_em.py                  # auto backend (astra here)
    python scripts/test_tomography_em.py --backend torch  # the pure-torch operator
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from toolcryo.physics import (  # noqa: E402
    TOMOGRAPHY_BACKENDS, TomographyEM, resolve_tomography_backend,
)
from toolcryo.utils.utils import load_mrc_volume  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset" / "empiar-11830" / "tomo_001"
TAG = "06022023_BrnoKrios_Arctis_xe_Position_70"
OUT_DIR = REPO_ROOT / "runs" / "physics_test"
DEVICE = "cuda"

# Calibrated once in an earlier axis/sign search (see git history of this
# script). The axis half of that calibration now lives in the loaders below,
# which read straight into astra's (Y, Z, X) order — TomographyEM no longer
# permutes internally (see src/toolcryo/physics/tomography.py).
ANGLE_SIGN = -1.0


# ---------------------------------------------------------------------------
# Data loading — same (Z,Y,X) -> (Y,Z,X) convention the package uses, via the
# shared helper (utils.utils.load_mrc_volume)
# ---------------------------------------------------------------------------

def load_volume(path: Path) -> torch.Tensor:
    # (ny, nz, nx) = astra (n_slices, n_rows, n_cols)
    return torch.from_numpy(load_mrc_volume(path, order="astra"))


def load_tilt_series(path: Path) -> torch.Tensor:
    with mrcfile.open(str(path), permissive=True, mode="r") as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)  # (n_angles, ny, nx)
    ts = np.moveaxis(data, 0, 2)  # (ny, nx, n_angles) = (D, H, A)
    return torch.from_numpy(np.ascontiguousarray(ts))


def to_display(img: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Z-score normalise a 2D array for display, clipped to +/- clip std.

    Reference and reconstructed volumes have very different intensity scales
    (e.g. our raw fbp has a much larger dynamic range than a denoised
    reference), so plain per-image auto-scaling makes them look falsely
    different. Normalising puts every panel on the same [-clip, clip] scale.
    """
    img = img - img.mean()
    std = img.std()
    if std > 1e-8:
        img = img / std
    return np.clip(img, -clip, clip)


def normalized_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    # Cast to double for a stable correlation, but do it on CPU: doubling a
    # multi-GB GPU volume (e.g. 1024^2x512 float32 -> float64) risks OOM.
    a = a.detach().cpu().flatten().double()
    b = b.detach().cpu().flatten().double()
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a * b).mean())


def normalized_diff_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """RMSE and MAE between ``a`` and ``b`` after z-score normalizing both.

    ``a``/``b`` live on very different intensity scales (e.g. raw fbp vs. a
    denoised reference), so normalize first — same convention as
    ``normalized_corr`` — otherwise the raw error is meaningless.
    """
    a = a.detach().cpu().flatten().double()
    b = b.detach().cpu().flatten().double()
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    diff = a - b
    rmse = float(torch.sqrt((diff ** 2).mean()))
    mae = float(diff.abs().mean())
    return rmse, mae


class Timer:
    """Wall-clock timer that syncs CUDA first, so it measures actual GPU work
    rather than the time to enqueue an async kernel."""

    def __enter__(self) -> "Timer":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# Forward check: A(ground_truth) vs. real tilt series
# ---------------------------------------------------------------------------

def check_forward(
    op: TomographyEM, icecream_vol: torch.Tensor, real_ts: torch.Tensor, save_fig: bool
) -> float:
    with torch.no_grad(), Timer() as t:
        sim_ts = op.A(icecream_vol[None, None]).squeeze(0).squeeze(0)  # (V,A,N)
    real_compare = real_ts.movedim(-1, 1)  # (D,H,A) -> (D,A,H), matches sinogram (V,A,N)
    corr = normalized_corr(sim_ts, real_compare)
    print(f"  [timing]   A (forward projection):  {t.seconds:.3f}s")
    print(f"  [forward]  A(icecream) vs real tilt series:  corr={corr:.4f}")

    if save_fig:
        n_show = 4
        show_idx = np.linspace(0, real_compare.shape[1] - 1, n_show).astype(int)
        fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
        for col, ai in enumerate(show_idx):
            axes[0, col].imshow(real_compare[:, ai, :].cpu().numpy(), cmap="gray")
            axes[0, col].set_title(f"real tilt series  #{ai}")
            axes[1, col].imshow(sim_ts[:, ai, :].cpu().numpy(), cmap="gray")
            axes[1, col].set_title(f"A(icecream)  #{ai}")
            axes[0, col].axis("off")
            axes[1, col].axis("off")
        fig.suptitle(f"Forward: A(icecream) vs real tilt series — {TAG}  corr={corr:.3f}")
        fig.tight_layout()
        out_path = OUT_DIR / f"forward_{TAG}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"    saved -> {out_path}")

    return corr


# ---------------------------------------------------------------------------
# Shared reconstruct-compare-plot helpers, used by check_backward and
# check_pipeline_demo below: both reconstruct a volume with fbp, compare it
# to a reference, and save the same truth/recon/diff figure shape -- they
# only differ in where y and the reference volume come from.
# ---------------------------------------------------------------------------

def save_comparison_figure(
    vol_a: torch.Tensor, vol_b: torch.Tensor, label_a: str, label_b: str, title: str, out_path: Path
) -> None:
    """Save a 3-row (vol_a / vol_b / diff) figure, one column per z-slice
    sampled evenly across the volume's depth.
    """
    n_show = 4
    a_cpu, b_cpu = vol_a.cpu(), vol_b.cpu()
    show_idx = np.linspace(0, a_cpu.shape[-1] - 1, n_show).astype(int)
    fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 12))
    for col, zi in enumerate(show_idx):
        a_disp = to_display(a_cpu[:, :, zi].numpy())
        b_disp = to_display(b_cpu[:, :, zi].numpy())
        axes[0, col].imshow(a_disp, cmap="gray", vmin=-3, vmax=3)
        axes[0, col].set_title(f"{label_a}  z={zi}")
        axes[1, col].imshow(b_disp, cmap="gray", vmin=-3, vmax=3)
        axes[1, col].set_title(f"{label_b}  z={zi}")
        # Difference map: same z-score scale as the panels above, so the
        # color range is directly comparable across slices/columns.
        axes[2, col].imshow(a_disp - b_disp, cmap="RdBu_r", vmin=-3, vmax=3)
        axes[2, col].set_title(f"diff  z={zi}")
        for r in range(3):
            axes[r, col].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    saved -> {out_path}")


def _reconstruct_and_compare(
    op: TomographyEM,
    y: torch.Tensor,
    reference_vol: torch.Tensor,
    ref_label: str,
    recon_label: str,
    log_desc: str,
    out_name: str,
    save_fig: bool,
) -> float:
    with torch.no_grad(), Timer() as t:
        fbp_vol = op.fbp(y).squeeze(0).squeeze(0)
    corr = normalized_corr(fbp_vol, reference_vol)
    rmse, mae = normalized_diff_stats(fbp_vol, reference_vol)
    print(f"  [timing]   fbp (filtered back-projection):  {t.seconds:.3f}s")
    print(f"  {log_desc}:  corr={corr:.4f}  rmse={rmse:.4f}  mae={mae:.4f}")

    if save_fig:
        title = f"{recon_label} vs {ref_label} — {TAG}\ncorr={corr:.3f}  rmse={rmse:.3f}  mae={mae:.3f}"
        save_comparison_figure(reference_vol, fbp_vol, ref_label, recon_label, title, OUT_DIR / out_name)

    return corr


# ---------------------------------------------------------------------------
# Backward check: fbp(y) vs. the dataset's own FBP reference, reconstructed
# from the same split1 (half-dose) tilt series.
# ---------------------------------------------------------------------------

def check_backward(
    op: TomographyEM, reference_fbp_vol: torch.Tensor, split1_ts: torch.Tensor, save_fig: bool
) -> float:
    y = split1_ts.movedim(-1, 1).contiguous()[None, None]  # (D,H,A) -> (1,1,V,A,N)
    return _reconstruct_and_compare(
        op, y, reference_fbp_vol,
        ref_label="reference split1_fbp (true, IMOD)",
        recon_label="our fbp(split1)",
        log_desc="[backward] fbp(split1) vs true IMOD fbp reference",
        out_name=f"backward_{TAG}.png",
        save_fig=save_fig,
    )


# ---------------------------------------------------------------------------
# Pipeline demo: forward-project icecream (as ground truth), reconstruct with
# fbp, and compare back to itself -- no real photos involved, just a clean
# illustration of the full A -> fbp round trip.
# ---------------------------------------------------------------------------

def check_pipeline_demo(op: TomographyEM, icecream_vol: torch.Tensor, save_fig: bool) -> float:
    with torch.no_grad():
        y = op.A(icecream_vol[None, None])
    return _reconstruct_and_compare(
        op, y, icecream_vol,
        ref_label="ground truth (icecream)",
        recon_label="fbp(A(icecream))",
        log_desc="[pipeline] fbp(A(icecream)) vs icecream itself",
        out_name=f"pipeline_demo_{TAG}.png",
        save_fig=save_fig,
    )


# ---------------------------------------------------------------------------
# Run both checks at native resolution
# ---------------------------------------------------------------------------

def run(
    angles_deg: np.ndarray,
    vol_path: Path,
    ts_path: Path,
    split1_ts_path: Path,
    split1_fbp_path: Path,
    op_cls: type = TomographyEM,
) -> None:
    icecream_vol = load_volume(vol_path).to(DEVICE)
    real_ts = load_tilt_series(ts_path).to(DEVICE)
    print(f"  volume {tuple(icecream_vol.shape)}   tilt series {tuple(real_ts.shape)}")

    detector_shape = (real_ts.shape[0], real_ts.shape[1])
    op = op_cls(
        volume_shape=tuple(icecream_vol.shape),
        angles_deg=angles_deg,
        detector_shape=detector_shape,
        angle_sign=ANGLE_SIGN,
        device=DEVICE,
    )

    check_forward(op, icecream_vol, real_ts, save_fig=True)
    torch.cuda.empty_cache()
    del real_ts
    torch.cuda.empty_cache()

    check_pipeline_demo(op, icecream_vol, save_fig=True)
    torch.cuda.empty_cache()
    del icecream_vol
    torch.cuda.empty_cache()

    split1_ts = load_tilt_series(split1_ts_path).to(DEVICE)
    reference_fbp_vol = load_volume(split1_fbp_path)  # stays on CPU: only used for comparison
    check_backward(op, reference_fbp_vol, split1_ts, save_fig=True)

    del split1_ts, reference_fbp_vol, op
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="auto", choices=["auto", "astra", "torch"],
                        help="tomography operator backend (default: auto)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA device (native-resolution volumes).")
    backend = resolve_tomography_backend(args.backend, DEVICE)
    op_cls = TOMOGRAPHY_BACKENDS[backend]
    print(f"[backend] {backend} ({op_cls.__name__})")
    # Keep astra's figures where they have always been; the torch run writes
    # alongside them so the two can be compared without overwriting.
    global OUT_DIR
    if backend != "astra":
        OUT_DIR = OUT_DIR.parent / f"physics_test_{backend}"

    angles_path = DATASET_DIR / f"angles_{TAG}.tlt"
    vol_path = DATASET_DIR / f"vol_{TAG}_Icecream.mrc"
    ts_path = DATASET_DIR / f"tilt_series_{TAG}.mrc"
    split1_ts_path = DATASET_DIR / f"tilt_series_{TAG}_split1.mrc"
    split1_fbp_path = DATASET_DIR / f"vol_{TAG}_split1_fbp_float16.mrc"
    for p in (angles_path, vol_path, ts_path, split1_ts_path, split1_fbp_path):
        if not p.exists():
            raise FileNotFoundError(p)

    angles_deg = np.loadtxt(str(angles_path))
    print(f"[data] {len(angles_deg)} tilt angles, range [{angles_deg.min():.2f}, {angles_deg.max():.2f}] deg\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run(angles_deg, vol_path, ts_path, split1_ts_path, split1_fbp_path, op_cls)


if __name__ == "__main__":
    main()
