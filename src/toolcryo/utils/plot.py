"""Plotting utilities for cryo-ET training.

  plot_metrics               — training curve figures from CSV files
  save_fsc_figure            — per-volume FSC curve PNG
  save_resolution_histogram  — histogram of FSC resolutions across volumes
  save_slice_figure          — orthogonal mid-slice grid PNG

Usage (standalone):
    python src/utils/plot.py --run-dir runs/<name>/
    python src/utils/plot.py --run-dir runs/<name>/ --save summary.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metrics(run_dir: Path, save: Path | str | None = None) -> None:
    """Load CSVs from *run_dir*/metrics/ and save (or show) a summary figure."""
    metrics_dir = Path(run_dir) / "metrics"
    if not metrics_dir.exists():
        print(f"[plot_metrics] no metrics/ dir found in {run_dir}, skipping.")
        return

    train_csv = metrics_dir / "train_epochs.csv"
    val_csv = metrics_dir / "val_epochs.csv"

    train_df = pd.read_csv(train_csv) if train_csv.exists() else pd.DataFrame()
    val_df = pd.read_csv(val_csv) if val_csv.exists() else pd.DataFrame()

    # Detect individual loss columns (exclude meta / total / PSNR / FSC columns)
    _skip = {"epoch", "lr", "step", "gradient_norm", "TotalLoss"}
    individual_loss_cols = [
        c for c in train_df.columns
        if c not in _skip
        and "psnr" not in c.lower()
        and "fsc" not in c.lower()
        and not train_df.empty
    ]
    # Layout: [losses (total + components) | PSNR]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training summary", fontsize=13)

    # ── Train losses (total + components) in log scale ──────────────────────
    ax = axes[0]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx = 0
    if not train_df.empty and "TotalLoss" in train_df.columns:
        ax.plot(train_df["epoch"], train_df["TotalLoss"], "o-",
                color=colors[color_idx], linewidth=2, label="TotalLoss")
        color_idx += 1
    for col in individual_loss_cols:
        ax.plot(train_df["epoch"], train_df[col], "s--",
                color=colors[color_idx % len(colors)], label=col)
        color_idx += 1
    ax.set_yscale("log")
    ax.set_title("Train loss (log scale)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # ── Val metric: FSC (equivariant) or PSNR (supervised) ─────────────────
    ax = axes[1]
    fsc_col_val  = next((c for c in val_df.columns  if "fsc"  in c.lower()), None)
    psnr_col_val = next((c for c in val_df.columns  if "psnr" in c.lower()), None)
    psnr_col_trn = next((c for c in train_df.columns if "psnr" in c.lower()), None)

    if not val_df.empty and fsc_col_val:
        # ── Equivariant: FSC resolution in Å ──
        epochs = val_df["epoch"]
        ax.plot(epochs, val_df[fsc_col_val], "s-",
                color="steelblue", linewidth=2, label="mean")
        median_col = next((c for c in val_df.columns if "median" in c.lower()), None)
        q1_col = next((c for c in val_df.columns if "q1" in c.lower()), None)
        q3_col = next((c for c in val_df.columns if "q3" in c.lower()), None)
        if median_col:
            ax.plot(epochs, val_df[median_col], "D-", color="mediumpurple",
                    linewidth=1.5, alpha=0.9, label="median")
        if q1_col and q3_col:
            ax.plot(epochs, val_df[q1_col], "--", color="steelblue",
                    linewidth=1, alpha=0.7, label="Q1")
            ax.plot(epochs, val_df[q3_col], ":",  color="steelblue",
                    linewidth=1, alpha=0.7, label="Q3")
            ax.fill_between(epochs, val_df[q1_col], val_df[q3_col],
                            color="steelblue", alpha=0.15, label="Q1–Q3")
        ax.set_title("Val FSC resolution @ 0.143 threshold (↓ better)")
        ax.set_ylabel("Resolution (Å)")
    elif (not val_df.empty and psnr_col_val) or (not train_df.empty and psnr_col_trn):
        # ── Supervised: PSNR ──
        if not train_df.empty and psnr_col_trn:
            ax.plot(train_df["epoch"], train_df[psnr_col_trn], "o-",
                    color="darkorange", label="Train")
        if not val_df.empty and psnr_col_val:
            ax.plot(val_df["epoch"], val_df[psnr_col_val], "s--",
                    color="seagreen", label="Val")
        ax.set_title("PSNR")
        ax.set_ylabel("PSNR (dB)")
    else:
        ax.text(0.5, 0.5, "No val metric data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("Val metric")
        ax.set_ylabel("")

    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot_metrics] saved to {out}")
    else:
        plt.show()

    plt.close(fig)


def save_fsc_figure(
    images_dir: Path,
    epoch: int,
    fname: str,
    fsc_curve: np.ndarray,
    res_shell: int,
    res_angstrom: float,
    title: str,
    threshold: float = 0.143,
    vol_size: int | None = None,
    pixel_size: float | None = None,
) -> None:
    """Save an FSC curve PNG with threshold and resolution marker lines."""
    n = len(fsc_curve)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(fsc_curve, lw=1.5)
    ax.axhline(threshold, color="r", ls="--", label=f"thr={threshold}")
    if vol_size and pixel_size:
        tick_shells = np.linspace(max(1, n // 8), n - 1, num=8, dtype=int)
        ax.axvline(res_shell, color="orange", ls=":", label=f"{res_angstrom:.1f} Å (shell {res_shell})")
        ax.set_xticks(tick_shells)
        ax.set_xticklabels([f"{vol_size * pixel_size / k:.1f}" for k in tick_shells],
                           rotation=30, ha="right", fontsize=7)
        ax.set_xlabel("Resolution (Å)")
    else:
        ax.axvline(res_shell, color="orange", ls=":", label=f"shell={res_shell} ({res_angstrom:.1f} Å)")
        ax.set_xlabel("Shell index")
    ax.set_ylabel("FSC")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    out = Path(images_dir) / f"fsc_epoch{epoch:04d}" / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def save_resolution_histogram(
    images_dir: Path,
    epoch: int,
    resolutions_angstrom: list[float],
    mean_res: float,
    median_res: float,
    q1_res: float,
    q3_res: float,
    threshold_label: str = "0.143",
) -> None:
    """Save a histogram of per-volume FSC resolutions (Å)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    n = len(resolutions_angstrom)
    ax.hist(resolutions_angstrom, bins=max(5, n // 2 + 1), color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(mean_res,   color="tomato",      ls="-",  lw=1.8, label=f"mean   = {mean_res:.1f} Å")
    ax.axvline(median_res, color="mediumpurple", ls="-",  lw=1.8, label=f"median = {median_res:.1f} Å")
    ax.axvline(q1_res,     color="goldenrod",   ls="--", lw=1.4, label=f"Q1     = {q1_res:.1f} Å")
    ax.axvline(q3_res,     color="goldenrod",   ls=":",  lw=1.4, label=f"Q3     = {q3_res:.1f} Å")
    ax.set_xlabel("FSC resolution (Å)  —  lower is better")
    ax.set_ylabel("# volumes")
    ax.set_title(f"Epoch {epoch} | FSC@{threshold_label} resolution ({n} vols)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(images_dir) / f"fsc_epoch{epoch:04d}" / "resolution_histogram.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def save_slice_figure(
    images_dir: Path,
    epoch: int,
    vol_idx: int,
    cols: list[np.ndarray],
    labels: list[str] | None = None,
    title: str | None = None,
    subdir: str | None = None,
    fname: str | None = None,
) -> None:
    """Save a 3×N PNG with all three orthogonal mid-slices (XY / XZ / YZ)."""
    def _slices(v: np.ndarray, max_px: int = 256) -> list[np.ndarray]:
        d, h, w = v.shape
        slcs = [v[d // 2, :, :], v[:, h // 2, :], v[:, :, w // 2]]
        out = []
        for s in slcs:
            factor = max(1, max(s.shape) // max_px)
            out.append(s[::factor, ::factor].astype(np.float32))
        return out

    n_cols = len(cols)
    if labels is None:
        labels = [f"Col {i}" for i in range(n_cols)]
    planes = ["XY (z-mid)", "XZ (y-mid)", "YZ (x-mid)"]
    pre = [_slices(v) for v in cols]
    row_ranges = [
        (float(np.percentile(np.concatenate([s[r].ravel() for s in pre]), 1)),
         float(np.percentile(np.concatenate([s[r].ravel() for s in pre]), 99)))
        for r in range(3)
    ]
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    for row, plane in enumerate(planes):
        vmin, vmax = row_ranges[row]
        for col, (label, slc_list) in enumerate(zip(labels, pre)):
            ax = axes[row, col]
            ax.imshow(slc_list[row], cmap="gray", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(plane)
    fig.suptitle(title if title is not None else f"Epoch {epoch} | Vol {vol_idx}")
    fig.tight_layout()
    folder = subdir if subdir is not None else f"fsc_epoch{epoch:04d}"
    out = Path(images_dir) / folder / (fname if fname is not None else f"vol{vol_idx:02d}_slices.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cryo training/val metrics.")
    parser.add_argument(
        "--run-dir", required=True, help="Run directory (containing metrics/)"
    )
    parser.add_argument(
        "--save", default=None, help="Save figure to this path (e.g. summary.png)"
    )
    args = parser.parse_args()
    plot_metrics(Path(args.run_dir), save=args.save)


if __name__ == "__main__":
    main()
