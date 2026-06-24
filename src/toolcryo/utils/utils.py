"""Shared utilities for cryo-ET training.

Covers: seeding, directory helpers, CSV logging, model building, timing (PerfProbe),
dataset discovery (_discover_pairs, _split_pairs), MRC I/O, volume preprocessing,
and visualisation helpers (save_slice_figure, GpuFSC, FSC curves).
"""
from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_config_json(path: Path, cfg_dict: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)


def append_metrics_row(path: Path | str, row: dict) -> None:
    """Append one row to a CSV file, writing a header on first write.

    If the file already has a header, uses those fieldnames so every row has
    the same column count.  Extra keys in *row* are dropped; missing keys get
    an empty string.
    """
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] | None = None
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            fieldnames = next(csv.reader(f), None)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        if fieldnames is None:
            fieldnames = list(row.keys())
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore").writerow(row)


class DRUNetWrapper(torch.nn.Module):
    """Wraps DRUNet so model(x) works — injects a fixed sigma as a float.

    DRUNet.forward(x, sigma) requires a noise level.  Passing sigma as a
    Python float uses the 3D-safe branch: torch.ones((B,1,*x.shape[2:]))*sigma,
    unlike the tensor branch which hard-codes 2D expand calls.
    """

    def __init__(self, drunet: torch.nn.Module, sigma: float = 0.0) -> None:
        super().__init__()
        self.drunet = drunet
        self.sigma = sigma

    def forward(self, x: torch.Tensor, physics=None, **kwargs) -> torch.Tensor:
        return self.drunet(x, self.sigma)


_UNET_F_MAPS = 64
_UNET_NUM_LEVELS = 4
_DRUNET_NB = 4


def build_ei_model(
    model_type: str,
    unet_dropout: float,
    drunet_sigma: float,
    device,
) -> tuple[torch.nn.Module, str]:
    """Build IceCreamUNetWrapper (unet) or DRUNetWrapper (drunet) on *device*."""
    import deepinv as dinv
    from ..icecream_orig.models import IceCreamUNetWrapper
    from ..icecream_orig.models.unet3d_bf import UNet3D as _IceCreamUNet3D

    if model_type == "unet":
        _inner = _IceCreamUNet3D(
            in_channels=1,
            out_channels=1,
            f_maps=_UNET_F_MAPS,
            num_levels=_UNET_NUM_LEVELS,
            layer_order="cr",
            use_bias=False,
            dropout_prob=unet_dropout,
        ).to(device)
        model = IceCreamUNetWrapper(_inner)
        info = f"unet  f_maps={_UNET_F_MAPS}  num_levels={_UNET_NUM_LEVELS}  dropout={unet_dropout}"
    elif model_type == "drunet":
        _nc = tuple(_UNET_F_MAPS * (2 ** i) for i in range(4))
        _inner = dinv.models.DRUNet(
            in_channels=1,
            out_channels=1,
            nc=_nc,
            nb=_DRUNET_NB,
            pretrained="download_2d",
            pretrained_2d_isotropic=False,
            dim=3,
        ).to(device)
        model = DRUNetWrapper(_inner, sigma=drunet_sigma)
        info = f"drunet  nc={_nc}  nb={_DRUNET_NB}  sigma={drunet_sigma}  init=pretrained_2d"
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'unet' or 'drunet'.")
    return model, info


class PerfProbe:
    """Context manager that measures wall time and peak GPU memory for a code block.

    """
    def __enter__(self) -> "PerfProbe":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_s: float = time.perf_counter() - self._t0
        self.peak_mb: float = (
            torch.cuda.max_memory_allocated() / 1e6
            if torch.cuda.is_available() else 0.0
        )


@dataclass
class EIDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_sampler: object | None = None  # DistributedSampler when DDP is active


def _read_tlt(path: Path) -> tuple[float, float]:
    """Read a tlt file and return (tilt_min, tilt_max) in degrees.

    Tlt files are plain text with one floating-point angle per line.
    """
    angles = np.loadtxt(str(path))
    return float(angles.min()), float(angles.max())


def _find_tlt_for_dir(tomo_dir: Path) -> Path | None:
    """Find the full-series tlt file for a tomo directory.

    Looks for ``angles_*.tlt`` files, preferring the full-series file
    (i.e. excluding ``*_split1.tlt`` / ``*_split2.tlt``).  Falls back to
    any tlt file if no full-series file is found.
    """
    candidates = sorted(tomo_dir.glob("angles_*.tlt"))
    full_series = [
        p for p in candidates
        if not (p.stem.endswith("_split1") or p.stem.endswith("_split2"))
    ]
    if full_series:
        return full_series[0]
    if candidates:
        return candidates[0]
    return None


def _resolve_tlt_ranges(
    tlt_paths: list[Path | None],
) -> list[tuple[float, float] | None]:
    """Read tilt ranges from tlt files, returning None for missing or unreadable ones."""
    ranges: list[tuple[float, float] | None] = []
    for tlt_path in tlt_paths:
        if tlt_path is None:
            ranges.append(None)
        else:
            try:
                ranges.append(_read_tlt(tlt_path))
            except Exception as e:
                print(f"[ei-data] WARNING: could not read {tlt_path}: {e}")
                ranges.append(None)
    return ranges


def _discover_pairs(
    input_dir: Path,
    evn_glob: str = "vol*split1*.mrc",
    odd_glob: str = "vol*split2*.mrc",
) -> tuple[list[Path], list[Path | None], list[Path | None]]:
    """Discover EVN and ODD volumes, and the per-tomo tlt file if present.

    Returns three parallel lists ``(evn_paths, odd_paths, tlt_paths)``.
    ``tlt_paths[i]`` is the path to the full-series tlt file for the i-th
    tomo, or ``None`` if no tlt file was found.
    """
    evn_paths: list[Path] = []
    odd_paths: list[Path | None] = []
    tlt_paths: list[Path | None] = []

    for tomo_dir in sorted(input_dir.glob("tomo_*")):
        evn_matches = sorted(tomo_dir.glob(evn_glob))
        if not evn_matches:
            # Fallback: any *IsoNet*.mrc (old convention)
            evn_matches = sorted(tomo_dir.glob("vol*IsoNet*.mrc"))
        if not evn_matches:
            print(f"[ei-data] WARNING: no EVN volume in {tomo_dir}, skipping.")
            continue

        odd_matches = sorted(tomo_dir.glob(odd_glob))
        evn_paths.append(evn_matches[0])
        odd_paths.append(odd_matches[0] if odd_matches else None)
        tlt_paths.append(_find_tlt_for_dir(tomo_dir))

    n_paired  = sum(p is not None for p in odd_paths)
    n_evnonly = len(evn_paths) - n_paired
    n_tlt     = sum(p is not None for p in tlt_paths)
    print(
        f"[ei-data] discovered {len(evn_paths)} tomo dirs: "
        f"{n_paired} paired EVN+ODD, {n_evnonly} EVN-only, {n_tlt} with tlt files."
    )
    return evn_paths, odd_paths, tlt_paths


def _split_pairs(
    evn_paths: list[Path],
    odd_paths: list[Path | None],
    n_val: int,
    seed: int,
    max_train: int | None,
    extra: list | None = None,
) -> tuple[list, list, list, list, list, list]:
    """Shuffle and split (evn, odd, extra) lists into train / val.

    ``extra`` is any parallel list (e.g. tilt_ranges); ``None`` values are
    fine.  Returns ``(train_evn, train_odd, val_evn, val_odd, train_extra, val_extra)``.
    """
    rng = random.Random(seed)
    indices = list(range(len(evn_paths)))
    rng.shuffle(indices)
    n_val = max(0, min(n_val, len(indices) - 1))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]
    if max_train is not None:
        train_idx = train_idx[:max_train]
    _extra = extra if extra is not None else [None] * len(evn_paths)
    train_evn   = [evn_paths[i] for i in train_idx]
    train_odd   = [odd_paths[i] for i in train_idx]
    train_extra = [_extra[i] for i in train_idx]
    val_evn     = [evn_paths[i] for i in val_idx]
    val_odd     = [odd_paths[i] for i in val_idx]
    val_extra   = [_extra[i] for i in val_idx]
    return train_evn, train_odd, val_evn, val_odd, train_extra, val_extra


# ---------------------------------------------------------------------------
# MRC I/O / volume helpers
# ---------------------------------------------------------------------------

def _find_mrc(tomo_dir: Path, *globs: str) -> Path | None:
    """Return the first file matching any glob in *tomo_dir*, or None."""
    for glob in globs:
        matches = sorted(tomo_dir.glob(glob))
        if matches:
            return matches[0]
    return None


def _save_mrc(path: Path, vol_dhw: np.ndarray) -> None:
    """Save a (D, H, W) float32 numpy array as an MRC file (axis order: Z, Y, X)."""
    vol_zyx = np.moveaxis(vol_dhw.astype(np.float32), 2, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(vol_zyx)


def _read_mrc_vol_size(path: Path) -> int:
    """Read an MRC header and return the smallest spatial dimension (cubic vol side)."""
    with mrcfile.open(str(path), permissive=True, mode="r") as mrc:
        nx, ny, nz = int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)
    return min(nx, ny, nz)


def _read_pixel_sizes(
    evn_paths: list[Path],
    fallback: float | None = None,
) -> list[float]:
    """Read voxel_size.x from each EVN MRC header (header-only, no data loaded)."""
    sizes = []
    for p in evn_paths:
        with mrcfile.open(str(p), permissive=True, mode="r") as mrc:
            px = float(mrc.voxel_size.x)
        if px <= 0.0:
            px = fallback if fallback is not None else 1.0
        sizes.append(px)
    return sizes


def _center_crop(vol: np.ndarray, size: int = 512) -> np.ndarray:
    """Center-crop (D, H, W) to a cube of min(size, smallest dim)."""
    D, H, W = vol.shape
    s = min(size, D, H, W)
    d0, h0, w0 = (D - s) // 2, (H - s) // 2, (W - s) // 2
    return vol[d0:d0 + s, h0:h0 + s, w0:w0 + s]


def _znorm(vol: np.ndarray) -> np.ndarray:
    """Z-score normalise a volume in-place (returns float32)."""
    mu, sigma = float(vol.mean()), float(vol.std())
    return ((vol - mu) / (sigma + 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# FSC helpers
# ---------------------------------------------------------------------------

def fsc_shell(fsc_curve: np.ndarray, threshold: float) -> int:
    """Return first shell index where FSC drops below *threshold* (or last shell)."""
    below = np.where(fsc_curve < threshold)[0]
    return int(below[0]) if len(below) > 0 else int(len(fsc_curve) - 1)


class GpuFSC:
    """Fourier Shell Correlation computed entirely on GPU (float32).

    Matches FCC/FSC from utils_FSC.py (phiArray=[0.0] case):
      1. Build radial shell index map once at construction (cached).
      2. torch.fft.fftn + fftshift on GPU in complex64.
      3. Shell-bin numerator and denominators via scatter_add.
      4. Return FSC curve as a small 1-D numpy array.

    Args:
        vol_size: side length of the cubic volume (D = H = W).
        device:   torch device string or object.
    """

    def __init__(self, vol_size: int, device: str | torch.device = "cuda") -> None:
        self.device   = torch.device(device)
        self.vol_size = vol_size

        D    = vol_size
        half = D / 2.0
        c    = torch.arange(D, dtype=torch.float32, device=self.device) - half
        z, y, x = torch.meshgrid(c, c, c, indexing="ij")
        rho  = torch.sqrt(x * x + y * y + z * z)
        self._shells  = torch.round(rho).long().reshape(-1)
        self._rhomax  = int(np.ceil(np.sqrt(3.0) * half) + 2)

    def __call__(self, vol1: torch.Tensor, vol2: torch.Tensor) -> np.ndarray:
        """Return FSC curve as 1-D numpy array (same format as ``FSC(a,b)[:,0]``)."""
        v1 = vol1.squeeze().to(self.device, dtype=torch.float32)
        v2 = vol2.squeeze().to(self.device, dtype=torch.float32)

        F1 = torch.fft.fftshift(torch.fft.fftn(v1))
        F2 = torch.fft.fftshift(torch.fft.fftn(v2))

        cross = (F1 * F2.conj()).real.reshape(-1)
        pow1  = (F1.real ** 2 + F1.imag ** 2).reshape(-1)
        pow2  = (F2.real ** 2 + F2.imag ** 2).reshape(-1)

        sh = self._shells.clamp(0, self._rhomax - 1)
        z_ = torch.zeros(self._rhomax, dtype=torch.float32, device=self.device)
        num  = z_.clone().scatter_add_(0, sh, cross)
        den1 = z_.clone().scatter_add_(0, sh, pow1)
        den2 = z_.clone().scatter_add_(0, sh, pow2)

        denom = torch.sqrt(den1 * den2)
        fsc   = torch.where(denom > 0.0, num / denom, torch.zeros_like(num))
        return fsc.cpu().numpy()


# ---------------------------------------------------------------------------
# Self-supervised reconstruction helper
# ---------------------------------------------------------------------------

def half_set_recon(
    model: "torch.nn.Module",
    physics,
    f_evn: "torch.Tensor",
    f_odd: "torch.Tensor",
) -> "torch.Tensor":
    """Self-supervised reconstruction: 0.5 * (f(A(f_evn)) + f(A(f_odd)))."""
    return 0.5 * (model(physics.A(f_evn)) + model(physics.A(f_odd)))




