"""Shared utilities for cryo-ET training.

Covers: seeding, directory helpers, CSV logging, timing (PerfProbe),
dataset discovery (_discover_pairs, select_train_val_by_name), MRC I/O, volume preprocessing,
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


# Shared column set so every FSC CSV (train/inference, full/patch) concatenates.
# fsc_curve holds the full per-shell curve as a JSON array string (index = shell);
# resolution per shell is n_ref * pixel_size / shell, derived at plot time.
FSC_CSV_COLUMNS = [
    "mode", "regime", "split", "epoch", "checkpoint", "vol_idx", "tomo",
    "pixel_size", "n_ref", "fsc_threshold", "fsc_shell", "fsc_res_angstrom",
    # Same score for the one-pass intermediate f(.), written during training
    # only, and only for presets whose recon has a real round trip. Blank
    # elsewhere. Lets one CSV track both numbers per (epoch, volume).
    "fsc_shell_1pass", "fsc_res_1pass_angstrom",
    "fsc_curve", "fsc_curve_1pass",
]


def append_fsc_row(path: Path | str, curve=None, curve_1pass=None, **fields) -> None:
    """Append one per-volume FSC record, padded to FSC_CSV_COLUMNS.

    Pass ``curve`` (the per-shell FSC array) to fill ``fsc_curve`` as JSON, and
    ``curve_1pass`` for the one-pass intermediate's curve — both land in the
    same row, so one file carries the whole pair.
    """
    if curve is not None:
        fields["fsc_curve"] = json.dumps([round(float(v), 4) for v in curve])
    if curve_1pass is not None:
        fields["fsc_curve_1pass"] = json.dumps([round(float(v), 4) for v in curve_1pass])
    append_metrics_row(path, {c: fields.get(c, "") for c in FSC_CSV_COLUMNS})


class PerfProbe:
    """Context manager that measures wall time and peak GPU memory for a code block.

    Reports *both* allocated and reserved peaks. ``max_memory_allocated`` alone
    is misleading whenever a non-PyTorch CUDA library shares the device — astra
    (``TomographyEM``) calls CUDA directly and cannot use PyTorch's cached
    blocks, so what it can allocate is bounded by total - *reserved*, not
    total - *allocated*. PyTorch's caching allocator does not return freed
    blocks to the driver by default, so ``reserved`` can sit far above
    ``allocated``.
    """
    def __enter__(self) -> "PerfProbe":
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_s: float = time.perf_counter() - self._t0
        cuda = torch.cuda.is_available()
        self.peak_mb: float = torch.cuda.max_memory_allocated() / 1e6 if cuda else 0.0
        self.peak_reserved_mb: float = torch.cuda.max_memory_reserved() / 1e6 if cuda else 0.0


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


def select_train_val_by_name(
    evn_paths: list[Path],
    odd_paths: list[Path | None],
    n_val: int,
    seed: int,
    max_train: int | None,
    train_names: list[str] | None = None,
    val_names: list[str] | None = None,
    extra: list | None = None,
) -> tuple[list, list, list, list, list, list]:
    """Select train / val volumes by name, falling back to a random split by count.

    Per set, independently: use the named tomo dirs (matched on
    ``evn_path.parent.name``) when provided, otherwise draw a random subset
    (``max_val`` / ``max_train``) from the pool of tomos not claimed by name.
    Named tomos never overlap, and the random draws come from the remaining
    pool so train and val stay disjoint.  With both name lists empty this is
    a plain random split by ``n_val`` / ``max_train``.
    """
    train_set = set(train_names or [])
    val_set   = set(val_names or [])
    names     = [p.parent.name for p in evn_paths]
    for nm in train_set | val_set:
        if nm not in names:
            print(f"[ei-data] WARNING: tomo '{nm}' not found, skipping.")

    pinned_train = [i for i, n in enumerate(names) if n in train_set]
    pinned_val   = [i for i, n in enumerate(names) if n in val_set]
    pinned       = set(pinned_train) | set(pinned_val)
    pool         = [i for i in range(len(evn_paths)) if i not in pinned]
    random.Random(seed).shuffle(pool)

    if val_set:
        val_idx = pinned_val
    else:
        n_val   = max(0, n_val)
        val_idx = pool[:n_val]
        pool    = pool[n_val:]

    if train_set:
        train_idx = pinned_train
    else:
        train_idx = pool if max_train is None else pool[:max_train]

    _extra = extra if extra is not None else [None] * len(evn_paths)
    def _take(idx):
        return ([evn_paths[i] for i in idx],
                [odd_paths[i] for i in idx],
                [_extra[i] for i in idx])
    train_evn, train_odd, train_extra = _take(train_idx)
    val_evn,   val_odd,   val_extra   = _take(val_idx)
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


# MRC files store (Z, Y, X). Two downstream orders are in use, both correct —
# this maps each to the destination axis for np.moveaxis(vol, 0, dest):
#   "native" -> (Y, X, Z): the missingwedge_ei / patch convention, the order
#       _save_mrc writes back from.
#   "astra"  -> (Y, Z, X): tilt axis first, which is what TomographyEM hands
#       straight to astra (see physics/tomography.py — keeping the reorder at
#       load time is what lets A()/A_adjoint() stay permute-free).
_MRC_AXIS_DEST = {"native": 2, "astra": 1}


def load_mrc_volume(path: Path, order: str = "native") -> np.ndarray:
    """Read an MRC volume and reorient it out of the file's (Z, Y, X) layout.

    Single source of truth for that reorientation — it was previously spelled
    out inline at every call site, so a convention change had to be applied to
    each one in lockstep. Callers keep their own resample/crop/normalise steps,
    which genuinely differ between them.

    :param Path path: MRC file to read.
    :param str order: ``"native"`` or ``"astra"`` — see ``_MRC_AXIS_DEST``.
    :return: contiguous float32 array (the copy also satisfies astra's
        ``assert data.is_contiguous()``; ``np.moveaxis`` alone returns a view).
    """
    if order not in _MRC_AXIS_DEST:
        raise ValueError(f"order must be one of {sorted(_MRC_AXIS_DEST)}, got {order!r}.")
    with mrcfile.open(str(path), permissive=True, mode="r") as mrc:
        vol_np = np.array(mrc.data, dtype=np.float32)  # (Z, Y, X)
    return np.ascontiguousarray(np.moveaxis(vol_np, 0, _MRC_AXIS_DEST[order]))


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


def _read_mrc_vol_shape(path: Path) -> tuple[int, int, int]:
    """Read an MRC header and return the native (Y, X, Z) = (D, H, W) shape
    (header-only, no data loaded) — matches ``load_mrc_volume(order="native")``."""
    with mrcfile.open(str(path), permissive=True, mode="r") as mrc:
        nx, ny, nz = int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)
    return (ny, nx, nz)


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


def fsc_resolution(fsc_curve: np.ndarray, shape, pixel_size: float,
                   threshold: float) -> tuple[int, float, int]:
    """Return (shell, resolution_angstrom, n_ref) for a volume of *shape*.

    Shell ``k`` sits at spatial frequency ``k / n_ref`` cycles/voxel, so its
    resolution is ``n_ref * pixel_size / k`` angstrom.
    """
    n_ref = int(max(shape))
    k     = fsc_shell(fsc_curve, threshold)
    return k, n_ref * pixel_size / max(k, 1), n_ref


class GpuFSC:
    """Fourier Shell Correlation computed entirely on GPU (float32).

    Works for any volume shape, cubic or not.  Shells are binned on the *true*
    spatial frequency radius: each axis index is divided by that axis' own
    length, so one step along a short axis (a large frequency step) is not
    confused with one step along a long axis (a small one)::

        f_i   = (idx_i - N_i // 2) / N_i        # cycles/voxel, in [-0.5, 0.5)
        rho   = sqrt(fz^2 + fy^2 + fx^2)
        shell = round(rho * n_ref),  n_ref = max(D, H, W)

    For a cubic volume this reduces to the plain index radius, so cubic results
    are unchanged.  Shell maps are cached per shape, so one instance may be
    reused across volumes of differing shapes.

    Args:
        device: torch device string or object.
    """

    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        self._cache: dict[tuple[int, ...], tuple[torch.Tensor, int]] = {}

    def _shell_map(self, shape: tuple[int, ...]) -> tuple[torch.Tensor, int]:
        cached = self._cache.get(shape)
        if cached is not None:
            return cached

        axes = [
            (torch.arange(n, dtype=torch.float32, device=self.device) - n // 2) / n
            for n in shape
        ]
        grid = torch.meshgrid(*axes, indexing="ij")
        rho  = torch.sqrt(sum(g * g for g in grid))

        shells = torch.round(rho * int(max(shape))).long().reshape(-1)
        rhomax = int(shells.max().item()) + 1

        self._cache[shape] = (shells, rhomax)
        return self._cache[shape]

    def __call__(self, vol1: torch.Tensor, vol2: torch.Tensor) -> np.ndarray:
        """Return FSC curve as 1-D numpy array (same format as ``FSC(a,b)[:,0]``)."""
        v1 = vol1.squeeze().to(self.device, dtype=torch.float32)
        v2 = vol2.squeeze().to(self.device, dtype=torch.float32)
        if v1.shape != v2.shape:
            raise ValueError(f"FSC needs matching shapes, got {tuple(v1.shape)} vs {tuple(v2.shape)}")

        sh, rhomax = self._shell_map(tuple(v1.shape))

        F1 = torch.fft.fftshift(torch.fft.fftn(v1))
        F2 = torch.fft.fftshift(torch.fft.fftn(v2))

        cross = (F1 * F2.conj()).real.reshape(-1)
        pow1  = (F1.real ** 2 + F1.imag ** 2).reshape(-1)
        pow2  = (F2.real ** 2 + F2.imag ** 2).reshape(-1)

        # scatter_add_ silently ignores trailing src values when the index is
        # shorter, which would bin only part of the volume against wrong radii.
        if sh.numel() != cross.numel():
            raise RuntimeError(
                f"shell map has {sh.numel()} entries but volume has {cross.numel()}"
            )

        z_ = torch.zeros(rhomax, dtype=torch.float32, device=self.device)
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
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """The two half reconstructions ``f(A(f(.)))`` — kept separate on purpose.

    Callers average them for display, but FSC must score them *apart*: it
    measures agreement between two independent half-sets, so handing it one
    pre-averaged volume is meaningless. This is also what makes the full-volume
    number comparable to patch inference, which reports FSC on exactly this
    two-pass pair (see inference/infer_patch.py::patch_inference).

    ``TomographyEMPair`` needs the round trip spelled differently: its ``A``
    maps volume -> sinogram, so a plain ``model(physics.A(v))`` would feed the
    denoiser a sinogram. Projecting through the real geometry and back via
    ``fbp`` plays the wedge mask's role — it discards exactly what the tilt
    range never measured. Duck-typed on ``physics_evn``, the same discriminator
    ``to_canonical_np``/``recon_panels`` use.
    """
    if hasattr(physics, "physics_evn"):
        pe, po = physics.physics_evn, physics.physics_odd
        return model(pe.fbp(pe.A(f_evn))), model(po.fbp(po.A(f_odd)))
    return model(physics.A(f_evn)), model(physics.A(f_odd))


def unrolled_recon(model, physics, f_evn: "torch.Tensor", f_odd: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    """The two half reconstructions for the unrolled preset — already done.

    No ``f(A(f(.)))`` round-trip as in ``half_set_recon``: there ``f`` is a
    denoiser and ``A`` a wedge mask, both volume->volume, so re-applying them is
    the icecream inference convention. Here ``f`` is the PGD net
    (sinogram -> volume) and ``A`` is volume -> sinogram, and ``f_evn``/``f_odd``
    are *already* the reconstructions — the PGD iteration
    ``x - gamma*A^T(A x - y)`` performed the measurement-consistency step
    internally, n_iter times, so the pair is returned as-is. Same signature as
    ``half_set_recon`` so it drops into ``EIFullTrainer._recon_strategy``; note
    it never calls ``model``, so unlike ``half_set_recon`` it involves no
    distributed collective.
    """
    return f_evn, f_odd


def to_canonical_np(vol: np.ndarray, physics) -> np.ndarray:
    """Bring a volume into the canonical ``(Y, X, Z)`` order for presentation.

    Tomography-physics volumes live in astra's ``(Y, Z, X)`` (tilt axis first —
    a hard astra requirement, and it must not be permuted inside
    ``A``/``A_adjoint``, see physics/tomography.py). Everything a human or a
    file ever sees is canonical instead, so figures, MRC output and
    cross-preset comparisons all share one axis order. The swap is axes 1<->2
    and happens only here, at the presentation boundary, outside autograd.

    ``physics`` is duck-typed on ``init_evn`` — the same discriminator
    ``recon_panels`` already uses to tell a TomographyEMPair from a
    MissingWedge (importing the class would make utils depend on physics).
    """
    return np.moveaxis(vol, 1, 2) if hasattr(physics, "init_evn") else vol


def recon_panels(x: "torch.Tensor", y: "torch.Tensor", physics):
    """The two 'before' panels shown next to a reconstruction.

    For missingwedge_ei ``x``/``y`` are the EVN/ODD *volumes*, so they are
    used directly. For unrolled they are *sinograms* (B,1,V,A,N) — a
    different domain and shape from the reconstructed volume — so the FBP
    init volumes on the physics container are shown instead, giving a
    FBP -> unrolled before/after. Shared by ``EIFullTrainer`` (training) and
    ``infer_full.py`` (inference) so both presets render the same way.
    """
    init_evn = getattr(physics, "init_evn", None)
    if init_evn is None:
        return (x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy(), ["EVN", "ODD"])

    def _znorm_np(arr):
        return (arr - arr.mean()) / (arr.std() + 1e-8)

    return (_znorm_np(to_canonical_np(init_evn.squeeze().cpu().numpy(), physics)),
            _znorm_np(to_canonical_np(physics.init_odd.squeeze().cpu().numpy(), physics)),
            ["FBP EVN", "FBP ODD"])


# ---------------------------------------------------------------------------
# Patch forward helper — f(A(f(.))), shared by inference and training probes
# ---------------------------------------------------------------------------

def _apply_wedge_batch(x: "torch.Tensor", wedge: "torch.Tensor") -> "torch.Tensor":
    """Apply wedge mask via FFT  (B, D, H, W) → (B, D, H, W)."""
    B, D, H, W = x.shape
    mask_shape = tuple(wedge.shape)
    X = torch.fft.fftshift(torch.fft.fftn(x, s=mask_shape, dim=(-3, -2, -1)), dim=(-3, -2, -1))
    X = X * wedge
    out = torch.fft.ifftn(torch.fft.ifftshift(X, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real
    return out[..., :D, :H, :W]


def denoise_patches(
    crops: "torch.Tensor",
    model: "torch.nn.Module",
    wedge: "torch.Tensor",
    device: "torch.device",
    amp_dtype: "torch.dtype" = None,
) -> "torch.Tensor":
    """Run ``f(A(f(.)))`` on a batch of crops — mirrors the inference forward.

    :param crops: (B, D, H, W) float tensor of ``crop_size³`` patches.
    :param wedge: wedge mask (mask_size³) applied between the two model passes.
    :param amp_dtype: autocast dtype, or ``None`` for pure fp32 (no autocast at
        all). Comes from the run's ``mixed_precision`` setting, so validation
        figures are produced in the same precision the model trained in.
    :returns: (B, D, H, W) float32 CPU tensor.  The model's train/eval mode is
        left unchanged — the caller manages it.
    """
    use_amp = device.type == "cuda" and amp_dtype is not None
    wedge_dev = wedge.to(device)
    batch = crops.to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type,
                            dtype=amp_dtype or torch.float16, enabled=use_amp):
            out = model(batch[:, None])[:, 0]
        out = _apply_wedge_batch(out.float(), wedge_dev)
        with torch.autocast(device_type=device.type,
                            dtype=amp_dtype or torch.float16, enabled=use_amp):
            out = model(out[:, None])[:, 0]
    return out.float().cpu()




