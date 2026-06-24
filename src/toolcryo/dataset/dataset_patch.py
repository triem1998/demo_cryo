"""CryoEIPatchDataset — patch dataset for equivariant imaging on cryo-ET half-sets.

Yields paired (evn_patch, odd_patch) random cubic crops from the same spatial
location in EVN and ODD half-set MRC volumes, following the same discovery and
normalisation conventions as CryoEIFullDataset.

Differences from the full-volume variant:
  - ``__getitem__`` returns the same 3-tuple ``(evn_patch, odd_patch, tilt_params)``
    so it plugs directly into EIPatchTrainer without changes.
  - Patches are random cubic crops of side ``crop_size`` extracted from the same
    coordinates in both EVN and ODD; this preserves the cross half-set pairing.
  - ``__len__`` = len(evn_paths) * n_crops_per_vol (virtual epoch length).
  - Volumes are memory-mapped: only the OS pages covering each requested crop
    (~1–3 MB) are read from disk per __getitem__ call.  The full volume is
    never copied into RAM unless explicitly requested (e.g. inference).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mrcfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.utils import EIDataBundle, _discover_pairs, _resolve_tlt_ranges, _split_pairs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EIPatchDataConfig:
    input_dir: str = "./dataset/empiar-11058"
    crop_size: int = 72
    n_crops_per_vol: int = 10          # virtual epoch = n_vols * n_crops_per_vol
    batch_size: int = 4
    num_workers: int = 1
    pin_memory: bool = True
    prefetch_factor: int = 1
    persistent_workers: bool = True
    max_train_vols: int | None = None
    max_val_vols: int = 5
    seed: int = 0
    normalize: bool = True             # zero-mean, unit-std per patch
    # Glob patterns — same as CryoEIFullDataset
    evn_glob: str = "vol*split1*.mrc"
    odd_glob: str = "vol*split2*.mrc"
    # Fallback tilt range when no tlt file is found.
    fallback_tilt_min: float = -60.0
    fallback_tilt_max: float = 60.0


# ---------------------------------------------------------------------------
# Per-process mmap cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _open_mrc_mmap(path_str: str) -> tuple:
    """Open an MRC file as a memory-mapped (D, H, W) array; cache the handle.

    Returns ``(mrc_handle, vol_ndarray)`` where ``vol_ndarray`` is a strided
    float32 view of shape (D, H, W) = (Y, X, Z) over the on-disk data.
    The OS reads only the pages corresponding to whatever region is sliced —
    the full volume is never copied into RAM by this call alone.

    Both objects are cached so the mapping stays alive across calls and file
    descriptors are not repeatedly opened.  The cache is per-process, so each
    DataLoader worker maintains its own independent cache.
    """
    mrc = mrcfile.mmap(path_str, permissive=True, mode='r')
    data = mrc.data  # numpy.memmap, shape (Z, Y, X)
    if data.dtype != np.float32:
        # Uncommon: force conversion only when needed (reads whole volume once)
        data = data.astype(np.float32)
    # moveaxis creates a non-contiguous view — no data pages are read here
    vol = np.moveaxis(data, 0, 2)   # (Z, Y, X) → (Y, X, Z) = (D, H, W)
    return mrc, vol


# ---------------------------------------------------------------------------
# Lazy volume list — inference compatibility
# ---------------------------------------------------------------------------

class _LazyVolList:
    """List-like proxy that loads MRC volumes as CPU torch.Tensor on demand.

    Each ``__getitem__`` call loads the full volume into RAM
    (acceptable for inference, which accesses each volume once).
    """

    def __init__(self, paths: list[Path | None], normalize: bool) -> None:
        self._paths = paths
        self._normalize = normalize

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, i: int) -> torch.Tensor | None:
        p = self._paths[i]
        if p is None:
            return None
        _, vol_np = _open_mrc_mmap(str(p))
        vol_t = torch.from_numpy(np.ascontiguousarray(vol_np))  # reads all pages
        if self._normalize:
            vol_t = (vol_t - vol_t.mean()) / (vol_t.std() + 1e-8)
        return vol_t

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CryoEIPatchDataset(Dataset):
    """Yields ``(evn_patch, odd_patch, tilt_params)`` random cubic crops.

    Both patches are cropped from the **same random spatial position** so the
    cross half-set pairing is preserved — ObsLoss and EqLoss can compare them
    exactly as they compare full volumes in CryoEIFullDataset.

    Volumes are memory-mapped at the OS level.  Each ``__getitem__`` call
    reads only the ~1–3 MB of disk pages covering the requested 72³ crop.
    Repeated access to the same region within a worker is served from the OS
    page cache (no disk I/O after the first touch).

    Normalisation (zero-mean, unit-std per patch) is applied independently to
    each patch after cropping when ``normalize=True``.

    When only EVN is available, ``odd_patch`` is a copy of ``evn_patch`` so
    the single-half ObsLoss fallback ``L = fourier_loss(y, f(y), wedge)``
    still works.

    :param list[Path] evn_paths: Paths to EVN half-set MRC volumes.
    :param list[Path | None] odd_paths: Paths to ODD half-set MRC volumes (or None).
    :param int crop_size: Cubic crop side length (default 72).
    :param int n_crops_per_vol: Virtual crops per volume per epoch (default 10).
    :param bool normalize: Standardise each patch independently (default True).
    :param list tilt_ranges: Per-volume (tilt_min, tilt_max) or None.
    :param float fallback_tilt_min: Used when tilt_ranges[i] is None.
    :param float fallback_tilt_max: Used when tilt_ranges[i] is None.
    """

    def __init__(
        self,
        evn_paths: list[Path],
        odd_paths: list[Path | None],
        crop_size: int = 72,
        n_crops_per_vol: int = 10,
        normalize: bool = False,
        tilt_ranges: list[tuple[float, float] | None] | None = None,
        fallback_tilt_min: float = -60.0,
        fallback_tilt_max: float = 60.0,
    ) -> None:
        assert len(evn_paths) == len(odd_paths)
        self.evn_paths         = evn_paths
        self.odd_paths         = odd_paths
        self.crop_size         = crop_size
        self.n_crops_per_vol   = n_crops_per_vol
        self.normalize         = normalize
        self.fallback_tilt_min = fallback_tilt_min
        self.fallback_tilt_max = fallback_tilt_max
        self._tilt_ranges: list[tuple[float, float] | None] = (
            tilt_ranges if tilt_ranges is not None else [None] * len(evn_paths)
        )

        # Lazy proxies keep the ds.evn_vols[i] / ds.odd_vols[i] interface that
        # the inference loop in run_ei_patch.py relies on.  No data is read here.
        self.evn_vols = _LazyVolList(evn_paths, normalize)
        self.odd_vols = _LazyVolList(odd_paths, normalize)

        n_paired = sum(p is not None for p in odd_paths)
        n_tlt    = sum(t is not None for t in self._tilt_ranges)
        print(
            f"[ei-patch] CryoEIPatchDataset: {len(evn_paths)} vols "
            f"({n_paired} paired EVN+ODD), crop_size={crop_size}, "
            f"n_crops_per_vol={n_crops_per_vol}"
            + (f", {n_tlt} with tlt" if n_tlt else
               f" (fallback tilt [{fallback_tilt_min}, {fallback_tilt_max}]°)")
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.evn_paths) * self.n_crops_per_vol

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        vol_idx = idx % len(self.evn_paths)

        _, evn_vol = _open_mrc_mmap(str(self.evn_paths[vol_idx]))
        odd_p = self.odd_paths[vol_idx]
        if odd_p is not None:
            _, odd_vol = _open_mrc_mmap(str(odd_p))
        else:
            odd_vol = evn_vol

        D, H, W = evn_vol.shape
        cs = self.crop_size
        d0 = random.randint(0, max(0, D - cs))
        h0 = random.randint(0, max(0, H - cs))
        w0 = random.randint(0, max(0, W - cs))

        # Slicing the memmap triggers OS page faults for only the ~1–3 MB
        # of data covering this crop; np.ascontiguousarray materialises those
        # pages into a fresh contiguous array.
        evn_patch = torch.from_numpy(
            np.ascontiguousarray(evn_vol[d0:d0 + cs, h0:h0 + cs, w0:w0 + cs])
        ).unsqueeze(0)  # (1, cs, cs, cs)
        odd_patch = torch.from_numpy(
            np.ascontiguousarray(odd_vol[d0:d0 + cs, h0:h0 + cs, w0:w0 + cs])
        ).unsqueeze(0)

        if self.normalize:
            evn_patch = (evn_patch - evn_patch.mean()) / (evn_patch.std() + 1e-8)
            odd_patch = (odd_patch - odd_patch.mean()) / (odd_patch.std() + 1e-8)

        tilt = self._tilt_ranges[vol_idx]
        if tilt is None:
            tilt = (self.fallback_tilt_min, self.fallback_tilt_max)
        tilt_params = {
            "tilt_min": torch.tensor(tilt[0], dtype=torch.float32),
            "tilt_max": torch.tensor(tilt[1], dtype=torch.float32),
        }
        return evn_patch, odd_patch, tilt_params


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def _make_patch_loader(
    dataset: Dataset,
    shuffle: bool,
    cfg: EIPatchDataConfig,
    sampler=None,
) -> DataLoader:
    kwargs: dict = dict(
        dataset=dataset,
        batch_size=int(cfg.batch_size),
        shuffle=shuffle and len(dataset) > 0 if sampler is None else False,
        sampler=sampler,
        drop_last=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"]    = int(cfg.prefetch_factor)
    return DataLoader(**kwargs)


def build_ei_patch_dataloaders(cfg: EIPatchDataConfig, rank: int = 0, world_size: int = 1) -> EIDataBundle:
    """Build train / val DataLoaders over paired EVN+ODD patch crops."""
    input_dir = Path(cfg.input_dir)
    all_evn, all_odd, all_tlt = _discover_pairs(
        input_dir, cfg.evn_glob, cfg.odd_glob
    )

    all_tilt_ranges = _resolve_tlt_ranges(all_tlt)

    train_evn, train_odd, val_evn, val_odd, train_tlt, val_tlt = _split_pairs(
        all_evn, all_odd, cfg.max_val_vols, cfg.seed, cfg.max_train_vols,
        extra=all_tilt_ranges,
    )

    ds_kwargs = dict(
        crop_size=int(cfg.crop_size),
        n_crops_per_vol=int(cfg.n_crops_per_vol),
        normalize=bool(cfg.normalize),
        fallback_tilt_min=cfg.fallback_tilt_min,
        fallback_tilt_max=cfg.fallback_tilt_max,
    )
    train_ds = CryoEIPatchDataset(train_evn, train_odd, tilt_ranges=train_tlt, **ds_kwargs)
    val_ds   = CryoEIPatchDataset(val_evn,   val_odd,   tilt_ranges=val_tlt,   **ds_kwargs)

    print(
        f"[ei-patch] total={len(all_evn)}  "
        f"train_vols={len(train_evn)}  val_vols={len(val_evn)}  "
        f"train_patches={len(train_ds)}  val_patches={len(val_ds)}"
    )

    train_sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
        )
    train_loader = _make_patch_loader(train_ds, shuffle=True, cfg=cfg, sampler=train_sampler)

    return EIDataBundle(
        train_loader=train_loader,
        val_loader  =_make_patch_loader(val_ds, shuffle=False, cfg=cfg),
        train_sampler=train_sampler,
    )
