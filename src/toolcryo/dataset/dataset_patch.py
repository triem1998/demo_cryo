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
  - Volumes are memory-mapped in their on-disk dtype (these MRC files are
    float16): only the OS pages covering each requested crop (~1-3 MB) are read
    per __getitem__ call, and the cast to float32 is applied to the crop, not
    to the volume.  Casting the volume would materialise a full anonymous copy
    (2.15 GB for a 512x1024x1024 float16 volume) that the OS can never reclaim.
    The full volume is never copied into RAM unless explicitly requested
    (e.g. inference, via _LazyVolList).
"""
from __future__ import annotations

import fcntl
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mrcfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..utils.utils import (
    EIDataBundle, _discover_pairs, _resolve_tlt_ranges, select_train_val_by_name,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EIPatchDataConfig:
    input_dir: str = "./dataset/empiar-11058"
    crop_size: int = 72
    n_crops_per_vol: int = 10          # virtual epoch = n_vols * n_crops_per_vol
    batch_size: int = 4
    # False = icecream's convention: every optimizer step draws its crops from a
    # single volume, so the step uses that volume's exact wedge. True (default)
    # lets a batch mix volumes, and the shared wedge becomes the intersection of
    # their tilt ranges (see MissingWedge.update_parameters).
    mix_volumes: bool = True
    num_workers: int = 1
    pin_memory: bool = True
    prefetch_factor: int = 1
    persistent_workers: bool = True
    max_train_vols: int | None = None
    max_val_vols: int = 5
    seed: int = 0
    train_names: list[str] | None = None   # select train vols by name; None = split
    val_names: list[str] | None = None     # select val vols by name; None = split
    normalize: bool = True              # whole-volume zero-mean, unit-std (icecream's load_volume)
    normalize_crops: bool = False       # also re-normalise each crop (icecream's normalize_crops)
    use_mask: bool = True               # sample crops inside the specimen mask
    mask_frac: float = 0.5              # min fraction of a crop inside the mask
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
    view of shape (D, H, W) = (Y, X, Z) over the on-disk data, in the file's
    **native dtype** (float16 for this dataset).  The OS reads only the pages
    corresponding to whatever region is sliced — the full volume is never
    copied into RAM by this call alone.  Callers cast their own slice to
    float32; casting here instead would defeat the mapping entirely.

    Both objects are cached so the mapping stays alive across calls and file
    descriptors are not repeatedly opened.  The cache is per-process, so each
    DataLoader worker maintains its own independent cache.
    """
    mrc = mrcfile.mmap(path_str, permissive=True, mode='r')
    data = mrc.data  # numpy.memmap, shape (Z, Y, X), native dtype
    # moveaxis creates a non-contiguous view — no data pages are read here
    vol = np.moveaxis(data, 0, 2)   # (Z, Y, X) → (Y, X, Z) = (D, H, W)
    return mrc, vol


@lru_cache(maxsize=64)
def _vol_mean_std(path_str: str) -> tuple[float, float]:
    """Whole-volume mean/std (icecream's ``load_volume`` normalisation),
    computed once per volume and cached on disk beside it as two floats.
    Without the sidecar the stream is repeated once per loader process (8
    under DDP) and once per run; one process computes, the rest wait on the
    lock, same discipline as _vol_mask.

    Streams the file in blocks rather than calling ``vol.std()``: numpy's std
    materialises a full-size ``arr - mean`` temporary (4.2 GB for one of these
    volumes), which is exactly the kind of allocation this module exists to
    avoid.  Accumulating sum and sum-of-squares in float64 keeps the peak at
    one block while being *more* accurate than the previous float32 reduction.

    Reduces over the on-disk ``(Z, Y, X)`` array rather than the transposed
    view — mean and variance are order-independent, and the untransposed array
    is contiguous, so the read is sequential.
    """
    cache = Path(path_str).with_suffix(".stats.npy")
    if not cache.exists():
        try:
            lf = open(cache.with_suffix(".lock"), "w")
            fcntl.flock(lf, fcntl.LOCK_EX)      # one process computes, the rest wait
        except OSError:
            lf = None                           # no flock here: everyone computes
        try:
            if not cache.exists():              # a waiter re-checks and skips the read
                mrc, _ = _open_mrc_mmap(path_str)
                data = mrc.data
                n = total = total_sq = 0.0
                for i in range(0, data.shape[0], 8):  # ~67 MB per block as float64
                    blk = np.asarray(data[i:i + 8], dtype=np.float64)
                    n += blk.size
                    total += blk.sum()
                    total_sq += (blk * blk).sum()
                mean = total / n
                std = np.sqrt(max(total_sq / n - mean * mean, 0.0))
                try:
                    tmp = cache.with_suffix(f".{os.getpid()}.tmp.npy")
                    np.save(tmp, np.array([mean, std]))
                    tmp.replace(cache)
                except OSError:
                    return float(mean), float(std)   # read-only dir
        finally:
            if lf is not None:
                lf.close()                      # releases the lock
    mean, std = np.load(cache)
    return float(mean), float(std)


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
        # Inference wants the whole volume, so the full read is intentional here.
        vol_t = torch.from_numpy(np.ascontiguousarray(vol_np, dtype=np.float32))
        if self._normalize:
            vol_t = (vol_t - vol_t.mean()) / (vol_t.std() + 1e-8)
        return vol_t

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _vol_mask(evn_path: str, odd_path: str) -> np.ndarray:
    """IsoNet-style specimen mask of the EVN/ODD average, as icecream builds it.

    Stored half-res: make_mask duplicates every value into a 2x2x2 block, so
    full-res is 8x redundant. Cached on disk beside the volume and memory-mapped
    -- a rebuild costs ~30s and several GB. Read it through _mask_crop_mean.
    """
    cache = Path(evn_path).with_suffix(".mask.npy")
    if not cache.exists():
        try:
            lf = open(cache.with_suffix(".lock"), "w")
            fcntl.flock(lf, fcntl.LOCK_EX)      # one process builds, the rest wait
        except OSError:
            lf = None                           # no flock here: everyone builds
        try:
            if not cache.exists():              # a waiter re-checks and skips the build
                from ..icecream_orig.utils.mask_util import make_mask
                _, evn = _open_mrc_mmap(evn_path)
                _, odd = _open_mrc_mmap(odd_path)
                avg = (np.asarray(evn, np.float32) + np.asarray(odd, np.float32)) / 2
                mask = make_mask(avg, side=5, density_percentage=50., std_percentage=50.)
                mask = mask[::2, ::2, ::2]
                try:
                    tmp = cache.with_suffix(f".{os.getpid()}.tmp.npy")
                    np.save(tmp, mask)
                    tmp.replace(cache)
                except OSError:
                    return mask                 # read-only dir
        finally:
            if lf is not None:
                lf.close()                      # releases the lock
    return np.load(cache, mmap_mode="r")


def _mask_crop_mean(half: np.ndarray, d0: int, h0: int, w0: int, cs: int) -> float:
    """Full-res mask fraction of a crop, from the half-res store.

    Expands only the covering block, so odd offsets stay exact -- indexing the
    half-res array directly with //2 drops an edge cell and flips ~0.4% of the
    accept/reject decisions.
    """
    sl = lambda x: slice(x // 2, (x + cs + 1) // 2)   # noqa: E731
    b = half[sl(d0), sl(h0), sl(w0)]
    b = np.repeat(np.repeat(np.repeat(b, 2, 0), 2, 1), 2, 2)
    return b[d0 % 2:d0 % 2 + cs, h0 % 2:h0 % 2 + cs, w0 % 2:w0 % 2 + cs].mean()


class CryoEIPatchDataset(Dataset):
    """Yields ``(evn_patch, odd_patch, tilt_params)`` random cubic crops.

    Both patches are cropped from the **same random spatial position** so the
    cross half-set pairing is preserved — ObsLoss and EqLoss can compare them
    exactly as they compare full volumes in CryoEIFullDataset.

    Volumes are memory-mapped at the OS level.  Each ``__getitem__`` call
    reads only the ~1–3 MB of disk pages covering the requested 72³ crop.
    Repeated access to the same region within a worker is served from the OS
    page cache (no disk I/O after the first touch).

    Normalisation is whole-volume (icecream's ``load_volume``), applied
    before cropping when ``normalize=True``; each crop is optionally
    re-normalised on top of that when ``normalize_crops=True`` (icecream's
    ``normalize_crops``). Whole-volume stats are cached per (path, process)
    so only the first crop drawn from a volume in a worker pays the cost of
    the full-file read.

    When only EVN is available, ``odd_patch`` is a copy of ``evn_patch`` so
    the single-half ObsLoss fallback ``L = fourier_loss(y, f(y), wedge)``
    still works.

    :param list[Path] evn_paths: Paths to EVN half-set MRC volumes.
    :param list[Path | None] odd_paths: Paths to ODD half-set MRC volumes (or None).
    :param int crop_size: Cubic crop side length (default 72).
    :param int n_crops_per_vol: Virtual crops per volume per epoch (default 10).
    :param bool normalize: Whole-volume standardise, applied before cropping (default True).
    :param bool normalize_crops: Also standardise each crop after cropping (default False).
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
        normalize_crops: bool = False,
        use_mask: bool = True,
        mask_frac: float = 0.5,
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
        self.normalize_crops   = normalize_crops
        self.use_mask          = use_mask
        self.mask_frac         = mask_frac
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

        evn_path = str(self.evn_paths[vol_idx])
        odd_p = self.odd_paths[vol_idx]
        odd_path = str(odd_p) if odd_p is not None else evn_path

        _, evn_vol = _open_mrc_mmap(evn_path)
        _, odd_vol = _open_mrc_mmap(odd_path)

        D, H, W = evn_vol.shape
        cs = self.crop_size
        # Retry until the crop is mask_frac inside the specimen; after 100
        # rejections keep the last draw rather than loop forever.
        mask = _vol_mask(evn_path, odd_path) if self.use_mask else None
        for _ in range(100):
            d0 = random.randint(0, max(0, D - cs))
            h0 = random.randint(0, max(0, H - cs))
            w0 = random.randint(0, max(0, W - cs))
            if mask is None or _mask_crop_mean(mask, d0, h0, w0, cs) >= self.mask_frac:
                break

        # Slicing the memmap triggers OS page faults for only the ~1–3 MB
        # of data covering this crop; np.ascontiguousarray materialises those
        # pages into a fresh contiguous array.
        evn_patch = torch.from_numpy(
            np.ascontiguousarray(evn_vol[d0:d0 + cs, h0:h0 + cs, w0:w0 + cs],
                                 dtype=np.float32)
        ).unsqueeze(0)  # (1, cs, cs, cs)
        odd_patch = torch.from_numpy(
            np.ascontiguousarray(odd_vol[d0:d0 + cs, h0:h0 + cs, w0:w0 + cs],
                                 dtype=np.float32)
        ).unsqueeze(0)

        if self.normalize:
            evn_mu, evn_sigma = _vol_mean_std(evn_path)
            odd_mu, odd_sigma = _vol_mean_std(odd_path)
            evn_patch = (evn_patch - evn_mu) / (evn_sigma + 1e-8)
            odd_patch = (odd_patch - odd_mu) / (odd_sigma + 1e-8)
            if self.normalize_crops:
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
# Targeted patch extraction (fixed positions) — used by training probes
# ---------------------------------------------------------------------------

def extract_patches_at_positions(
    evn_path: Path,
    odd_path: Path | None,
    positions: list[tuple[int, int, int]],
    crop_size: int,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int]]]:
    """Extract ``crop_size³`` EVN/ODD patches at the given (d, h, w) origins.

    Each patch is sliced out of the memory-mapped volume and only then cast and
    normalised, using the cached whole-volume statistics — normalising the whole
    volume and cropping afterwards is the same operation, but it materialised
    three full-volume copies per half-set (~6 GB), which OOM-killed rank 0 when
    the training probe looped over a dozen tomograms.

    Each origin is clamped so a **full** ``crop_size³`` window fits inside the
    volume — the model always receives a full-size crop.  The clamped origins
    are returned.

    :returns: ``(evn_crops, odd_crops, used_origins)`` where crops are
        (N, crop_size, crop_size, crop_size) tensors.
    """
    _, evn_vol = _open_mrc_mmap(str(evn_path))
    odd_vol = evn_vol if odd_path is None else _open_mrc_mmap(str(odd_path))[1]

    if normalize:
        evn_mu, evn_sigma = _vol_mean_std(str(evn_path))
        odd_mu, odd_sigma = (
            (evn_mu, evn_sigma) if odd_path is None else _vol_mean_std(str(odd_path))
        )

    D, H, W = evn_vol.shape
    cs = crop_size
    evn_crops, odd_crops, used = [], [], []
    for d0, h0, w0 in positions:
        d0 = int(min(max(0, d0), max(0, D - cs)))
        h0 = int(min(max(0, h0), max(0, H - cs)))
        w0 = int(min(max(0, w0), max(0, W - cs)))
        sl = (slice(d0, d0 + cs), slice(h0, h0 + cs), slice(w0, w0 + cs))
        evn_c = torch.from_numpy(np.ascontiguousarray(evn_vol[sl], dtype=np.float32))
        odd_c = torch.from_numpy(np.ascontiguousarray(odd_vol[sl], dtype=np.float32))
        if normalize:
            evn_c = (evn_c - evn_mu) / (evn_sigma + 1e-8)
            odd_c = (odd_c - odd_mu) / (odd_sigma + 1e-8)
        evn_crops.append(evn_c)
        odd_crops.append(odd_c)
        used.append((d0, h0, w0))
    return torch.stack(evn_crops), torch.stack(odd_crops), used


# ---------------------------------------------------------------------------
# Single-volume batching (icecream convention)
# ---------------------------------------------------------------------------

class SingleVolumeBatchSampler(Sampler[list[int]]):
    """Batches whose crops all come from a single volume — icecream's convention.

    ``CryoEIPatchDataset`` maps index ``i`` to volume ``i % n_vols``, so each
    residue class mod ``n_vols`` is exactly that volume's pool of crop indices.
    Keeping a batch inside one residue class therefore gives every crop in an
    optimizer step the same tilt range, and ``MissingWedge.update_parameters``
    reduces to the identity instead of intersecting several volumes' wedges.

    Under DDP the ranks split *one* volume's global batch of
    ``world_size * batch_size`` crops between them, so every rank sits on the
    same volume — and hence the same wedge — at the same step, exactly as
    icecream's single-process loop does.  Everything up to the final per-rank
    slice is computed identically on every rank, which is what keeps them in
    lockstep for the gradient all-reduce.

    A volume contributes ``n_crops_per_vol // (world_size * batch_size)`` global
    batches per epoch; any remainder is dropped so all ranks run the same number
    of steps.

    :param int n_vols: Number of volumes in the dataset.
    :param int n_crops_per_vol: Crop indices available per volume per epoch.
    :param int batch_size: Crops per rank per step.
    :param bool shuffle: Shuffle crop order within a volume, and batch order.
    :param int seed: Base seed, combined with the epoch set by ``set_epoch``.
    :param int rank: This process's rank.
    :param int world_size: Total number of processes.
    """

    def __init__(
        self,
        n_vols: int,
        n_crops_per_vol: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        global_batch = int(world_size) * int(batch_size)
        if n_vols > 0 and n_crops_per_vol < global_batch:
            raise ValueError(
                "mix_volumes=False needs n_crops_per_vol >= world_size * batch_size "
                f"({world_size} * {batch_size} = {global_batch}), got "
                f"n_crops_per_vol={n_crops_per_vol}."
            )
        self.n_vols          = int(n_vols)
        self.n_crops_per_vol = int(n_crops_per_vol)
        self.batch_size      = int(batch_size)
        self.global_batch    = global_batch
        self.shuffle         = bool(shuffle)
        self.seed            = int(seed)
        self.rank            = int(rank)
        self.world_size      = int(world_size)
        self.epoch           = 0

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle for the next epoch — called by ``BaseTrainer.train``."""
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        # Seeded without the rank, so every rank builds the same global batches.
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        gbs = self.global_batch
        global_batches: list[list[int]] = []
        for vol_idx in range(self.n_vols):
            pool = [vol_idx + k * self.n_vols for k in range(self.n_crops_per_vol)]
            if self.shuffle:
                pool = [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()]
            # Drop the remainder: a partial global batch would leave ranks with
            # unequal step counts and hang the gradient all-reduce.
            global_batches += [pool[s:s + gbs]
                               for s in range(0, len(pool) - gbs + 1, gbs)]

        if self.shuffle and global_batches:
            order = torch.randperm(len(global_batches), generator=g).tolist()
            global_batches = [global_batches[i] for i in order]

        lo = self.rank * self.batch_size
        return [gb[lo:lo + self.batch_size] for gb in global_batches]

    def __iter__(self):
        yield from self._batches()

    def __len__(self) -> int:
        if self.n_vols == 0:
            return 0
        return self.n_vols * (self.n_crops_per_vol // self.global_batch)


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def _make_patch_loader(
    dataset: Dataset,
    shuffle: bool,
    cfg: EIPatchDataConfig,
    sampler=None,
    batch_sampler=None,
) -> DataLoader:
    kwargs: dict = dict(
        dataset=dataset,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )
    if batch_sampler is not None:
        # DataLoader rejects batch_size / shuffle / sampler / drop_last alongside
        # batch_sampler — the batch sampler already decides all four.
        kwargs["batch_sampler"] = batch_sampler
    else:
        kwargs.update(
            batch_size=int(cfg.batch_size),
            shuffle=shuffle and len(dataset) > 0 if sampler is None else False,
            sampler=sampler,
            drop_last=False,
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

    train_evn, train_odd, val_evn, val_odd, train_tlt, val_tlt = select_train_val_by_name(
        all_evn, all_odd, cfg.max_val_vols, cfg.seed, cfg.max_train_vols,
        train_names=cfg.train_names, val_names=cfg.val_names,
        extra=all_tilt_ranges,
    )

    ds_kwargs = dict(
        crop_size=int(cfg.crop_size),
        n_crops_per_vol=int(cfg.n_crops_per_vol),
        normalize=bool(cfg.normalize),
        normalize_crops=bool(cfg.normalize_crops),
        use_mask=bool(cfg.use_mask),
        mask_frac=float(cfg.mask_frac),
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

    if not cfg.mix_volumes:
        # icecream convention: one volume per optimizer step, with its own wedge.
        train_sampler = SingleVolumeBatchSampler(
            n_vols=len(train_evn), n_crops_per_vol=int(cfg.n_crops_per_vol),
            batch_size=int(cfg.batch_size), shuffle=True, seed=int(cfg.seed),
            rank=rank, world_size=world_size,
        )
        # Val is not sharded across ranks today — every rank evaluates all of it.
        val_sampler = SingleVolumeBatchSampler(
            n_vols=len(val_evn), n_crops_per_vol=int(cfg.n_crops_per_vol),
            batch_size=int(cfg.batch_size), shuffle=False, seed=int(cfg.seed),
        )
        if rank == 0:
            print(f"[ei-patch] mix_volumes=False (icecream): one volume per step, "
                  f"global batch={world_size * int(cfg.batch_size)}, "
                  f"{len(train_sampler)} train step(s)/epoch")
        return EIDataBundle(
            train_loader=_make_patch_loader(train_ds, shuffle=True, cfg=cfg,
                                            batch_sampler=train_sampler),
            val_loader  =_make_patch_loader(val_ds, shuffle=False, cfg=cfg,
                                            batch_sampler=val_sampler),
            train_sampler=train_sampler,
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
