"""CryoEIFullDataset — full-volume dataset for equivariant imaging on full tomograms.

Yields paired (evn_vol, odd_vol) full sub-tomogram volumes from cryo-ET
half-set MRCs, following the same discovery / normalisation conventions as
CryoEIPatchDataset but without any spatial cropping.

Differences from the patch variant:
  - ``__getitem__`` returns a 3-tuple ``(evn, odd, tilt_params)`` where
    ``tilt_params = {"tilt_min": tensor, "tilt_max": tensor}``.  deepinv's
    training loop passes this dict to ``physics.update_parameters()`` so the
    wedge is rebuilt in-place before each training step.
  - DataLoader ``batch_size`` is always 1; effective batch size is controlled
    by gradient accumulation.
  - Optional ``target_shape`` trilinearly resamples volumes to a fixed (D, H, W)
    shape, matching supervised CryoDataConfig.target_shape semantics.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mrcfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ..utils.utils import (
    EIDataBundle, _discover_pairs, _resolve_tlt_ranges, load_mrc_volume,
    select_train_val_by_name,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EIFullDataConfig:
    input_dir: str = "./dataset/empiar-11058"
    num_workers: int = 1
    pin_memory: bool = True
    prefetch_factor: int = 1
    persistent_workers: bool = True
    max_train_vols: int | None = None
    max_val_vols: int = 5
    seed: int = 0
    train_names: list[str] | None = None   # select train vols by name; None = split
    val_names: list[str] | None = None     # select val vols by name; None = split
    # If set, volumes are trilinearly resampled to this (D, H, W) shape after
    # loading — same semantics as supervised CryoDataConfig.target_shape.
    target_shape: tuple[int, int, int] | None = None
    # If set, a random cubic crop of this side is taken after normalisation
    # (icecream's Volume.get_random_crop). None = no crop, full volume — used
    # for FSC/inference so evaluation always sees the whole tomogram.
    crop_size: int | None = None
    # Also re-normalise each crop after cropping (icecream's
    # normalize_crops). Whole-volume normalisation always happens first.
    normalize_crops: bool = False
    # Glob patterns used to discover EVN and ODD volumes inside each tomo_* dir.
    evn_glob: str = "vol*split1*.mrc"
    odd_glob: str = "vol*split2*.mrc"
    # Fallback tilt range used when no tlt file is found for a volume.
    # Should be set to match RunEIFullConfig.tilt_min / tilt_max.
    fallback_tilt_min: float = -60.0
    fallback_tilt_max: float = 60.0
    # "fbp": today's behaviour — load precomputed FBP volumes, crop+normalise
    #   (missingwedge_ei preset).
    # "measurement": load the real split1/split2 tilt series at native
    #   resolution, no crop/resample/normalise (unrolled preset). Angles and
    #   the FBP-init volumes are read independently by the physics builder,
    #   not by this dataset.
    data_source: Literal["fbp", "measurement"] = "fbp"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CryoEIFullDataset(Dataset):
    """Yields ``(evn_vol, odd_vol, tilt_params)`` where each volume is ``(1, D, H, W)``.

    Volume normalisation (zero-mean, unit-std) is applied at load time —
    matching icecream's ``load_volume`` behaviour.

    When only EVN is available, ``odd_vol`` is a copy of ``evn_vol`` so the
    single-mode ObsLoss fallback ``L = fourier_loss(y, f(y), wedge)`` works.

    ``tilt_params`` is a dict ``{"tilt_min": scalar_tensor, "tilt_max": scalar_tensor}``
    holding the per-volume tilt range read from the tlt file.  deepinv's
    training loop passes this dict straight to ``physics.update_parameters()``
    so the wedge is rebuilt in-place before each forward pass.  When no tlt
    file was found, the fallback values from ``EIFullDataConfig`` are used.

    :param list[Path] evn_paths: Paths to EVN half-set MRC volumes.
    :param list[Path] odd_paths: Paths to ODD half-set MRC volumes.
    :param tuple | None target_shape: If set, trilinearly resample each volume to
        this (D, H, W) shape after loading.
    :param list tilt_ranges: Per-volume ``(tilt_min, tilt_max)`` or ``None``.
    :param float fallback_tilt_min: Used when tilt_ranges[i] is None.
    :param float fallback_tilt_max: Used when tilt_ranges[i] is None.
    """

    def __init__(
        self,
        evn_paths: list[Path],
        odd_paths: list[Path],
        target_shape: tuple[int, int, int] | None = None,
        tilt_ranges: list[tuple[float, float] | None] | None = None,
        fallback_tilt_min: float = -60.0,
        fallback_tilt_max: float = 60.0,
        data_source: str = "fbp",
        index_offset: int = 0,
        crop_size: int | None = None,
        normalize_crops: bool = False,
    ) -> None:
        assert len(evn_paths) == len(odd_paths)
        self.evn_paths         = evn_paths
        self.odd_paths         = odd_paths
        self.target_shape      = target_shape
        self.fallback_tilt_min = fallback_tilt_min
        self.fallback_tilt_max = fallback_tilt_max
        self.data_source       = data_source
        self.crop_size         = crop_size
        self.normalize_crops   = normalize_crops
        # Global identity for measurement mode — lets TomographyEMPair.update()
        # know which tomogram this item is when train_ds/val_ds are separate
        # 0-indexed datasets (see physics/__init__.py::build_tomography_physics).
        self.index_offset      = index_offset
        self._tilt_ranges: list[tuple[float, float] | None] = (
            tilt_ranges if tilt_ranges is not None else [None] * len(evn_paths)
        )

        n_tlt = sum(t is not None for t in self._tilt_ranges)
        print(
            f"[ei-full] CryoEIFullDataset: {len(evn_paths)} paired EVN+ODD vols [lazy]"
            + (f", {n_tlt} with tlt angles" if n_tlt else
               f" (fallback tilt [{fallback_tilt_min}, {fallback_tilt_max}]°)")
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.evn_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if self.data_source == "measurement":
            evn_sino, odd_sino = self._load_measurement(idx)
            tomo_idx = torch.tensor(idx + self.index_offset)
            return evn_sino, odd_sino, {"tomo_idx": tomo_idx}

        evn = self._load_and_prepare(self.evn_paths[idx])   # (1, D, H, W), whole-volume normalised
        odd = self._load_and_prepare(self.odd_paths[idx])
        evn, odd = self._crop_pair(evn, odd)

        tilt = self._tilt_ranges[idx]
        if tilt is None:
            tilt = (self.fallback_tilt_min, self.fallback_tilt_max)
        tilt_min, tilt_max = tilt

        tilt_params = {
            "tilt_min": torch.tensor(tilt_min, dtype=torch.float32),
            "tilt_max": torch.tensor(tilt_max, dtype=torch.float32),
            "vol_shape": torch.tensor(evn.shape[-3:], dtype=torch.int64),
            "tomo_idx": torch.tensor(idx + self.index_offset),
        }
        return evn, odd, tilt_params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_measurement(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load real split1/split2 tilt series — no crop, no normalise. If
        ``target_shape`` is set (local testing only), each projection is
        resampled to match its (D, H) so the sinogram stays
        consistent with the correspondingly-resampled FBP-init volume built by
        ``physics/__init__.py::build_tomography_physics``. Angles and FBP-init
        volumes are read independently by the physics builder, not here.

        Returns ``(1, V, A, N)`` tensors — matches ``TomographyEM.A()``'s
        ``(B, C, V, A, N)`` convention (V<-ny, A<-n_angles, N<-nx; see
        ``scripts/test_tomography_em.py``) — collated by the default
        DataLoader into ``(B, 1, V, A, N)``.
        """
        tomo_dir = self.evn_paths[idx].parent
        evn_sino = self._load_tilt_series(tomo_dir, "split1")
        odd_sino = self._load_tilt_series(tomo_dir, "split2")
        if self.target_shape is not None:
            # target_shape is canonical (Y, X, Z); the detector grid is
            # (V, N) = (Y, X), i.e. its first two axes.
            d, h, _ = self.target_shape
            evn_sino = self._resample_sinogram(evn_sino, d, h)
            odd_sino = self._resample_sinogram(odd_sino, d, h)
        return evn_sino, odd_sino

    @staticmethod
    def _load_tilt_series(tomo_dir: Path, split: str) -> torch.Tensor:
        matches = sorted(tomo_dir.glob(f"tilt_series_*_{split}.mrc"))
        if not matches:
            raise FileNotFoundError(
                f"measurement mode: no tilt_series_*_{split}.mrc found in {tomo_dir}"
            )
        with mrcfile.open(str(matches[0]), permissive=True, mode="r") as mrc:
            ts_np = np.array(mrc.data, dtype=np.float32)  # (n_angles, ny, nx)
        ts_np = np.ascontiguousarray(np.moveaxis(ts_np, 0, 1))  # (ny, n_angles, nx) = (V, A, N)
        ts = torch.from_numpy(ts_np).unsqueeze(0)  
        return (ts - ts.mean()) / (ts.std() + 1e-8)

    @staticmethod
    def _resample_sinogram(sino: torch.Tensor, ny: int, nx: int) -> torch.Tensor:
        """Resample a (1, V0, A, N0) sinogram to (1, ny, A, nx) — resizes V, N
        only, keeps the angle axis A untouched."""
        sino = sino.permute(2, 0, 1, 3)  # (A, C, V0, N0)
        sino = torch.nn.functional.interpolate(
            sino, size=(ny, nx), mode="bilinear", align_corners=False,
        )
        return sino.permute(1, 2, 0, 3).contiguous()  # (C, ny, A, nx)

    def _load_and_prepare(self, path: Path) -> torch.Tensor:
        """Load MRC, reorder axes, optional resample, normalise → (1, D, H, W).

        Normalisation is whole-volume (icecream's ``load_volume`` /
        ``normalize_volume``), applied before any cropping — see
        ``_crop_pair`` for the crop step.
        """
        vol = torch.from_numpy(load_mrc_volume(path, order="native"))  # (Y, X, Z) = (D, H, W)

        if self.target_shape is not None:
            # interpolate expects (B, C, D, H, W)
            vol = torch.nn.functional.interpolate(
                vol.unsqueeze(0).unsqueeze(0),
                size=self.target_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)  # back to (D, H, W)

        mu = vol.mean()
        sigma = vol.std()
        vol = (vol - mu) / (sigma + 1e-8)

        return vol.unsqueeze(0)  # (1, D, H, W)

    def _crop_pair(self, evn: torch.Tensor, odd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Crop EVN/ODD at one shared random origin (icecream's ``get_random_crop``).

        ``self.crop_size is None`` → no crop, full (possibly non-cubic) volume —
        used for FSC/inference. Otherwise a random cubic crop of that side,
        clamped to fit; optionally re-normalised per crop (``normalize_crops``).
        """
        if self.crop_size is None:
            return evn, odd

        _, D, H, W = evn.shape
        cs = min(self.crop_size, D, H, W)
        d0 = random.randint(0, D - cs)
        h0 = random.randint(0, H - cs)
        w0 = random.randint(0, W - cs)

        evn = evn[:, d0:d0 + cs, h0:h0 + cs, w0:w0 + cs]
        odd = odd[:, d0:d0 + cs, h0:h0 + cs, w0:w0 + cs]

        if self.normalize_crops:
            evn = (evn - evn.mean()) / (evn.std() + 1e-8)
            odd = (odd - odd.mean()) / (odd.std() + 1e-8)

        return evn, odd


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def _make_full_loader(
    dataset: Dataset,
    shuffle: bool,
    cfg: EIFullDataConfig,
    sampler=None,
) -> DataLoader:
    kwargs: dict = dict(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=shuffle and sampler is None and len(dataset) > 0,
        drop_last=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(**kwargs)


def build_ei_full_dataloaders(cfg: EIFullDataConfig, ctx=None) -> EIDataBundle:
    """Build train / val DataLoaders over full cryo-ET volumes.

    ``ctx`` with more than one data-parallel replica shards the train volumes
    across replicas; every rank in a replica still sees the same volume.
    """
    input_dir = Path(cfg.input_dir)
    all_evn, all_odd, all_tlt = _discover_pairs(
        input_dir, cfg.evn_glob, cfg.odd_glob
    )

    all_tilt_ranges = _resolve_tlt_ranges(all_tlt)

    train_evn, train_odd, val_evn, val_odd, train_tlt_ranges, val_tlt_ranges = select_train_val_by_name(
        all_evn, all_odd, cfg.max_val_vols, cfg.seed, cfg.max_train_vols,
        train_names=cfg.train_names, val_names=cfg.val_names,
        extra=all_tilt_ranges,
    )

    ds_kwargs = dict(
        target_shape=cfg.target_shape,
        fallback_tilt_min=cfg.fallback_tilt_min,
        fallback_tilt_max=cfg.fallback_tilt_max,
        data_source=cfg.data_source,
    )
    # Train sees random crops (or the whole volume if crop_size is None);
    # val/FSC always evaluates the whole volume — crop_size=None regardless
    # of the training config.
    train_ds = CryoEIFullDataset(train_evn, train_odd, tilt_ranges=train_tlt_ranges,
                                  crop_size=cfg.crop_size, normalize_crops=cfg.normalize_crops,
                                  **ds_kwargs)
    val_ds   = CryoEIFullDataset(val_evn,   val_odd,   tilt_ranges=val_tlt_ranges,
                                  index_offset=len(train_evn), crop_size=None, **ds_kwargs)

    print(
        f"[ei-full] total={len(all_evn)}  "
        f"train_vols={len(train_evn)}  val_vols={len(val_evn)}"
    )

    train_sampler = (ctx.distributed_data_sampler(train_ds, shuffle=True)
                     if ctx is not None and ctx.dp_world_size > 1 else None)

    return EIDataBundle(
        train_loader = _make_full_loader(train_ds, shuffle=True,  cfg=cfg, sampler=train_sampler),
        val_loader   = _make_full_loader(val_ds,   shuffle=False, cfg=cfg),
        train_sampler= train_sampler,
    )
