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

from dataclasses import dataclass
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
class EIFullDataConfig:
    input_dir: str = "./dataset/empiar-11058"
    num_workers: int = 1
    pin_memory: bool = True
    prefetch_factor: int = 1
    persistent_workers: bool = True
    max_train_vols: int | None = None
    max_val_vols: int = 5
    seed: int = 0
    # If set, volumes are trilinearly resampled to this (D, H, W) shape after
    # loading — same semantics as supervised CryoDataConfig.target_shape.
    target_shape: tuple[int, int, int] | None = None
    # Glob patterns used to discover EVN and ODD volumes inside each tomo_* dir.
    evn_glob: str = "vol*split1*.mrc"
    odd_glob: str = "vol*split2*.mrc"
    # Fallback tilt range used when no tlt file is found for a volume.
    # Should be set to match RunEIFullConfig.tilt_min / tilt_max.
    fallback_tilt_min: float = -60.0
    fallback_tilt_max: float = 60.0


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
    ) -> None:
        assert len(evn_paths) == len(odd_paths)
        self.evn_paths         = evn_paths
        self.odd_paths         = odd_paths
        self.target_shape      = target_shape
        self.fallback_tilt_min = fallback_tilt_min
        self.fallback_tilt_max = fallback_tilt_max
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
        evn = self._load_and_prepare(self.evn_paths[idx])   # (1, D, H, W)
        odd = self._load_and_prepare(self.odd_paths[idx])

        tilt = self._tilt_ranges[idx]
        if tilt is None:
            tilt = (self.fallback_tilt_min, self.fallback_tilt_max)
        tilt_min, tilt_max = tilt

        tilt_params = {
            "tilt_min": torch.tensor(tilt_min, dtype=torch.float32),
            "tilt_max": torch.tensor(tilt_max, dtype=torch.float32),
        }
        return evn, odd, tilt_params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_and_prepare(self, path: Path) -> torch.Tensor:
        """Load MRC, reorder axes, optional resample, centre-crop to cube, normalise → (1, D, H, W)."""
        # MRC stores (Z, Y, X); moveaxis → (Y, X, Z) = (D, H, W)
        with mrcfile.open(str(path), permissive=True, mode="r") as mrc:
            vol_np = np.array(mrc.data, dtype=np.float32)
        vol = torch.from_numpy(np.moveaxis(vol_np, 0, 2))  # (D, H, W)

        if self.target_shape is not None:
            # interpolate expects (B, C, D, H, W)
            vol = torch.nn.functional.interpolate(
                vol.unsqueeze(0).unsqueeze(0),
                size=self.target_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)  # back to (D, H, W)

        # Centre-crop to cube of side min(D, H, W)
        D, H, W = vol.shape
        S = min(D, H, W)
        d0, h0, w0 = (D - S) // 2, (H - S) // 2, (W - S) // 2
        vol = vol[d0:d0 + S, h0:h0 + S, w0:w0 + S]  # (S, S, S)

        # Normalise after crop so stats reflect the kept region
        mu = vol.mean()
        sigma = vol.std()
        vol = (vol - mu) / (sigma + 1e-8)

        return vol.unsqueeze(0)  # (1, S, S, S)


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def _make_full_loader(
    dataset: Dataset,
    shuffle: bool,
    cfg: EIFullDataConfig,
) -> DataLoader:
    kwargs: dict = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=shuffle and len(dataset) > 0,
        drop_last=False,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
    )
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(**kwargs)


def build_ei_full_dataloaders(cfg: EIFullDataConfig) -> EIDataBundle:
    """Build train / val DataLoaders over full cryo-ET volumes."""
    input_dir = Path(cfg.input_dir)
    all_evn, all_odd, all_tlt = _discover_pairs(
        input_dir, cfg.evn_glob, cfg.odd_glob
    )

    all_tilt_ranges = _resolve_tlt_ranges(all_tlt)

    train_evn, train_odd, val_evn, val_odd, train_tlt_ranges, val_tlt_ranges = _split_pairs(
        all_evn, all_odd, cfg.max_val_vols, cfg.seed, cfg.max_train_vols,
        extra=all_tilt_ranges,
    )

    ds_kwargs = dict(
        target_shape=cfg.target_shape,
        fallback_tilt_min=cfg.fallback_tilt_min,
        fallback_tilt_max=cfg.fallback_tilt_max,
    )
    train_ds = CryoEIFullDataset(train_evn, train_odd, tilt_ranges=train_tlt_ranges, **ds_kwargs)
    val_ds   = CryoEIFullDataset(val_evn,   val_odd,   tilt_ranges=val_tlt_ranges,   **ds_kwargs)

    print(
        f"[ei-full] total={len(all_evn)}  "
        f"train_vols={len(train_evn)}  val_vols={len(val_evn)}"
    )

    return EIDataBundle(
        train_loader = _make_full_loader(train_ds, shuffle=True,  cfg=cfg),
        val_loader   = _make_full_loader(val_ds,   shuffle=False, cfg=cfg),
    )
