"""infer_patch.py — Inference for a trained patch-based EI model.

Matches icecream's inference_util.inference exactly:
  1. Pre-pad the volume.
  2. Slide a crop_size³ window with stride, extracting overlapping patches.
  3. For each patch:  output = f(A(f(patch)))  — model applied twice, wedge in between.
  4. Reassemble via window-weighted overlap averaging.
  5. Average EVN and ODD reconstructions:  recon = 0.5 * (result_evn + result_odd).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mrcfile
import numpy as np
import torch
import torch.nn as nn

from ..base_config import RunEIBaseConfig
from ..dataset.dataset_patch import EIPatchDataConfig, build_ei_patch_dataloaders
from ..physics import MissingWedge
from ..losses.losses import _initialize_window, _symmetrize_and_binarize
from ..utils.utils import (
    GpuFSC,
    _find_mrc,
    _read_pixel_sizes,
    _save_mrc,
    _znorm,
    fsc_shell,
    build_ei_model,
    dump_config_json,
    ensure_dir,
    seed_everything,
)
from ..utils.plot import save_fsc_figure, save_resolution_histogram, save_slice_figure


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class RunEIPatchInferenceConfig(RunEIBaseConfig):
    # ── Checkpoint ──────────────────────────────────────────────────────────
    checkpoint_path: str = ""

    # ── Data ────────────────────────────────────────────────────────────────
    output_dir: str = "./runs/inference_patch"
    max_infer_vols: int = 5
    normalize: bool = True

    # ── DataLoader ──────────────────────────────────────────────────────────
    num_workers: int = 1
    prefetch_factor: int = 1

    # ── Patch inference (must match training config) ─────────────────────────
    crop_size: int = 72
    stride: int = 36
    infer_batch_size: int = 4
    infer_downsample: int = 1
    pre_pad: bool = True

    # ── Comparison globs ─────────────────────────────────────────────────────
    icecream_glob: str = "vol_*[Ii]cecream*"
    isonet_glob: str = "vol_*[Ii]so[Nn]et*"
    isonet_fallback_glob: str = "vol_*DDW*"

    # ── Output ───────────────────────────────────────────────────────────────
    save_recon_mrc: bool = False

    @classmethod
    def from_yaml(cls, conf: dict) -> "RunEIPatchInferenceConfig":
        return cls.model_validate(cls._flat_from_yaml(conf, "demo-cryo-ei-patch-inference"))


# ---------------------------------------------------------------------------
# Core sliding-window inference  (mirrors icecream's inference_util.inference)
# ---------------------------------------------------------------------------

def _compute_padd(N: int, filt_size: int, stride: int) -> int:
    w = (N - filt_size) // stride + 1
    N_rec = (w - 1) * stride + filt_size
    return (N_rec - N) % filt_size


def patch_inference(
    vol: torch.Tensor,
    model: nn.Module,
    wedge: torch.Tensor,
    crop_size: int,
    stride: int,
    infer_batch_size: int,
    device: torch.device,
    pre_pad: bool = True,
) -> np.ndarray:
    """Sliding-window f(A(f(.))) inference — mirrors icecream's inference_util.inference exactly.

    :param vol: (D, H, W) CPU float32 tensor (globally normalised).
    :param wedge: wedge_input mask (mask_size³) on CPU.
    :returns: (D, H, W) float32 numpy array.
    """
    pre_pad_size = crop_size // 4

    vol_fbp = vol.to(torch.float16)
    if pre_pad:
        vol_fbp = torch.nn.functional.pad(vol_fbp, (pre_pad_size, 0, pre_pad_size, 0, pre_pad_size, 0))
    wedge_dev = wedge.to(device)

    pad_i = _compute_padd(vol_fbp.shape[0], crop_size, stride)
    pad_j = _compute_padd(vol_fbp.shape[1], crop_size, stride)
    pad_k = _compute_padd(vol_fbp.shape[2], crop_size, stride)
    vol_fbp_pad = torch.nn.functional.pad(vol_fbp, (0, pad_k, 0, pad_j, 0, pad_i))
    del vol_fbp

    N1_orig, N2_orig, N3_orig = vol.shape
    vol_est = torch.zeros((N1_orig, N2_orig, N3_orig))
    if pre_pad:
        vol_est = torch.nn.functional.pad(vol_est, (pre_pad_size, 0, pre_pad_size, 0, pre_pad_size, 0))
    N1_pad, N2_pad, N3_pad = vol_est.shape
    pad_i = _compute_padd(vol_est.shape[0], crop_size, stride)
    pad_j = _compute_padd(vol_est.shape[1], crop_size, stride)
    pad_k = _compute_padd(vol_est.shape[2], crop_size, stride)
    vol_est = torch.nn.functional.pad(vol_est, (0, pad_k, 0, pad_j, 0, pad_i))
    N1, N2, N3 = vol_est.shape
    mask = torch.zeros_like(vol_est)
    window = _initialize_window(crop_size).cpu()

    positions = [
        (i, j, k)
        for i in range(0, N1, stride)
        for j in range(0, N2, stride)
        for k in range(0, N3, stride)
        if i + crop_size <= N1 and j + crop_size <= N2 and k + crop_size <= N3
    ]

    use_amp = device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(positions), infer_batch_size):
            batch_positions = positions[batch_start:batch_start + infer_batch_size]
            batch = torch.stack([
                vol_fbp_pad[i:i + crop_size, j:j + crop_size, k:k + crop_size]
                for i, j, k in batch_positions
            ]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(batch[:, None])[:, 0]        # f(crop)
            output = output.float()
            output = _apply_wedge_batch(output, wedge_dev)  # A(f(crop))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(output[:, None])[:, 0]       # f(A(f(crop)))
            out_cpu = output.detach().cpu()
            for b, (i, j, k) in enumerate(batch_positions):
                vol_est[i:i + crop_size, j:j + crop_size, k:k + crop_size] += out_cpu[b] * window
                mask[  i:i + crop_size, j:j + crop_size, k:k + crop_size] += window

    del vol_fbp_pad, wedge_dev
    torch.cuda.empty_cache()

    mask[mask == 0] = 1
    vol_est = vol_est / mask
    del mask
    vol_est = vol_est[:N1_pad, :N2_pad, :N3_pad]
    vol_est_np = vol_est.numpy().copy()
    del vol_est

    if pre_pad:
        vol_est_np = vol_est_np[pre_pad_size:, pre_pad_size:, pre_pad_size:]

    return vol_est_np


def _apply_wedge_batch(x: torch.Tensor, wedge: torch.Tensor) -> torch.Tensor:
    """Apply wedge mask via FFT  (B, D, H, W) → (B, D, H, W)."""
    B, D, H, W = x.shape
    mask_shape = tuple(wedge.shape)
    X = torch.fft.fftshift(torch.fft.fftn(x, s=mask_shape, dim=(-3, -2, -1)), dim=(-3, -2, -1))
    X = X * wedge
    out = torch.fft.ifftn(torch.fft.ifftshift(X, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real
    return out[..., :D, :H, :W]


# ---------------------------------------------------------------------------
# Volume I/O helpers
# ---------------------------------------------------------------------------

def _load_vol_normalized(path: Path, normalize: bool) -> torch.Tensor:
    """Load MRC → (D, H, W) CPU tensor with optional global normalization."""
    with mrcfile.open(str(path), permissive=True, mode="r") as mrc:
        vol_np = np.array(mrc.data, dtype=np.float32)
    vol_t = torch.from_numpy(np.moveaxis(vol_np, 0, 2))  # (Z,Y,X) → (D,H,W)
    if normalize:
        vol_t = (vol_t - vol_t.mean()) / (vol_t.std() + 1e-8)
    return vol_t


def _load_comparison(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    try:
        with mrcfile.open(str(path), permissive=True, mode="r") as mrc:
            vol_np = np.array(mrc.data, dtype=np.float32)
        vol_t = torch.from_numpy(np.moveaxis(vol_np, 0, 2))
        vol_t = (vol_t - vol_t.mean()) / (vol_t.std() + 1e-8)
        return vol_t.numpy()
    except Exception as exc:
        print(f"  WARNING: could not load {path}: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Per-volume inference  (function scope = automatic memory cleanup on return)
# ---------------------------------------------------------------------------

def _infer_one_volume(
    vol_idx: int,
    evn_path: Path,
    odd_path: Path | None,
    tilt_range: tuple[float, float] | None,
    model: nn.Module,
    cfg: RunEIPatchInferenceConfig,
    device: torch.device,
    stride: int,
    images_dir: Path,
    infer_downsample: int = 1,
) -> dict:
    tomo_dir  = evn_path.parent
    tomo_name = tomo_dir.name
    tilt_min, tilt_max = tilt_range if tilt_range is not None else (cfg.tilt_min, cfg.tilt_max)

    print(
        f"[patch-infer] vol{vol_idx:02d} ({tomo_name})  tilt=[{tilt_min:.1f}, {tilt_max:.1f}]°",
        flush=True,
    )

    physics = MissingWedge(
        tilt_max=float(tilt_max), tilt_min=float(tilt_min),
        crop_size=int(cfg.crop_size),
        use_spherical_support=bool(cfg.use_spherical_support),
        wedge_double_size=bool(cfg.wedge_double_size),
        wedge_low_support=float(cfg.wedge_low_support),
        ref_wedge_support=float(cfg.ref_wedge_support),
        device="cpu",
    )
    wedge_input = _symmetrize_and_binarize(physics.mask[:-1, :-1, :-1]).cpu()

    evn_vol = _load_vol_normalized(evn_path, cfg.normalize)
    odd_vol = _load_vol_normalized(odd_path, cfg.normalize) if odd_path is not None else evn_vol

    if infer_downsample > 1:
        evn_vol = torch.nn.functional.avg_pool3d(
            evn_vol.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=infer_downsample, stride=infer_downsample,
        ).squeeze()
        odd_vol = torch.nn.functional.avg_pool3d(
            odd_vol.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=infer_downsample, stride=infer_downsample,
        ).squeeze()
        print(f"  downsampled ×{infer_downsample} → {tuple(evn_vol.shape)}", flush=True)

    infer_kw = dict(
        model=model, wedge=wedge_input,
        crop_size=int(cfg.crop_size), stride=stride,
        infer_batch_size=int(cfg.infer_batch_size),
        device=device, pre_pad=bool(cfg.pre_pad),
    )
    print("  running inference on EVN ...", flush=True)
    recon_evn = patch_inference(evn_vol, **infer_kw)
    print("  running inference on ODD ...", flush=True)
    recon_odd = patch_inference(odd_vol, **infer_kw)

    recon_np = 0.5 * (recon_evn + recon_odd)

    # FSC between half-reconstructions
    evn_t = torch.from_numpy(recon_evn).unsqueeze(0).unsqueeze(0).to(device)
    odd_t = torch.from_numpy(recon_odd).unsqueeze(0).unsqueeze(0).to(device)
    fsc_curve = GpuFSC(evn_t.shape[-1], device=device)(evn_t, odd_t)
    px  = _read_pixel_sizes([evn_path], cfg.pixel_size_angstrom)[0]
    D   = int(recon_evn.shape[-1])
    k   = fsc_shell(fsc_curve, cfg.fsc_threshold)
    res = D * px / max(k, 1)
    fsc_str = f"FSC@{cfg.fsc_threshold}={res:.1f} Å (shell {k})"
    print(f"  {fsc_str}", flush=True)

    evn_np      = evn_vol.numpy()
    odd_np      = odd_vol.numpy()
    isonet_np   = _load_comparison(_find_mrc(tomo_dir, cfg.isonet_glob, cfg.isonet_fallback_glob))
    icecream_np = _load_comparison(_find_mrc(tomo_dir, cfg.icecream_glob))

    methods = [(evn_np, "EVN"), (odd_np, "ODD"), (isonet_np, "IsoNet"),
               (icecream_np, "IceCream"), (recon_np, "ours")]
    valid = [(v, lbl) for v, lbl in methods if v is not None]
    if valid:
        vcols, vlabels = zip(*valid)
        save_slice_figure(
            images_dir, epoch=0, vol_idx=vol_idx,
            cols=list(vcols), labels=list(vlabels),
            title=f"{tomo_name} | method comparison | {fsc_str}",
            subdir=".", fname=f"vol{vol_idx:02d}_methods.png",
        )

    save_fsc_figure(
        images_dir, epoch=0, fname=f"vol{vol_idx:02d}_fsc.png",
        fsc_curve=fsc_curve, res_shell=k, res_angstrom=res,
        title=f"{tomo_name} | FSC {res:.1f} Å",
        threshold=cfg.fsc_threshold, vol_size=D, pixel_size=px,
    )

    if cfg.save_recon_mrc:
        mrc_path = images_dir / f"vol{vol_idx:02d}_recon.mrc"
        _save_mrc(mrc_path, recon_np)
        print(f"  saved {mrc_path.name}", flush=True)

    return {
        "vol_idx": vol_idx, "tomo": tomo_name,
        "fsc_shell": int(k), "fsc_res_angstrom": float(res), "pixel_size": float(px),
    }


# ---------------------------------------------------------------------------
# Post-training inference  (called from run_patch after training)
# ---------------------------------------------------------------------------

def run_post_training_inference(
    datasets,
    raw_model: nn.Module,
    physics,
    ctx,
    output_dir: Path,
    *,
    crop_size: int,
    stride: int,
    infer_batch_size: int,
    infer_downsample: int = 1,
    tilt_min: float = -60.0,
    tilt_max: float = 60.0,
    use_spherical_support: bool = True,
    wedge_double_size: bool = True,
    wedge_low_support: float = 0.0,
    ref_wedge_support: float = 1.0,
    fsc_threshold: float = 0.143,
    pixel_size_angstrom: float | None = None,
    save_mrc: bool = False,
) -> None:
    """Sliding-window EVN+ODD inference over (train, val) datasets post-training.

    Distributes volumes across DDP ranks: each rank processes every world_size-th volume.
    """
    rank       = int(ctx.rank)
    world_size = int(ctx.world_size)
    device     = ctx.device

    wedge_cpu  = _symmetrize_and_binarize(physics.mask[:-1, :-1, :-1]).cpu()
    images_dir = ensure_dir(output_dir / "inference_images")
    recon_dir  = ensure_dir(output_dir / "reconstructions") if save_mrc else None
    _gpu_fsc   = GpuFSC(crop_size, device=device)
    raw_model.eval()

    if rank == 0:
        n_total = sum(len(ds.evn_vols) for _, ds in datasets)
        print(f"\n[ei-patch] Running inference on {n_total} volume(s) "
              f"(distributed across {world_size} GPU(s)) ...", flush=True)

    for split_label, ds in datasets:
        if rank == 0:
            print(f"[ei-patch] Reconstructing {split_label} volumes ({len(ds.evn_vols)}) ...", flush=True)
        for i in range(rank, len(ds.evn_vols), world_size):
            tilt = ds._tilt_ranges[i]
            if tilt is None:
                tilt = (tilt_min, tilt_max)
            tilt_min_i, tilt_max_i = tilt

            if tilt_min_i != tilt_min or tilt_max_i != tilt_max:
                physics_i = MissingWedge(
                    tilt_max=float(tilt_max_i), tilt_min=float(tilt_min_i),
                    crop_size=crop_size,
                    use_spherical_support=use_spherical_support,
                    wedge_double_size=wedge_double_size,
                    wedge_low_support=wedge_low_support,
                    ref_wedge_support=ref_wedge_support,
                    device="cpu",
                )
                wedge_i = _symmetrize_and_binarize(physics_i.mask[:-1, :-1, :-1]).cpu()
            else:
                wedge_i = wedge_cpu

            tomo_name = ds.evn_paths[i].parent.name
            evn_vol   = _load_vol_normalized(ds.evn_paths[i], ds.normalize)
            odd_vol   = _load_vol_normalized(ds.odd_paths[i], ds.normalize) if ds.odd_paths[i] is not None else evn_vol

            if infer_downsample > 1:
                evn_vol = torch.nn.functional.avg_pool3d(
                    evn_vol.unsqueeze(0).unsqueeze(0).float(),
                    kernel_size=infer_downsample, stride=infer_downsample,
                ).squeeze()
                odd_vol = torch.nn.functional.avg_pool3d(
                    odd_vol.unsqueeze(0).unsqueeze(0).float(),
                    kernel_size=infer_downsample, stride=infer_downsample,
                ).squeeze()
                print(f"  downsampled ×{infer_downsample} → {tuple(evn_vol.shape)}", flush=True)

            infer_kw = dict(model=raw_model, wedge=wedge_i, crop_size=crop_size,
                            stride=stride, infer_batch_size=infer_batch_size,
                            device=device, pre_pad=True)

            t0 = time.perf_counter()
            print(f"  [{tomo_name}] EVN inference ...", flush=True)
            recon_evn = patch_inference(evn_vol, **infer_kw)
            t_evn = time.perf_counter() - t0

            t1 = time.perf_counter()
            print(f"  [{tomo_name}] ODD inference ...", flush=True)
            recon_odd = patch_inference(odd_vol, **infer_kw)
            t_odd = time.perf_counter() - t1
            print(f"  [{tomo_name}] done  EVN={t_evn:.1f}s  ODD={t_odd:.1f}s  "
                  f"total={t_evn+t_odd:.1f}s", flush=True)

            recon = 0.5 * (recon_evn + recon_odd)

            recon_evn_t = torch.from_numpy(recon_evn).to(device)
            recon_odd_t = torch.from_numpy(recon_odd).to(device)
            fsc_curve_i = _gpu_fsc(recon_evn_t, recon_odd_t)
            del recon_evn_t, recon_odd_t, recon_evn, recon_odd
            torch.cuda.empty_cache()

            px_i    = _read_pixel_sizes([ds.evn_paths[i]], pixel_size_angstrom)[0]
            D_i     = int(recon.shape[-1])
            k_i     = fsc_shell(fsc_curve_i, threshold=fsc_threshold)
            res_i   = D_i * px_i / max(k_i, 1)
            fsc_str = f"FSC@{fsc_threshold}={res_i:.1f} Å (shell {k_i})"
            print(f"  [{tomo_name}] {fsc_str}", flush=True)

            save_fsc_figure(
                images_dir, epoch=0,
                fname=f"{split_label}_{tomo_name}_fsc.png",
                fsc_curve=fsc_curve_i, res_shell=k_i, res_angstrom=res_i,
                title=f"{tomo_name} ({split_label}) | {fsc_str}",
                threshold=fsc_threshold, vol_size=D_i,
                pixel_size=px_i if pixel_size_angstrom else None,
            )

            if save_mrc:
                evn_stem = ds.evn_paths[i].stem
                odd_stem = ds.odd_paths[i].stem if ds.odd_paths[i] is not None else evn_stem
                _save_mrc(recon_dir / f"{evn_stem}_{odd_stem}_recon.mrc", recon)
                print(f"  saved {evn_stem}_{odd_stem}_recon.mrc", flush=True)

            tomo_dir      = ds.evn_paths[i].parent
            icecream_path = _find_mrc(tomo_dir, "vol_*[Ii]cecream*", "vol_*[Ii]ce[Cc]ream*")
            isonet_path   = _find_mrc(tomo_dir, "vol_*[Ii]so[Nn]et*", "vol_*DDW*")

            recon_crop = _znorm(recon)
            del recon
            evn_crop   = _znorm(evn_vol.numpy())
            del evn_vol
            odd_crop   = _znorm(odd_vol.numpy())
            del odd_vol

            icecream_np = _load_comparison(icecream_path)
            isonet_np   = _load_comparison(isonet_path)

            cols, labels = [evn_crop, odd_crop, recon_crop], ["EVN", "ODD", "ours"]
            if icecream_np is not None:
                cols.append(_znorm(icecream_np))
                labels.append("IceCream")
            del icecream_np
            if isonet_np is not None:
                cols.append(_znorm(isonet_np))
                labels.append("IsoNet")
            del isonet_np

            save_slice_figure(
                images_dir, epoch=0, vol_idx=i,
                cols=cols, labels=labels,
                title=f"{tomo_name} ({split_label}) | {fsc_str}",
                subdir=".", fname=f"{split_label}_{tomo_name}_recon.png",
            )
            del cols

    if rank == 0:
        print(f"[ei-patch] Inference images saved to {images_dir}", flush=True)


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

def run_inference(cfg: RunEIPatchInferenceConfig) -> None:
    seed_everything(int(cfg.seed))

    output_dir = ensure_dir(cfg.output_dir)
    images_dir = ensure_dir(output_dir / "inference_images")
    dump_config_json(output_dir / "config.json", cfg.model_dump())

    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[patch-infer] device={device}", flush=True)

    data_cfg = EIPatchDataConfig(
        input_dir=cfg.input_dir,
        crop_size=int(cfg.crop_size),
        n_crops_per_vol=1,       # not used for inference
        batch_size=1,            # not used for inference
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=int(cfg.prefetch_factor),
        persistent_workers=bool(cfg.persistent_workers),
        max_train_vols=0,
        max_val_vols=int(cfg.max_infer_vols),
        seed=int(cfg.seed),
        normalize=bool(cfg.normalize),
        fallback_tilt_min=cfg.tilt_min,
        fallback_tilt_max=cfg.tilt_max,
    )

    data_bundle = build_ei_patch_dataloaders(data_cfg)
    val_ds = data_bundle.val_loader.dataset

    if not val_ds.evn_paths:
        raise RuntimeError(f"No volumes found in {cfg.input_dir}.")

    model, model_info = build_ei_model(
        cfg.model_type, cfg.unet_dropout, cfg.drunet_sigma, device,
    )

    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    print(
        f"[patch-infer] loaded {Path(cfg.checkpoint_path).name}  "
        f"model={model_info}  params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    stride = int(cfg.stride) if cfg.stride > 0 else cfg.crop_size // 2

    results = []
    for vol_idx in range(len(val_ds.evn_paths)):
        result = _infer_one_volume(
            vol_idx,
            val_ds.evn_paths[vol_idx],
            val_ds.odd_paths[vol_idx],
            val_ds._tilt_ranges[vol_idx],
            model, cfg, device, stride, images_dir,
            infer_downsample=int(cfg.infer_downsample),
        )
        results.append(result)
        torch.cuda.empty_cache()

    resolutions = [r["fsc_res_angstrom"] for r in results]
    if resolutions:
        res_arr = np.array(resolutions)
        mean_res, median_res = float(np.mean(res_arr)), float(np.median(res_arr))
        q1_res, q3_res = float(np.percentile(res_arr, 25)), float(np.percentile(res_arr, 75))

        save_resolution_histogram(
            images_dir, epoch=0, resolutions_angstrom=resolutions,
            mean_res=mean_res, median_res=median_res,
            q1_res=q1_res, q3_res=q3_res,
            threshold_label=str(cfg.fsc_threshold),
        )
        with open(output_dir / "results.json", "w") as f:
            json.dump({
                "fsc_threshold": cfg.fsc_threshold, "n_vols": len(resolutions),
                "mean_res_angstrom": mean_res, "median_res_angstrom": median_res,
                "q1_res_angstrom": q1_res, "q3_res_angstrom": q3_res,
                "per_volume": results,
            }, f, indent=2)

        print(
            f"\n[patch-infer] DONE  n={len(resolutions)}  "
            f"mean={mean_res:.1f} Å  median={median_res:.1f} Å  (lower=better)",
            flush=True,
        )
