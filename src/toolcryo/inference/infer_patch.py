"""infer_patch.py — Inference for a trained patch-based EI model.

Matches icecream's inference_util.inference exactly:
  1. Pre-pad the volume.
  2. Slide a crop_size³ window with stride, extracting overlapping patches.
  3. For each patch:  output = f(A(f(patch)))  — model applied twice, wedge in between.
  4. Reassemble via window-weighted overlap averaging.
  5. Average EVN and ODD reconstructions:  recon = 0.5 * (result_evn + result_odd).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from ..base_config import RunEIBaseConfig, amp_dtype_from_str
from ..dataset.dataset_patch import EIPatchDataConfig, build_ei_patch_dataloaders
from ..physics import MissingWedge
from ..losses.losses_equivariant_wedge import _initialize_window, _symmetrize_and_binarize
from ..registry import get_preset
from ..utils.utils import (
    GpuFSC,
    append_fsc_row,
    _apply_wedge_batch,
    _find_mrc,
    _read_pixel_sizes,
    _save_mrc,
    _znorm,
    fsc_resolution,
    dump_config_json,
    ensure_dir,
    load_mrc_volume,
    psnr,
    sharpness_3d,
    seed_everything,
)
from ..utils.plot import save_fsc_figure, save_slice_figure


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class RunEIPatchInferenceConfig(RunEIBaseConfig):
    # ── Checkpoint ──────────────────────────────────────────────────────────
    checkpoint_paths: list[str] = []

    # ── Data ────────────────────────────────────────────────────────────────
    output_dir: str = "./runs/inference_patch"
    max_infer_vols: int = 5
    normalize: bool = True

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
    # Ground truth, when the dataset ships one (dataset/synthetic). Scored by
    # PSNR into fsc_rank*.csv, not just drawn — the point of a synthetic tomogram.
    gt_glob: str = "vol_*_[Gg][Tt].mrc"

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
    amp_dtype: torch.dtype | None = None,
    return_1pass: bool = False,
) -> np.ndarray:
    """Sliding-window f(A(f(.))) inference — mirrors icecream's inference_util.inference exactly.

    :param vol: (D, H, W) CPU float32 tensor (globally normalised).
    :param wedge: wedge_input mask (mask_size³) on CPU.
    :param amp_dtype: autocast dtype, or ``None`` for pure fp32.
    :param return_1pass: also return the ``f(.)`` result from before the round
        trip, as ``(two_pass, one_pass)``. Default ``False`` keeps the
        single-array return. Costs one extra accumulator, no extra model passes.
    :returns: (D, H, W) float32 array, or a ``(two_pass, one_pass)`` tuple when
        ``return_1pass``.
    """
    pre_pad_size = crop_size // 4

    use_amp = device.type == "cuda" and amp_dtype is not None

    vol_fbp = vol.to(amp_dtype) if use_amp else vol
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

    vol_est_1 = torch.zeros_like(vol_est) if return_1pass else None

    positions = [
        (i, j, k)
        for i in range(0, N1, stride)
        for j in range(0, N2, stride)
        for k in range(0, N3, stride)
        if i + crop_size <= N1 and j + crop_size <= N2 and k + crop_size <= N3
    ]

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(positions), infer_batch_size):
            batch_positions = positions[batch_start:batch_start + infer_batch_size]
            batch = torch.stack([
                vol_fbp_pad[i:i + crop_size, j:j + crop_size, k:k + crop_size]
                for i, j, k in batch_positions
            ]).to(device)
            with torch.autocast(device_type=device.type,
                                dtype=amp_dtype or torch.float16, enabled=use_amp):
                output = model(batch[:, None])[:, 0]        # f(crop)
            output = output.float()
            out1_cpu = output.detach().cpu() if return_1pass else None
            output = _apply_wedge_batch(output, wedge_dev)  # A(f(crop))
            with torch.autocast(device_type=device.type,
                                dtype=amp_dtype or torch.float16, enabled=use_amp):
                output = model(output[:, None])[:, 0]       # f(A(f(crop)))
            out_cpu = output.detach().cpu()
            for b, (i, j, k) in enumerate(batch_positions):
                vol_est[i:i + crop_size, j:j + crop_size, k:k + crop_size] += out_cpu[b] * window
                mask[  i:i + crop_size, j:j + crop_size, k:k + crop_size] += window
                if return_1pass:
                    vol_est_1[i:i + crop_size, j:j + crop_size, k:k + crop_size] += out1_cpu[b] * window

    del vol_fbp_pad, wedge_dev
    torch.cuda.empty_cache()

    mask[mask == 0] = 1
    vol_est = vol_est / mask

    # Same normalise / crop / unpad as the 2-pass result, so the two align.
    vol_est_1_np = None
    if return_1pass:
        vol_est_1 = (vol_est_1 / mask)[:N1_pad, :N2_pad, :N3_pad]
        vol_est_1_np = vol_est_1.numpy().copy()
        del vol_est_1
        if pre_pad:
            vol_est_1_np = vol_est_1_np[pre_pad_size:, pre_pad_size:, pre_pad_size:]

    del mask
    vol_est = vol_est[:N1_pad, :N2_pad, :N3_pad]
    vol_est_np = vol_est.numpy().copy()
    del vol_est

    if pre_pad:
        vol_est_np = vol_est_np[pre_pad_size:, pre_pad_size:, pre_pad_size:]

    return (vol_est_np, vol_est_1_np) if return_1pass else vol_est_np


# ---------------------------------------------------------------------------
# Volume I/O helpers
# ---------------------------------------------------------------------------

def _load_vol_normalized(path: Path, normalize: bool) -> torch.Tensor:
    """Load MRC → (D, H, W) CPU tensor with optional global normalization."""
    vol_t = torch.from_numpy(load_mrc_volume(path, order="native"))  # (D, H, W)
    if normalize:
        vol_t = (vol_t - vol_t.mean()) / (vol_t.std() + 1e-8)
    return vol_t


def _load_comparison(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    try:
        vol_t = torch.from_numpy(load_mrc_volume(path, order="native"))
        vol_t = (vol_t - vol_t.mean()) / (vol_t.std() + 1e-8)
        return vol_t.numpy()
    except Exception as exc:
        print(f"  WARNING: could not load {path}: {exc}", flush=True)
        return None


def _find_gt(tomo_dir: Path, glob: str, shape) -> tuple[np.ndarray | None, str]:
    """Ground-truth volume resampled to ``shape``, and the file name it came from.

    The resample is not cosmetic: ``psnr`` raises on a shape mismatch, so a
    downsampled or cropped inference run would lose the score entirely rather
    than report it against a stretched reference.
    """
    path = _find_mrc(tomo_dir, glob)
    gt = _load_comparison(path)
    if gt is not None and gt.shape != tuple(shape):
        gt = nn.functional.interpolate(
            torch.from_numpy(gt)[None, None], size=tuple(shape),
            mode="trilinear", align_corners=False).squeeze().numpy()
    return gt, (path.name if path is not None else "")


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
    gt_glob: str = "vol_*_[Gg][Tt].mrc",
    save_mrc: bool = False,
    save_fsc_curves: bool = True,
    amp_dtype: torch.dtype | None = None,
    checkpoint: str = "",
    mode: str = "train",
) -> None:
    """Sliding-window EVN+ODD inference over (train, val) datasets.

    Called post-training and, once per checkpoint, by standalone ``run_inference``.
    Distributes volumes across DDP ranks: each rank processes every world_size-th volume.
    """
    rank       = int(ctx.rank)
    world_size = int(ctx.world_size)
    device     = ctx.device

    wedge_cpu  = _symmetrize_and_binarize(physics.mask[:-1, :-1, :-1]).cpu()
    images_dir = ensure_dir(output_dir / "inference_images" / checkpoint)
    recon_dir  = ensure_dir(output_dir / "reconstructions") if save_mrc else None
    _gpu_fsc   = GpuFSC(device=device)
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
                            device=device, pre_pad=True, amp_dtype=amp_dtype,
                            return_1pass=True)

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

            (recon_evn, recon_evn_1), (recon_odd, recon_odd_1) = recon_evn, recon_odd
            recon   = 0.5 * (recon_evn + recon_odd)
            recon_1 = 0.5 * (recon_evn_1 + recon_odd_1)
            del recon_evn_1, recon_odd_1

            recon_evn_t = torch.from_numpy(recon_evn).to(device)
            recon_odd_t = torch.from_numpy(recon_odd).to(device)
            fsc_curve_i = _gpu_fsc(recon_evn_t, recon_odd_t)
            del recon_evn_t, recon_odd_t, recon_evn, recon_odd
            torch.cuda.empty_cache()

            px_i    = _read_pixel_sizes([ds.evn_paths[i]], pixel_size_angstrom)[0]
            k_i, res_i, D_i = fsc_resolution(fsc_curve_i, recon.shape, px_i, fsc_threshold)
            fsc_str = f"FSC@{fsc_threshold}={res_i:.1f} Å (shell {k_i})"
            print(f"  [{tomo_name}] {fsc_str}", flush=True)

            # Before the row is written, since its PSNR goes into that row.
            gt_np, gt_name = _find_gt(ds.evn_paths[i].parent, gt_glob, recon.shape)
            psnr_gt = psnr(recon, gt_np) if gt_np is not None else ""
            psnr_1p = psnr(recon_1, gt_np) if gt_np is not None else ""
            del recon_1
            if gt_np is not None:
                print(f"  [{tomo_name}] PSNR vs {gt_name} = {psnr_gt:.2f} dB", flush=True)

            # Volumes are sharded across ranks, so each rank writes its own file.
            append_fsc_row(output_dir / "metrics" / f"fsc_rank{rank}.csv",
                           curve=fsc_curve_i if save_fsc_curves else None,
                           mode=mode, regime="patch", split=split_label,
                           checkpoint=checkpoint,
                           vol_idx=i, tomo=tomo_name, pixel_size=px_i, n_ref=D_i,
                           fsc_threshold=fsc_threshold,
                           fsc_shell=int(k_i), fsc_res_angstrom=float(res_i),
                           psnr_gt=psnr_gt, psnr_1pass_gt=psnr_1p, psnr_ref=gt_name,
                           sharpness=sharpness_3d(torch.from_numpy(recon).to(device)))

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

            # Column order matches full inference / training: EVN, ODD, comparisons, ours last.
            cols, labels = [evn_crop, odd_crop], ["EVN", "ODD"]
            if isonet_np is not None:
                cols.append(_znorm(isonet_np))
                labels.append("IsoNet")
            del isonet_np
            if icecream_np is not None:
                cols.append(_znorm(icecream_np))
                labels.append("IceCream")
            del icecream_np
            if gt_np is not None:
                cols.append(_znorm(gt_np))
                labels.append("GT")
            del gt_np
            cols.append(recon_crop)
            labels.append("ours")

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

    # No process group: each rank takes every world_size-th volume and writes
    # its own fsc_rank{r}.csv. submitit exports these; unset = single GPU.
    rank, local_rank = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    ctx = SimpleNamespace(rank=rank, world_size=world_size, device=device)

    output_dir = ensure_dir(cfg.output_dir)
    if rank == 0:
        dump_config_json(output_dir / "config.json", cfg.model_dump())

    if not cfg.checkpoint_paths:
        raise ValueError("checkpoint_paths must be set.")

    if cfg.train_names:
        print(
            f"[patch-infer] WARNING: train_names={cfg.train_names} is ignored — inference "
            f"reads the val split. Use val_names to pin the volumes to evaluate.",
            flush=True,
        )

    print(f"[patch-infer] rank {rank}/{world_size}  device={device}", flush=True)

    # Only the dataset's path lists are used below; the DataLoader is never
    # iterated (volumes are read in run_post_training_inference), so the
    # DataLoader knobs are left at their defaults.
    data_cfg = EIPatchDataConfig(
        input_dir=cfg.input_dir,
        crop_size=int(cfg.crop_size),
        n_crops_per_vol=1,       # not used for inference
        batch_size=1,            # not used for inference
        max_train_vols=0,
        max_val_vols=int(cfg.max_infer_vols),
        seed=int(cfg.seed),
        val_names=cfg.val_names,
        normalize=bool(cfg.normalize),
        fallback_tilt_min=cfg.tilt_min,
        fallback_tilt_max=cfg.tilt_max,
    )

    data_bundle = build_ei_patch_dataloaders(data_cfg)
    val_ds = data_bundle.val_loader.dataset

    if not val_ds.evn_paths:
        raise RuntimeError(f"No volumes found in {cfg.input_dir}.")

    preset = get_preset(cfg.preset)
    model, model_info = preset["model"](
        cfg.model_type, cfg.unet_dropout, cfg.drunet_sigma, device,
    )

    stride = int(cfg.stride) if cfg.stride > 0 else cfg.crop_size // 2
    physics = MissingWedge(
        tilt_max=float(cfg.tilt_max), tilt_min=float(cfg.tilt_min),
        crop_size=int(cfg.crop_size),
        use_spherical_support=bool(cfg.use_spherical_support),
        wedge_double_size=bool(cfg.wedge_double_size),
        wedge_low_support=float(cfg.wedge_low_support),
        ref_wedge_support=float(cfg.ref_wedge_support),
        device="cpu",
    )

    for ckpt_path in cfg.checkpoint_paths:
        # map to CPU: the file also carries optimizer state (~2x the weights) that
        # inference never uses, and load_state_dict copies CPU->GPU params directly.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}   # torch.compile wrapper
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
            print("[patch-infer] stripped 'module.' prefix from checkpoint keys", flush=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        del ckpt, state

        if ctx.rank == 0:
            print(
                f"\n[patch-infer] loaded {Path(ckpt_path).name}  "
                f"model={model_info}  params={sum(p.numel() for p in model.parameters()):,}",
                flush=True,
            )

        run_post_training_inference(
            [("val", val_ds)], model, physics, ctx, output_dir,
            crop_size=int(cfg.crop_size),
            stride=stride,
            infer_batch_size=int(cfg.infer_batch_size),
            infer_downsample=int(cfg.infer_downsample),
            tilt_min=float(cfg.tilt_min),
            tilt_max=float(cfg.tilt_max),
            use_spherical_support=bool(cfg.use_spherical_support),
            wedge_double_size=bool(cfg.wedge_double_size),
            wedge_low_support=float(cfg.wedge_low_support),
            ref_wedge_support=float(cfg.ref_wedge_support),
            fsc_threshold=float(cfg.fsc_threshold),
            pixel_size_angstrom=cfg.pixel_size_angstrom,
            gt_glob=cfg.gt_glob,
            save_mrc=bool(cfg.save_recon_mrc),
            save_fsc_curves=bool(cfg.save_fsc_curves),
            amp_dtype=amp_dtype_from_str(cfg.mixed_precision),
            checkpoint=Path(ckpt_path).stem,
            mode="inference",
        )

    if ctx.rank == 0:
        print(f"\n[patch-infer] DONE  {len(cfg.checkpoint_paths)} checkpoint(s) -> "
              f"{output_dir / 'metrics'}/fsc_rank*.csv", flush=True)
