"""infer_full.py — Inference-only evaluation for a trained EI full-volume model.

Loads a pretrained checkpoint and runs on the validation split (same seed as
training).  Per volume saves:
  vol{i}_methods.png  — EVN | ODD | IsoNet | IceCream | ours
  vol{i}_fsc.png      — FSC curve
  vol{i}_recon.mrc    — reconstructed volume  (save_recon_mrc=True, off by default)

Summary: resolution_histogram.png + results.json

Invoked via main.py  (local or SLURM):
    python main.py --config configs/conf_ei_inference.yml
"""
from __future__ import annotations

import json
from pathlib import Path

import mrcfile
import numpy as np
import torch
from deepinv.distributed import DistributedContext, distribute

from ..base_config import RunEIBaseConfig, _build_physics
from ..dataset.dataset_full import EIFullDataConfig, build_ei_full_dataloaders
from ..utils.utils import (
    GpuFSC,
    _find_mrc,
    _read_mrc_vol_size,
    _read_pixel_sizes,
    _save_mrc,
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

class RunEIFullInferenceConfig(RunEIBaseConfig):
    # ── Checkpoint ──────────────────────────────────────────────────────────
    checkpoint_path: str = ""

    # ── Data ────────────────────────────────────────────────────────────────
    output_dir: str = "./runs/inference_full"
    max_infer_vols: int = 5
    target_shape: tuple[int, int, int] | None = None

    # ── DataLoader ──────────────────────────────────────────────────────────
    num_workers: int = 1
    prefetch_factor: int = 1

    # ── Distributed model tiling (must match training config) ────────────────
    patch_size: tuple[int, int, int] = (64, 64, 64)
    overlap: tuple[int, int, int] = (8, 8, 8)
    max_batch_size: int | None = 2
    checkpoint_batches: str | int | None = "auto"

    # ── Comparison volume globs (searched inside each tomo_* directory) ─────
    icecream_glob: str = "vol_*[Ii]cecream*"
    isonet_glob: str = "vol_*[Ii]so[Nn]et*"
    isonet_fallback_glob: str = "vol_*DDW*"

    # ── Output options ───────────────────────────────────────────────────────
    save_recon_mrc: bool = False

    @classmethod
    def from_yaml(cls, conf: dict) -> "RunEIFullInferenceConfig":
        return cls.model_validate(cls._flat_from_yaml(conf, "demo-cryo-ei-inference"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mrc_vol(path: Path, target_shape: tuple | None = None) -> np.ndarray:
    """Load MRC, reorder axes, optional resample, centre-crop to cube, normalise.

    Mirrors ``CryoEIFullDataset._load_and_prepare`` but returns a float32
    numpy array of shape (D, H, W) instead of a tensor.
    """
    with mrcfile.open(str(path), permissive=True, mode="r") as f:
        vol_np = np.array(f.data, dtype=np.float32)   # (Z, Y, X)

    vol = torch.from_numpy(np.moveaxis(vol_np, 0, 2))  # (D, H, W)

    if target_shape is not None:
        vol = torch.nn.functional.interpolate(
            vol.unsqueeze(0).unsqueeze(0),
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    D, H, W = vol.shape
    S = min(D, H, W)
    d0, h0, w0 = (D - S) // 2, (H - S) // 2, (W - S) // 2
    vol = vol[d0:d0 + S, h0:h0 + S, w0:w0 + S]

    mu, sigma = vol.mean(), vol.std()
    vol = (vol - mu) / (sigma + 1e-8)
    return vol.numpy()   # (D, H, W) float32


# ---------------------------------------------------------------------------
# Inference entry-point
# ---------------------------------------------------------------------------

def run_inference(cfg: RunEIFullInferenceConfig) -> None:
    seed_everything(int(cfg.seed))

    output_dir = ensure_dir(cfg.output_dir)
    images_dir = ensure_dir(output_dir / "inference_images")
    dump_config_json(output_dir / "config.json", cfg.model_dump())

    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be set in the config.")

    data_cfg = EIFullDataConfig(
        input_dir=cfg.input_dir,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=int(cfg.prefetch_factor),
        persistent_workers=bool(cfg.persistent_workers),
        max_train_vols=0,
        max_val_vols=int(cfg.max_infer_vols),
        seed=int(cfg.seed),
        target_shape=cfg.target_shape,
        fallback_tilt_min=cfg.tilt_min,
        fallback_tilt_max=cfg.tilt_max,
    )

    with DistributedContext(seed=int(cfg.seed), seed_offset=False, cleanup=True) as ctx:
        rank = int(ctx.rank)

        data_bundle = build_ei_full_dataloaders(data_cfg)
        val_loader = data_bundle.val_loader
        val_ds = val_loader.dataset

        if not val_ds.evn_paths:
            raise RuntimeError(f"No volumes found in {cfg.input_dir} — check input_dir and globs.")

        if cfg.target_shape is not None:
            vol_size = int(min(cfg.target_shape))
            print(f"[inference] target_shape={cfg.target_shape} → physics crop_size={vol_size}", flush=True)
        else:
            vol_size = _read_mrc_vol_size(val_ds.evn_paths[0])
            print(f"[inference] auto vol_size={vol_size}  (from {val_ds.evn_paths[0].name})", flush=True)

        physics = _build_physics(cfg, vol_size, ctx.device)

        # Build model and load checkpoint before distribute — mirrors run_full
        wrapper, model_info = build_ei_model(
            cfg.model_type, cfg.unet_dropout, cfg.drunet_sigma, ctx.device,
        )

        ckpt_path = Path(cfg.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location=ctx.device, weights_only=True)
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        if any(k.startswith("processor.") for k in state):
            state = {k.removeprefix("processor."): v for k, v in state.items()}
            if rank == 0:
                print("[inference] stripped 'processor.' prefix from checkpoint keys", flush=True)
        wrapper.load_state_dict(state, strict=True)

        backbone = distribute(wrapper, ctx, type_object="denoiser",
                              patch_size=tuple(int(v) for v in cfg.patch_size),
                              overlap=tuple(int(v) for v in cfg.overlap),
                              tiling_dims=(-3, -2, -1),
                              max_batch_size=cfg.max_batch_size,
                              checkpoint_batches=cfg.checkpoint_batches)
        backbone.eval()

        pixel_sizes = _read_pixel_sizes(val_ds.evn_paths, fallback=cfg.pixel_size_angstrom)

        if rank == 0:
            n_params = sum(p.numel() for p in backbone.parameters())
            print(
                f"[inference] loaded checkpoint {ckpt_path.name}  "
                f"model={model_info}  params={n_params:,}  vol_size={vol_size}",
                flush=True,
            )

        # ── Inference loop ────────────────────────────────────────────────────
        resolutions: list[float] = []
        gpu_fsc: GpuFSC | None = None
        results: list[dict] = []

        for vol_idx, (evn, odd, tilt_params) in enumerate(val_loader):
            evn = evn.to(ctx.device)   # (1, 1, D, H, W)
            odd = odd.to(ctx.device)

            physics.update_parameters(
                tilt_min=tilt_params["tilt_min"],
                tilt_max=tilt_params["tilt_max"],
            )
            if rank == 0 and vol_idx == 0:
                print(
                    f"[physics] vol={vol_idx}  "
                    f"tilt_min={float(tilt_params['tilt_min']):.1f}°  "
                    f"tilt_max={float(tilt_params['tilt_max']):.1f}°",
                    flush=True,
                )

            with torch.no_grad():
                f_evn_t = backbone(evn)
                f_odd_t = backbone(odd)
                recon_t = 0.5 * (backbone(physics.A(f_evn_t)) + backbone(physics.A(f_odd_t)))

            if hasattr(ctx.device, "type") and ctx.device.type == "cuda":
                torch.cuda.synchronize()

            # ── FSC ───────────────────────────────────────────────────────────
            if gpu_fsc is None:
                gpu_fsc = GpuFSC(f_evn_t.shape[-1], device=f_evn_t.device)

            fsc_curve = gpu_fsc(f_evn_t, f_odd_t)
            D   = int(f_evn_t.shape[-1])
            px  = pixel_sizes[vol_idx] if vol_idx < len(pixel_sizes) else 1.0
            k   = fsc_shell(fsc_curve, cfg.fsc_threshold)
            res = D * px / max(k, 1)
            resolutions.append(res)

            tomo_name = val_ds.evn_paths[vol_idx].parent.name

            if rank == 0:
                print(
                    f"[inference] vol{vol_idx:02d} ({tomo_name})  "
                    f"FSC@{cfg.fsc_threshold}={res:.1f} Å  (shell {k})",
                    flush=True,
                )

            # ── numpy conversion ──────────────────────────────────────────────
            evn_np   = evn.squeeze().cpu().numpy()
            odd_np   = odd.squeeze().cpu().numpy()
            recon_np = recon_t.squeeze().cpu().numpy()

            # ── IsoNet / IceCream comparison volumes ──────────────────────────
            tomo_dir      = val_ds.evn_paths[vol_idx].parent
            isonet_path   = _find_mrc(tomo_dir, cfg.isonet_glob, cfg.isonet_fallback_glob)
            icecream_path = _find_mrc(tomo_dir, cfg.icecream_glob)

            isonet_np: np.ndarray | None = None
            icecream_np: np.ndarray | None = None
            if rank == 0:
                if isonet_path is not None:
                    try:
                        isonet_np = _load_mrc_vol(isonet_path, cfg.target_shape)
                        print(f"  [isonet]   {isonet_path.name}", flush=True)
                    except Exception as exc:
                        print(f"  [isonet]   FAILED to load {isonet_path}: {exc}", flush=True)
                else:
                    print(f"  [isonet]   not found in {tomo_dir}", flush=True)

                if icecream_path is not None:
                    try:
                        icecream_np = _load_mrc_vol(icecream_path, cfg.target_shape)
                        print(f"  [icecream] {icecream_path.name}", flush=True)
                    except Exception as exc:
                        print(f"  [icecream] FAILED to load {icecream_path}: {exc}", flush=True)
                else:
                    print(f"  [icecream] not found in {tomo_dir}", flush=True)

            if rank == 0:
                # ── Figure: methods — EVN | ODD | IsoNet | IceCream | ours ──
                methods_cols   = [evn_np, odd_np, isonet_np, icecream_np, recon_np]
                methods_labels = ["EVN",  "ODD",  "IsoNet",  "IceCream",  "ours"]
                valid_pairs = [(v, lbl) for v, lbl in zip(methods_cols, methods_labels) if v is not None]
                valid_cols, valid_labels = zip(*valid_pairs) if valid_pairs else ([], [])
                save_slice_figure(
                    images_dir, epoch=0, vol_idx=vol_idx,
                    cols=list(valid_cols),
                    labels=list(valid_labels),
                    title=f"{tomo_name} | method comparison",
                    subdir=".",
                    fname=f"vol{vol_idx:02d}_methods.png",
                )

                # ── FSC figure ────────────────────────────────────────────────
                save_fsc_figure(
                    images_dir, epoch=0,
                    fname=f"vol{vol_idx:02d}_fsc.png",
                    fsc_curve=fsc_curve, res_shell=k, res_angstrom=res,
                    title=f"{tomo_name} | FSC  {res:.1f} Å",
                    threshold=cfg.fsc_threshold,
                    vol_size=D, pixel_size=px,
                )

                # ── Optional: save recon MRC ──────────────────────────────────
                if cfg.save_recon_mrc:
                    recon_mrc_path = images_dir / f"vol{vol_idx:02d}_recon.mrc"
                    _save_mrc(recon_mrc_path, recon_np)
                    print(f"  [recon mrc] saved {recon_mrc_path.name}", flush=True)

            results.append({
                "vol_idx":          vol_idx,
                "tomo":             tomo_name,
                "fsc_shell":        int(k),
                "fsc_res_angstrom": float(res),
                "pixel_size":       float(px),
            })

        # ── Summary ───────────────────────────────────────────────────────────
        if rank == 0 and resolutions:
            res_arr    = np.array(resolutions)
            mean_res   = float(np.mean(res_arr))
            median_res = float(np.median(res_arr))
            q1_res     = float(np.percentile(res_arr, 25))
            q3_res     = float(np.percentile(res_arr, 75))

            save_resolution_histogram(
                images_dir, epoch=0,
                resolutions_angstrom=resolutions,
                mean_res=mean_res, median_res=median_res,
                q1_res=q1_res, q3_res=q3_res,
                threshold_label=str(cfg.fsc_threshold),
            )

            summary = {
                "fsc_threshold":       cfg.fsc_threshold,
                "n_vols":              len(resolutions),
                "mean_res_angstrom":   mean_res,
                "median_res_angstrom": median_res,
                "q1_res_angstrom":     q1_res,
                "q3_res_angstrom":     q3_res,
                "per_volume":          results,
            }
            with open(output_dir / "results.json", "w") as f:
                json.dump(summary, f, indent=2)

            print(
                f"\n[inference] DONE  n={len(resolutions)}  "
                f"mean={mean_res:.1f} Å  median={median_res:.1f} Å  "
                f"Q1={q1_res:.1f} Å  Q3={q3_res:.1f} Å  (lower=better)",
                flush=True,
            )
