"""infer_full.py — Inference-only evaluation for trained EI full-volume models.

Loads one or more checkpoints (``cfg.checkpoint_paths``) and runs each on the
volumes selected by ``val_names`` (or, when unset, the first ``max_infer_vols``
volumes of a seeded shuffle over ``input_dir``).  Note this split is
re-derived here, not inherited from the training run: to evaluate the exact
volumes used for training, list them in ``val_names``.  Per checkpoint, per
volume, saves (under ``inference_images/<checkpoint_name>/``):
  vol{i}_methods.png  — EVN | ODD | IsoNet | IceCream | ours
  vol{i}_fsc.png      — FSC curve
  vol{i}_recon.mrc    — reconstructed volume  (save_recon_mrc=True, off by default)
  resolution_histogram.png — per-checkpoint FSC summary

Dataset/physics are built once and reused across every checkpoint; the model
is rebuilt fresh per checkpoint (its weights, and for ``unrolled`` even its
internal shape, depend on the checkpoint) and dropped afterward.

Combined summary across all checkpoints: results.json.

Invoked via main.py  (local or SLURM):
    python main.py --config configs/conf_ei_inference.yml
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from deepinv.distributed import DistributedContext

from ..base_config import RunEIBaseConfig
from ..dataset.dataset_full import EIFullDataConfig, build_ei_full_dataloaders
from ..models import build_distributed_denoiser
from ..registry import get_preset
from ..utils.utils import (
    GpuFSC,
    append_fsc_row,
    _find_mrc,
    _read_mrc_vol_shape,
    _read_pixel_sizes,
    _save_mrc,
    _znorm,
    fsc_resolution,
    dump_config_json,
    ensure_dir,
    load_mrc_volume,
    recon_panels,
    to_canonical_np,
    seed_everything,
)
from ..utils.plot import save_fsc_figure, save_resolution_histogram, save_slice_figure


class _InferenceModel:
    """Minimal shim so ``preset["forward"]`` (written for ``dinv.Trainer``)
    can be called standalone, without a full Trainer instance."""

    def __init__(self, model) -> None:
        self.model = model

    def model_inference(self, y, physics, x=None, train=False, **kwargs):
        self.model.eval()
        with torch.no_grad():
            return self.model(y, physics, **kwargs)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class RunEIFullInferenceConfig(RunEIBaseConfig):
    # ── Checkpoint(s) ───────────────────────────────────────────────────────
    # One run evaluates every checkpoint here, in order, over the same
    # volumes — for comparing checkpoints (epochs, runs) without re-loading
    # data each time. A single-entry list evaluates just one, as before.
    checkpoint_paths: list[str] = []

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
    # Angle-sharded physics — same meaning as RunEIFullConfig.num_operators
    # (null = one full operator per rank, "auto" = one per rank, int = that
    # many). Declared here too because this config inherits RunEIBaseConfig,
    # not RunEIFullConfig, so without it build_tomography_physics' getattr
    # would silently read None and inference could never shard.
    num_operators: int | Literal["auto"] | None = None

    # ── Unrolled preset only (must match the training config) ────────────────
    n_iter: int = 4
    init_stepsize: float = 0.9
    train_algo_params: bool = True
    stepsize_learning_rate: float | None = None

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
    """Load MRC, reorder axes, optional resample, normalise — no crop.

    Returns a float32 numpy array in the canonical ``(Y, X, Z)`` order — the
    same order used by ``target_shape`` and by every other figure column, for
    both presets (unrolled reconstructions are brought back from astra order
    by ``to_canonical_np`` before they reach the figure). No crop: inference
    always evaluates the whole volume, and this must match its shape.
    """
    vol = torch.from_numpy(load_mrc_volume(path, order="native"))

    if target_shape is not None:
        vol = torch.nn.functional.interpolate(
            vol.unsqueeze(0).unsqueeze(0),
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    mu, sigma = vol.mean(), vol.std()
    vol = (vol - mu) / (sigma + 1e-8)
    return vol.numpy()   # (D, H, W) float32


# ---------------------------------------------------------------------------
# Inference entry-point
# ---------------------------------------------------------------------------

def run_inference(cfg: RunEIFullInferenceConfig) -> None:
    seed_everything(int(cfg.seed))

    output_dir = ensure_dir(cfg.output_dir)
    dump_config_json(output_dir / "config.json", cfg.model_dump())

    if not cfg.checkpoint_paths:
        raise ValueError("checkpoint_paths must be set in the config.")

    if cfg.train_names:
        print(
            f"[inference] WARNING: train_names={cfg.train_names} is ignored — inference "
            f"reads the val split. Use val_names to pin the volumes to evaluate.",
            flush=True,
        )

    is_tomo     = cfg.preset in ("unrolled", "tomo_ei")   # TomographyEM physics
    is_unrolled = cfg.preset == "unrolled"                # PGD-unfold model

    data_cfg = EIFullDataConfig(
        input_dir=cfg.input_dir,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=int(cfg.prefetch_factor),
        persistent_workers=bool(cfg.persistent_workers),
        max_train_vols=0,
        max_val_vols=int(cfg.max_infer_vols),
        seed=int(cfg.seed),
        val_names=cfg.val_names,
        target_shape=cfg.target_shape,
        fallback_tilt_min=cfg.tilt_min,
        fallback_tilt_max=cfg.tilt_max,
        data_source="measurement" if is_tomo else "fbp",
    )

    with DistributedContext(seed=int(cfg.seed), seed_offset=False, cleanup=True) as ctx:
        rank = int(ctx.rank)

        data_bundle = build_ei_full_dataloaders(data_cfg)
        val_loader = data_bundle.val_loader
        val_ds = val_loader.dataset

        if not val_ds.evn_paths:
            raise RuntimeError(f"No volumes found in {cfg.input_dir} — check input_dir and globs.")

        preset = get_preset(cfg.preset)

        # Physics doesn't depend on checkpoint content — build once, reused
        # across every checkpoint in cfg.checkpoint_paths.
        if is_tomo:
            # See run.py's matching guard: tomo_ei calls physics.fbp() every
            # eval (half_set_recon), which sharded physics doesn't implement.
            if not is_unrolled and cfg.num_operators is not None:
                raise ValueError(
                    f"num_operators={cfg.num_operators!r} is not supported for preset "
                    f"{cfg.preset!r}: it calls physics.fbp() (half_set_recon), which "
                    f"sharded (distributed) physics does not implement. Set "
                    f"num_operators: null for this preset, or use preset: unrolled."
                )
            physics = preset["physics"](cfg, val_ds.evn_paths, val_ds.odd_paths, ctx.device, ctx)
        else:
            # No crop_size guess here — evaluation always sees the whole
            # volume, so physics is built at that volume's actual shape.
            # Later volumes of a different shape are handled per-batch below
            # via update_parameters(vol_shape=...).
            if cfg.target_shape is not None:
                vol_shape = tuple(int(v) for v in cfg.target_shape)
                print(f"[inference] target_shape={cfg.target_shape} → physics shape={vol_shape}", flush=True)
            else:
                vol_shape = _read_mrc_vol_shape(val_ds.evn_paths[0])
                print(f"[inference] auto vol_shape={vol_shape}  (from {val_ds.evn_paths[0].name})", flush=True)
            physics = preset["physics"](cfg, vol_shape, ctx.device)

        pixel_sizes = _read_pixel_sizes(val_ds.evn_paths, fallback=cfg.pixel_size_angstrom)
        gpu_fsc: GpuFSC | None = None
        all_rows: list[dict] = []

        for ckpt_path_str in cfg.checkpoint_paths:
            ckpt_path = Path(ckpt_path_str)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            ckpt_name = ckpt_path.stem
            images_dir = ensure_dir(output_dir / "inference_images" / ckpt_name)

            if is_tomo and is_unrolled:
                # preset["model"] (build_unrolled_model) loads the checkpoint
                # (denoiser+stepsize+g_param) and tiles the denoiser
                # internally, so no separate load/distribute code is needed.
                cfg.pretrained_ckpt = str(ckpt_path)
                model, model_info = preset["model"](cfg, physics, ctx)
            else:
                # missingwedge_ei / tomo_ei: plain denoiser + distribute,
                # loading this checkpoint directly. Defaults to no permutation
                # — checkpoint_paths is assumed to already be in the axis
                # order this preset expects (produced by training under this
                # same preset). Set cfg.permute_native_to_astra=True to
                # evaluate a native-order missingwedge_ei/patch checkpoint
                # directly under tomo_ei inference instead.
                permute = cfg.permute_native_to_astra if cfg.permute_native_to_astra is not None else False
                model, model_info = build_distributed_denoiser(
                    cfg, ctx, rank, ckpt_path, permute_native_to_astra=permute, log_prefix="inference")
            model.eval()

            if rank == 0:
                n_params = sum(p.numel() for p in model.parameters())
                print(
                    f"\n[inference] loaded checkpoint {ckpt_path.name}  "
                    f"model={model_info}  params={n_params:,}",
                    flush=True,
                )

            # ── Inference loop over volumes ───────────────────────────────────
            resolutions: list[float] = []

            for vol_idx, (evn, odd, batch_params) in enumerate(val_loader):
                evn = evn.to(ctx.device)   # unrolled: (1,1,V,A,N) sinogram; else (1,1,D,H,W) volume
                odd = odd.to(ctx.device)

                if is_tomo:
                    physics.update(tomo_idx=batch_params["tomo_idx"])
                else:
                    physics.update_parameters(
                        tilt_min=batch_params["tilt_min"],
                        tilt_max=batch_params["tilt_max"],
                        vol_shape=batch_params["vol_shape"],
                    )
                    if rank == 0 and vol_idx == 0:
                        print(
                            f"[physics] vol={vol_idx}  "
                            f"tilt_min={float(batch_params['tilt_min']):.1f}°  "
                            f"tilt_max={float(batch_params['tilt_max']):.1f}°",
                            flush=True,
                        )

                f_evn_t, f_odd_t = preset["forward"](_InferenceModel(model), evn, odd, physics, train=False)
                with torch.no_grad():
                    r_evn, r_odd = preset["recon"](model, physics, f_evn_t, f_odd_t)
                recon_t = 0.5 * (r_evn + r_odd)

                if hasattr(ctx.device, "type") and ctx.device.type == "cuda":
                    torch.cuda.synchronize()

                # ── FSC ────────────────────────────────────────────────────────
                if gpu_fsc is None:
                    gpu_fsc = GpuFSC(device=f_evn_t.device)

                fsc_curve = gpu_fsc(r_evn, r_odd)
                px  = pixel_sizes[vol_idx] if vol_idx < len(pixel_sizes) else 1.0
                k, res, D = fsc_resolution(fsc_curve, r_evn.squeeze().shape,
                                           px, cfg.fsc_threshold)
                resolutions.append(res)

                tomo_name = val_ds.evn_paths[vol_idx].parent.name

                if rank == 0:
                    print(
                        f"[inference] vol{vol_idx:02d} ({tomo_name})  "
                        f"FSC@{cfg.fsc_threshold}={res:.1f} Å  (shell {k})",
                        flush=True,
                    )

                # ── numpy conversion ───────────────────────────────────────────
                # For unrolled, evn/odd are sinograms — recon_panels swaps in the
                # FBP init volumes instead (same convention as EIFullTrainer).
                evn_np, odd_np, evn_odd_labels = recon_panels(evn, odd, physics)
                # Canonical (Y, X, Z) from here on — this feeds both the figure and
                # _save_mrc, which assumes canonical and would otherwise write a
                # mis-oriented MRC for the unrolled preset.
                recon_np = to_canonical_np(recon_t.squeeze().cpu().numpy(), physics)

                # ── IsoNet / IceCream comparison volumes ───────────────────────
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
                    # _znorm(recon_np): save_slice_figure shares one vmin/vmax per row
                    # across all columns, and every other column is already z-scored
                    # (recon_panels / _load_mrc_vol). An un-normalised column would be
                    # rendered with the others' range and come out flat grey. Applied
                    # here, not to recon_np itself — that is also written to MRC below.
                    methods_cols   = [evn_np, odd_np, isonet_np, icecream_np, _znorm(recon_np)]
                    methods_labels = [*evn_odd_labels, "IsoNet", "IceCream", "ours"]
                    valid_pairs = [(v, lbl) for v, lbl in zip(methods_cols, methods_labels) if v is not None]
                    valid_cols, valid_labels = zip(*valid_pairs) if valid_pairs else ([], [])
                    save_slice_figure(
                        images_dir, epoch=0, vol_idx=vol_idx,
                        cols=list(valid_cols),
                        labels=list(valid_labels),
                        title=f"{tomo_name} | method comparison",
                        subdir=".",
                        fname=f"{tomo_name}_methods.png",
                    )

                    # ── FSC figure ─────────────────────────────────────────────
                    save_fsc_figure(
                        images_dir, epoch=0,
                        fname=f"{tomo_name}_fsc.png",
                        fsc_curve=fsc_curve, res_shell=k, res_angstrom=res,
                        title=f"{tomo_name} | FSC  {res:.1f} Å",
                        threshold=cfg.fsc_threshold,
                        vol_size=D, pixel_size=px,
                    )

                    # ── Optional: save recon MRC ────────────────────────────────
                    if cfg.save_recon_mrc:
                        recon_mrc_path = images_dir / f"{tomo_name}_recon.mrc"
                        _save_mrc(recon_mrc_path, recon_np)
                        print(f"  [recon mrc] saved {recon_mrc_path.name}", flush=True)

                all_rows.append({
                    "checkpoint":       ckpt_name,
                    "vol_idx":          vol_idx,
                    "tomo":             tomo_name,
                    "fsc_shell":        int(k),
                    "fsc_res_angstrom": float(res),
                    "pixel_size":       float(px),
                })

                if rank == 0:
                    append_fsc_row(output_dir / "metrics" / "fsc.csv",
                                   curve=fsc_curve if cfg.save_fsc_curves else None,
                                   mode="inference", regime="full", split="val",
                                   checkpoint=ckpt_name, vol_idx=vol_idx,
                                   tomo=tomo_name, pixel_size=float(px), n_ref=D,
                                   fsc_threshold=cfg.fsc_threshold,
                                   fsc_shell=int(k), fsc_res_angstrom=float(res))

            # ── Per-checkpoint summary ─────────────────────────────────────────
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
                print(
                    f"[inference] {ckpt_name}  n={len(resolutions)}  "
                    f"mean={mean_res:.1f} Å  median={median_res:.1f} Å  "
                    f"Q1={q1_res:.1f} Å  Q3={q3_res:.1f} Å  (lower=better)",
                    flush=True,
                )

            # Drop this checkpoint's model and hand its memory back to the
            # CUDA driver before building the next one — distribute()/astra
            # buffers are large at native resolution and PyTorch won't
            # release them on its own (same reasoning as the train/eval
            # empty_cache() in trainer.py).
            del model
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # ── Combined summary across all checkpoints ────────────────────────────
        if rank == 0 and all_rows:
            results_path = output_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(all_rows, f, indent=2)
            print(
                f"\n[inference] DONE  {len(cfg.checkpoint_paths)} checkpoint(s)  "
                f"{len(all_rows)} row(s) -> {results_path}",
                flush=True,
            )
