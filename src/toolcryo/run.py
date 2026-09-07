"""EI training entry-points: patch-based and full-volume variants."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Literal

import torch
from deepinv.distributed import DistributedContext

from .base_config import RunEIBaseConfig, amp_dtype_from_str
from .dataset.dataset_full import EIFullDataConfig, build_ei_full_dataloaders, _make_full_loader
from .dataset.dataset_patch import (
    EIPatchDataConfig, build_ei_patch_dataloaders, extract_patches_at_positions,
)
from .inference.infer_patch import run_post_training_inference
from .losses.losses_equivariant_wedge import _symmetrize_and_binarize
from .models import build_distributed_denoiser
from .registry import get_preset
from .trainer import EIFullTrainer, EIPatchTrainer
from .transform import Rotate3D
from .utils.plot import plot_metrics
from .utils.utils import (
    _read_mrc_vol_size, _read_pixel_sizes,
    dump_config_json, ensure_dir, seed_everything,
)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

class RunEIFullConfig(RunEIBaseConfig):
    # ── Data ────────────────────────────────────────────────────────────────
    output_dir: str = "./runs/demo_cryo_ei_full"
    target_shape: tuple[int, int, int] | None = None
    # Random-crop side used for training; null = min(dim) after target_shape
    # (today's cube size, but re-cropped at a random origin every epoch
    # instead of a fixed centre crop). Evaluation always sees the whole
    # volume regardless of this setting.
    crop_size: int | None = None
    normalize_crops: bool = False

    # ── DataLoader ──────────────────────────────────────────────────────────
    batch_size: int = 1
    num_workers: int = 1
    prefetch_factor: int = 1

    # ── Distributed model tiling ─────────────────────────────────────────────
    patch_size: tuple[int, int, int] = (64, 64, 64)
    overlap: tuple[int, int, int] = (8, 8, 8)
    max_batch_size: int | None = 1
    checkpoint_batches: str | int | None = "auto"

    # ── Training ────────────────────────────────────────────────────────────
    num_epochs: int = 10
    grad_accumulation_steps: int = 4

    # ── Unrolled preset only ───────────────────────────────────────────────
    n_iter: int = 4
    init_stepsize: float = 0.9
    # Trains `stepsize` (+ `g_param` for drunet only) jointly with the
    # denoiser (models.py::build_unrolled_model). No beta/relaxation.
    train_algo_params: bool = True
    # LR for the stepsize param group when train_algo_params — None falls
    # back to `learning_rate` (used for the denoiser + g_param).
    stepsize_learning_rate: float | None = None
    # Angle-sharded physics: null = off (default), "auto" = one op/rank, int = that many.
    num_operators: int | Literal["auto"] | None = None

    # ── Evaluation ──────────────────────────────────────────────────────────
    eval_fsc: bool = True

    @classmethod
    def from_yaml(cls, conf: dict) -> "RunEIFullConfig":
        return cls.model_validate(cls._flat_from_yaml(conf, "demo-cryo-ei-full"))


class RunEIPatchConfig(RunEIBaseConfig):
    # ── Data ────────────────────────────────────────────────────────────────
    output_dir: str = "./runs/demo_cryo_ei_patch"

    # ── Patch ───────────────────────────────────────────────────────────────
    crop_size: int = 72
    n_crops_per_vol: int = 10
    batch_size: int = 4
    # False = icecream: one volume per optimizer step, using its exact wedge.
    # True = a batch may mix volumes and shares the intersected wedge.
    mix_volumes: bool = True
    num_workers: int = 1
    prefetch_factor: int = 1
    normalize: bool = True
    normalize_crops: bool = False

    # Crop origins [d, h, w] to evaluate every log interval on the val (fallback
    # train) volumes; empty = no probe. Saved under runs/.../patch_probe/.
    patch_positions: list[list[int]] = []

    # ── Training ────────────────────────────────────────────────────────────
    num_epochs: int = 100
    grad_accumulation_steps: int = 1

    # ── Inference (post-training sliding-window) ─────────────────────────────
    infer_stride: int = 36
    infer_batch_size: int = 0
    infer_downsample: int = 1
    infer_train: bool = True
    infer_val: bool = True
    save_mrc: bool = False

    # ── Evaluation ──────────────────────────────────────────────────────────
    eval_fsc: bool = False

    @classmethod
    def from_yaml(cls, conf: dict) -> "RunEIPatchConfig":
        return cls.model_validate(cls._flat_from_yaml(conf, "demo-cryo-ei-patch"))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_plateau_scheduler(cfg: RunEIBaseConfig, optimizer):
    """ReduceLROnPlateau on TotalLoss, or None when disabled.

    Stepped manually from BaseTrainer.log_metrics_mlops, not via dinv.Trainer's
    scheduler= (its bare .step() call is incompatible with ReduceLROnPlateau).
    """
    if not cfg.use_lr_scheduler:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=float(cfg.lr_scheduler_factor),
        patience=int(cfg.lr_scheduler_patience),
    )


def _find_psnr_ref(tomo_dir, globs, rank: int):
    """First file in ``tomo_dir`` matching ``globs`` in order, or None.

    The globs are ordered ground-truth-first, so a dataset that ships one is
    scored against it and everything else falls back to icecream.
    """
    for pattern in globs or []:
        hit = next(iter(sorted(tomo_dir.glob(pattern))), None)
        if hit is not None:
            if rank == 0:
                print(f"[psnr] {tomo_dir.name}: reference {hit.name}", flush=True)
            return hit
    if rank == 0 and globs:
        print(f"[psnr] {tomo_dir.name}: no reference matched {globs} — PSNR off", flush=True)
    return None


def _resume_training_state(cfg, trainer, optimizer, ckpt_path, permute: bool, rank: int) -> None:
    """Restore optimizer + scheduler + global epoch from a same-run checkpoint.

    The weights themselves are loaded separately (build_distributed_denoiser /
    the inline load in run_patch); this is everything *else* a resume needs.

    """
    if not cfg.resume_optimizer:
        return
    if ckpt_path is None:
        raise ValueError("resume_optimizer=True but pretrained_ckpt is not set.")
    if permute:
        raise ValueError(
            "resume_optimizer=True is incompatible with permute_native_to_astra=True: "
            "the permutation reorders conv-kernel axes and Adam's moments are "
            "per-element, so they would land on the wrong axes. Resume from an "
            "astra-order tomo_ei checkpoint (permute_native_to_astra: false) instead."
        )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    # periodic checkpoints (trainer.log_metrics_mlops) use "optimizer";
    # ckp_final.pth (below) uses "optimizer_state_dict".
    opt_state = ckpt.get("optimizer") or ckpt.get("optimizer_state_dict")
    if opt_state is None:
        raise ValueError(f"{ckpt_path} carries no optimizer state — cannot resume from it.")
    optimizer.load_state_dict(opt_state)
    # Absent from checkpoints written before scheduler state was saved; those
    # resume with a fresh plateau baseline, which is recoverable (a stale LR is not).
    if trainer._plateau_scheduler is not None and ckpt.get("scheduler") is not None:
        trainer._plateau_scheduler.load_state_dict(ckpt["scheduler"])
        sched_msg = ""
    else:
        sched_msg = "  (no scheduler state — plateau baseline restarts)"
    trainer._resume_epoch = ckpt.get("epoch")
    if rank == 0:
        print(f"[resume] optimizer state from {ckpt_path.name}  epoch={ckpt.get('epoch')}  "
              f"lr={optimizer.param_groups[0]['lr']:.3g}{sched_msg}", flush=True)


def _configure_trainer(
    trainer,
    cfg: RunEIBaseConfig,
    output_dir: Path,
    rank: int,
    images_subdir: str,
    train_images_subdir: str | None = None,
    train_sampler=None,
) -> None:
    trainer._init_trainer_state()
    trainer._is_rank0           = (rank == 0)
    trainer._log_every_n_epochs = int(cfg.eval_interval)
    trainer._train_sampler      = train_sampler
    trainer._metrics_dir        = ensure_dir(output_dir / "metrics")
    trainer._images_dir         = ensure_dir(output_dir / images_subdir) if rank == 0 else None
    if train_images_subdir is not None:
        trainer._train_images_dir = ensure_dir(output_dir / train_images_subdir) if rank == 0 else None
    trainer._ckpt_dir           = ensure_dir(output_dir / "checkpoints") if rank == 0 else None
    trainer._grad_accum_steps   = max(1, int(cfg.grad_accumulation_steps))
    trainer.ckp_interval        = int(cfg.ckp_interval)
    # "off" leaves the trainer untouched: _autocast/_amp_dtype/_scaler all stay
    # None, which every AMP site below treats as "do nothing".
    if cfg.mixed_precision != "off":
        trainer._enable_mixed_precision(dtype=cfg.mixed_precision)
        if rank == 0:
            print(f"[ei] mixed precision enabled ({cfg.mixed_precision})", flush=True)
    elif rank == 0:
        print("[ei] mixed precision off (fp32)", flush=True)


# ---------------------------------------------------------------------------
# Entry-points
# ---------------------------------------------------------------------------

def run_full(cfg: RunEIFullConfig) -> None:
    seed_everything(int(cfg.seed))

    output_dir = ensure_dir(cfg.output_dir)
    dump_config_json(output_dir / "config.json", cfg.model_dump())

    # `preset` selects the method; data_source follows automatically below.
    is_tomo     = cfg.preset in ("unrolled", "tomo_ei")   # TomographyEM physics
    is_unrolled = cfg.preset == "unrolled"                # PGD-unfold model

    data_cfg = EIFullDataConfig(
        input_dir=cfg.input_dir,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=int(cfg.prefetch_factor),
        persistent_workers=bool(cfg.persistent_workers),
        max_train_vols=cfg.max_train_vols,
        max_val_vols=int(cfg.max_val_vols),
        seed=int(cfg.seed),
        train_names=cfg.train_names,
        val_names=cfg.val_names,
        target_shape=cfg.target_shape,
        fallback_tilt_min=cfg.tilt_min,
        fallback_tilt_max=cfg.tilt_max,
        data_source="measurement" if is_tomo else "fbp",
        crop_size=cfg.crop_size,
        normalize_crops=bool(cfg.normalize_crops),
    )

    preset = get_preset(cfg.preset)

    with DistributedContext(seed=int(cfg.seed), seed_offset=False, cleanup=True) as ctx:
        rank = int(ctx.rank)

        data_bundle = build_ei_full_dataloaders(data_cfg)
        train_ds = data_bundle.train_loader.dataset
        val_ds   = data_bundle.val_loader.dataset

        # Shared by both branches below and by _resume_training_state.
        ckpt_path = Path(cfg.pretrained_ckpt) if cfg.pretrained_ckpt else None
        if is_tomo and not is_unrolled:
            permute = cfg.permute_native_to_astra if cfg.permute_native_to_astra is not None else True
        else:
            permute = False   # missingwedge_ei stays native-order throughout

        if is_tomo:
            physics = preset["physics"](
                cfg, train_ds.evn_paths + val_ds.evn_paths, train_ds.odd_paths + val_ds.odd_paths, ctx.device, ctx)

            if is_unrolled:
                transform = None  # no equivariance term in v1
                # tiles the denoiser across ranks; physics stays local per rank
                model, model_info = preset["model"](cfg, physics, ctx)
            else:
                # tomo_ei: plain denoiser on each half's FBP volume. Astra's
                # volume isn't a cube, so Rotate3D must only use shape-preserving rotations.
                transform = Rotate3D(n_trans=1, volume_shape=physics.physics_evn.volume_shape)
                model, model_info = build_distributed_denoiser(
                    cfg, ctx, rank, ckpt_path, permute_native_to_astra=permute, log_prefix="ei-full")
        else:
            # crop_size sets the training crop + physics shape; eval always
            # uses the whole volume via MissingWedge.update_parameters(vol_shape=...).
            if cfg.crop_size is not None:
                vol_size = int(cfg.crop_size)
                print(f"[ei-full] crop_size={vol_size}  (config)", flush=True)
            elif cfg.target_shape is not None:
                vol_size = int(min(cfg.target_shape))
                print(f"[ei-full] target_shape={cfg.target_shape} → crop_size={vol_size}", flush=True)
            else:
                first_path = train_ds.evn_paths[0]
                vol_size = _read_mrc_vol_size(first_path)
                print(f"[ei-full] auto crop_size={vol_size}  (from {first_path.name})", flush=True)
            train_ds.crop_size = vol_size

            physics   = preset["physics"](cfg, vol_size, ctx.device)
            transform = Rotate3D(n_trans=1)
            model, model_info = build_distributed_denoiser(
                cfg, ctx, rank, ckpt_path, permute_native_to_astra=permute, log_prefix="ei-full")

            if rank == 0:
                print(f"[ei-full] vol_size={vol_size}  patch_size={cfg.patch_size}  "
                      f"overlap={cfg.overlap}  max_batch_size={cfg.max_batch_size}  "
                      f"checkpoint_batches={cfg.checkpoint_batches}", flush=True)

        if rank == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[ei-full] model={model_info}  params={n_params:,}", flush=True)

        losses = preset["losses"](cfg, physics, transform)
        if is_unrolled and cfg.train_algo_params:
            # stepsize gets its own LR (cfg.stepsize_learning_rate, falls back to learning_rate)
            stepsize_params = list(model.params_algo["stepsize"])
            stepsize_ids = {id(p) for p in stepsize_params}
            other_params = [p for p in model.parameters() if id(p) not in stepsize_ids]
            stepsize_lr = (float(cfg.stepsize_learning_rate)
                           if cfg.stepsize_learning_rate is not None else float(cfg.learning_rate))
            optimizer = torch.optim.Adam([
                {"params": other_params, "lr": float(cfg.learning_rate)},
                {"params": stepsize_params, "lr": stepsize_lr},
            ])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))

        # FSC eval targets val volumes; if none are paired (val empty/unpaired),
        # fall back to running it on the train volumes instead.
        n_paired_val = sum(1 for p in val_ds.odd_paths if p is not None)
        if n_paired_val > 0:
            fsc_loader, fsc_ds, fsc_label = data_bundle.val_loader, val_ds, "val"
        else:
            # Same volumes as train_ds, but FSC must see whole volumes, not
            # the training crop — a shallow copy shares the (read-only) path
            # lists and only needs crop_size overridden.
            fsc_ds = copy.copy(train_ds)
            fsc_ds.crop_size = None
            fsc_loader = _make_full_loader(fsc_ds, shuffle=False, cfg=data_cfg)
            fsc_label  = "train"
        n_paired_fsc    = sum(1 for p in fsc_ds.odd_paths if p is not None)
        eval_dataloader = fsc_loader if n_paired_fsc > 0 else None

        trainer = EIFullTrainer(
            model=model, physics=physics, optimizer=optimizer,
            train_dataloader=data_bundle.train_loader,
            eval_dataloader=eval_dataloader,
            epochs=int(cfg.num_epochs), losses=losses, metrics=[],
            online_measurements=False, device=ctx.device, save_path=None,
            ckp_interval=int(cfg.ckp_interval), eval_interval=int(cfg.eval_interval),
            grad_clip=cfg.grad_clip, check_grad=cfg.grad_clip is not None,
            plot_images=False, verbose=rank == 0, show_progress_bar=rank == 0,
            log_train_batch=False, optimizer_step_multi_dataset=False,
        )
        _configure_trainer(trainer, cfg, output_dir, rank,
                           images_subdir=f"{fsc_label}_fsc_images" if fsc_label == "train" else "val_images",
                           train_images_subdir="train_images")
        trainer._plateau_scheduler = _build_plateau_scheduler(cfg, optimizer)
        _resume_training_state(cfg, trainer, optimizer, ckpt_path, permute, rank)
        trainer._forward_strategy = preset["forward"]
        trainer._post_optimizer_step = lambda: preset["post_optimizer_step"](model)
        trainer._recon_strategy      = preset["recon"]
        trainer._fsc_tomo_names  = [p.parent.name for p in fsc_ds.evn_paths]
        trainer._psnr_refs = [_find_psnr_ref(p.parent, cfg.psnr_ref_globs, rank)
                              for p in fsc_ds.evn_paths]
        trainer._fsc_split       = fsc_label
        trainer._save_fsc_curves = bool(cfg.save_fsc_curves)

        fsc_pixel_sizes = _read_pixel_sizes(fsc_ds.evn_paths, fallback=cfg.pixel_size_angstrom)
        if rank == 0:
            print(f"[fsc-eval] {fsc_label} pixel sizes (Å/px): {[f'{v:.2f}' for v in fsc_pixel_sizes]}", flush=True)

        if cfg.eval_fsc and n_paired_fsc > 0:
            trainer._val_pixel_sizes = fsc_pixel_sizes
            trainer._fsc_threshold   = float(cfg.fsc_threshold)
            if rank == 0:
                print(f"[fsc-eval] enabled for {n_paired_fsc} paired {fsc_label} volumes  thr={cfg.fsc_threshold}", flush=True)
        else:
            trainer._val_pixel_sizes = []
            trainer._fsc_threshold   = float(cfg.fsc_threshold)
            if rank == 0:
                msg = "disabled (eval_fsc=False)" if not cfg.eval_fsc else f"disabled — no paired ODD {fsc_label} volumes"
                print(f"[fsc-eval] {msg}", flush=True)

        trainer.train()

        if rank == 0 and trainer._ckpt_dir is not None:
            ckpt_path = Path(trainer._ckpt_dir) / "ckp_final.pth"
            # .processor exists for a tiled bare denoiser (missingwedge_ei/tomo_ei);
            # unrolled keeps trainer.model as the PGD object itself.
            raw_model = trainer.model.processor if hasattr(trainer.model, "processor") else trainer.model
            torch.save({
                "epoch": int(cfg.num_epochs) - 1,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"[ckpt] saved final {ckpt_path}", flush=True)
            plot_metrics(output_dir, save=output_dir / "metrics" / "summary.png")


def run_patch(cfg: RunEIPatchConfig) -> None:
    seed_everything(int(cfg.seed))

    output_dir = ensure_dir(cfg.output_dir)
    dump_config_json(output_dir / "config.json", cfg.model_dump())

    data_cfg = EIPatchDataConfig(
        input_dir=cfg.input_dir,
        crop_size=int(cfg.crop_size),
        n_crops_per_vol=int(cfg.n_crops_per_vol),
        batch_size=int(cfg.batch_size),
        mix_volumes=bool(cfg.mix_volumes),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=int(cfg.prefetch_factor),
        persistent_workers=bool(cfg.persistent_workers),
        max_train_vols=cfg.max_train_vols,
        max_val_vols=int(cfg.max_val_vols),
        seed=int(cfg.seed),
        train_names=cfg.train_names,
        val_names=cfg.val_names,
        normalize=bool(cfg.normalize),
        normalize_crops=bool(cfg.normalize_crops),
        fallback_tilt_min=cfg.tilt_min,
        fallback_tilt_max=cfg.tilt_max,
    )

    preset = get_preset(cfg.preset)

    with DistributedContext(seed=int(cfg.seed), seed_offset=False, cleanup=True) as ctx:
        rank = int(ctx.rank)

        data_bundle = build_ei_patch_dataloaders(data_cfg, rank=rank, world_size=ctx.world_size)

        if rank == 0:
            train_ds = data_bundle.train_loader.dataset
            val_ds   = data_bundle.val_loader.dataset
            print("[ei-patch] Train volumes:")
            for p in train_ds.evn_paths:
                print(f"  {p.parent.name} / {p.name}")
            print("[ei-patch] Val volumes:")
            for p in val_ds.evn_paths:
                print(f"  {p.parent.name} / {p.name}")

        physics   = preset["physics"](cfg, int(cfg.crop_size), ctx.device)
        transform = Rotate3D(n_trans=1)

        model, model_info = preset["model"](
            cfg.model_type, cfg.unet_dropout, cfg.drunet_sigma, ctx.device,
        )

        if cfg.pretrained_ckpt is not None:
            ckpt = torch.load(cfg.pretrained_ckpt, map_location=ctx.device, weights_only=True)
            state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}   # torch.compile wrapper
            if any(k.startswith("module.") for k in state):
                state = {k.removeprefix("module."): v for k, v in state.items()}
                if rank == 0:
                    print("[ei-patch] stripped 'module.' prefix from checkpoint keys", flush=True)
            if any(k.startswith("processor.") for k in state):
                state = {k.removeprefix("processor."): v for k, v in state.items()}
                if rank == 0:
                    print("[ei-patch] stripped 'processor.' prefix from checkpoint keys", flush=True)
            model.load_state_dict(state, strict=True)
            if rank == 0:
                print(f"[ei-patch] loaded pretrained weights from {cfg.pretrained_ckpt}", flush=True)

        if cfg.compile:
            model = torch.compile(model)

        if ctx.world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ctx.local_rank])
            if rank == 0:
                print(f"[ei-patch] DDP enabled: {ctx.world_size} GPUs  "
                      f"effective_batch={cfg.batch_size * ctx.world_size}", flush=True)

        if rank == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[ei-patch] model={model_info}  params={n_params:,}", flush=True)
            print(f"[ei-patch] crop_size={cfg.crop_size}  batch_size={cfg.batch_size}  "
                  f"wedge_double_size={cfg.wedge_double_size}  eq_weight={cfg.eq_weight}", flush=True)

        losses    = preset["losses"](cfg, physics, transform)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))

        trainer = EIPatchTrainer(
            model=model, physics=physics, optimizer=optimizer,
            train_dataloader=data_bundle.train_loader,
            eval_dataloader=None if cfg.max_val_vols == 0 else data_bundle.val_loader,
            epochs=int(cfg.num_epochs), losses=losses, metrics=[],
            online_measurements=False, device=ctx.device, save_path=None,
            ckp_interval=int(cfg.ckp_interval), eval_interval=int(cfg.eval_interval),
            grad_clip=cfg.grad_clip, check_grad=cfg.grad_clip is not None,
            plot_images=False, verbose=False, show_progress_bar=False,
            log_train_batch=False, optimizer_step_multi_dataset=False,
            freq_update_progress_bar=100,
        )
        _configure_trainer(trainer, cfg, output_dir, rank,
                           images_subdir="train_images",
                           train_sampler=data_bundle.train_sampler)
        trainer._plateau_scheduler = _build_plateau_scheduler(cfg, optimizer)
        # run_patch never permutes — it trains and resumes in native axis order.
        _resume_training_state(
            cfg, trainer, optimizer,
            Path(cfg.pretrained_ckpt) if cfg.pretrained_ckpt else None, False, rank)

        # ── Patch-position probe: pre-extract fixed crops once (rank 0) ──────
        # Evaluated on val volumes, falling back to train when val is empty.
        if cfg.patch_positions and rank == 0:
            positions = [tuple(int(v) for v in p) for p in cfg.patch_positions]
            probe_ds = data_bundle.val_loader.dataset
            if len(probe_ds.evn_paths) == 0:
                probe_ds = data_bundle.train_loader.dataset
            probes = []
            for ep, op in zip(probe_ds.evn_paths, probe_ds.odd_paths):
                evn_crops, odd_crops, used = extract_patches_at_positions(
                    ep, op, positions, int(cfg.crop_size), bool(cfg.normalize),
                )
                probes.append((ep.parent.name, evn_crops, odd_crops, used))
            if probes:
                trainer._patch_probes = probes
                trainer._patch_probe_wedge = _symmetrize_and_binarize(
                    physics.mask[:-1, :-1, :-1]
                ).cpu()
                trainer._patch_probe_dir = ensure_dir(output_dir / "patch_probe")
                print(f"[ei-patch] patch probe: {len(probes)} tomo(s) × "
                      f"{len(positions)} position(s)", flush=True)

        trainer.train()

        raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        if rank == 0 and trainer._ckpt_dir is not None:
            ckpt_path = Path(trainer._ckpt_dir) / "ckp_final.pth"
            torch.save({
                # last *completed* epoch index, matching the periodic checkpoints
                # written by trainer.log_metrics_mlops — _resume_training_state
                # reads this and continues at epoch+1.
                "epoch": int(cfg.num_epochs) - 1,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"[ckpt] saved final {ckpt_path}", flush=True)
            plot_metrics(output_dir, save=output_dir / "metrics" / "summary.png")

    # ── Post-training sliding-window inference ───────────────────────────────
    infer_datasets = []
    train_ds = data_bundle.train_loader.dataset
    val_ds   = data_bundle.val_loader.dataset
    if cfg.infer_train and len(train_ds.evn_vols) > 0:
        infer_datasets.append(("train", train_ds))
    if cfg.infer_val and len(val_ds.evn_vols) > 0:
        infer_datasets.append(("val", val_ds))

    if infer_datasets:
        infer_bs = int(cfg.infer_batch_size) if cfg.infer_batch_size > 0 else int(cfg.batch_size)
        run_post_training_inference(
            datasets=infer_datasets,
            raw_model=raw_model,
            physics=physics,
            ctx=ctx,
            output_dir=output_dir,
            crop_size=int(cfg.crop_size),
            stride=max(1, int(cfg.infer_stride)),
            infer_batch_size=infer_bs,
            infer_downsample=max(1, int(cfg.infer_downsample)),
            tilt_min=float(cfg.tilt_min),
            tilt_max=float(cfg.tilt_max),
            use_spherical_support=bool(cfg.use_spherical_support),
            wedge_double_size=bool(cfg.wedge_double_size),
            wedge_low_support=float(cfg.wedge_low_support),
            ref_wedge_support=float(cfg.ref_wedge_support),
            fsc_threshold=float(cfg.fsc_threshold),
            pixel_size_angstrom=cfg.pixel_size_angstrom,
            save_mrc=bool(cfg.save_mrc),
            save_fsc_curves=bool(cfg.save_fsc_curves),
            # Infer in the dtype the model was trained in — and in fp32 when
            # the run is "off", rather than autocasting regardless as before.
            amp_dtype=amp_dtype_from_str(cfg.mixed_precision),
        )
