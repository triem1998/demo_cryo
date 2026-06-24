"""EI training entry-points: patch-based and full-volume variants."""
from __future__ import annotations

from pathlib import Path

import torch
from deepinv.distributed import DistributedContext, distribute

from .base_config import RunEIBaseConfig, _build_physics
from .dataset.dataset_full import EIFullDataConfig, build_ei_full_dataloaders
from .dataset.dataset_patch import EIPatchDataConfig, build_ei_patch_dataloaders
from .inference.infer_patch import run_post_training_inference
from .losses.losses import EqLoss, ObsLoss
from .losses.losses_custom import EqLoss as EqLossCustom, ObsLoss as ObsLossCustom
from .trainer import EIFullTrainer, EIPatchTrainer
from .transform import Rotate3D
from .utils.plot import plot_metrics
from .utils.utils import (
    _read_mrc_vol_size, _read_pixel_sizes,
    build_ei_model, dump_config_json, ensure_dir, seed_everything,
)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

class RunEIFullConfig(RunEIBaseConfig):
    # ── Data ────────────────────────────────────────────────────────────────
    output_dir: str = "./runs/demo_cryo_ei_full"
    target_shape: tuple[int, int, int] | None = None

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
    log_every_n_epochs: int = 1

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
    num_workers: int = 1
    prefetch_factor: int = 1
    normalize: bool = True

    # ── Training ────────────────────────────────────────────────────────────
    num_epochs: int = 100
    grad_accumulation_steps: int = 1
    log_every_n_epochs: int = 100

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

def _build_losses(cfg: RunEIBaseConfig, physics, transform) -> list:
    if str(cfg.loss_type) == "custom":
        return [
            ObsLossCustom(physics, weight=1.0),
            EqLossCustom(physics, transform, weight=float(cfg.eq_weight)),
        ]
    return [
        ObsLoss(physics, weight=1.0,
                use_fourier=False, view_as_real=True, no_window=False),
        EqLoss(physics, transform, weight=float(cfg.eq_weight),
               use_fourier=False, view_as_real=True, eq_use_direct=False, no_window=False),
    ]


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
    trainer._log_every_n_epochs = int(cfg.log_every_n_epochs)
    trainer._train_sampler      = train_sampler
    trainer._metrics_dir        = ensure_dir(output_dir / "metrics")
    trainer._images_dir         = ensure_dir(output_dir / images_subdir) if rank == 0 else None
    if train_images_subdir is not None:
        trainer._train_images_dir = ensure_dir(output_dir / train_images_subdir) if rank == 0 else None
    trainer._ckpt_dir           = ensure_dir(output_dir / "checkpoints") if rank == 0 else None
    trainer._grad_accum_steps   = max(1, int(cfg.grad_accumulation_steps))
    trainer.ckp_interval        = int(cfg.ckp_interval)
    if cfg.use_mixed_precision:
        trainer._enable_mixed_precision()
        if rank == 0:
            print("[ei] mixed precision enabled", flush=True)


# ---------------------------------------------------------------------------
# Entry-points
# ---------------------------------------------------------------------------

def run_full(cfg: RunEIFullConfig) -> None:
    seed_everything(int(cfg.seed))

    output_dir = ensure_dir(cfg.output_dir)
    dump_config_json(output_dir / "config.json", cfg.model_dump())

    data_cfg = EIFullDataConfig(
        input_dir=cfg.input_dir,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=int(cfg.prefetch_factor),
        persistent_workers=bool(cfg.persistent_workers),
        max_train_vols=cfg.max_train_vols,
        max_val_vols=int(cfg.max_val_vols),
        seed=int(cfg.seed),
        target_shape=cfg.target_shape,
        fallback_tilt_min=cfg.tilt_min,
        fallback_tilt_max=cfg.tilt_max,
    )

    with DistributedContext(seed=int(cfg.seed), seed_offset=False, cleanup=True) as ctx:
        rank = int(ctx.rank)

        data_bundle = build_ei_full_dataloaders(data_cfg)

        if cfg.target_shape is not None:
            vol_size = int(min(cfg.target_shape))
            print(f"[ei-full] target_shape={cfg.target_shape} → physics crop_size={vol_size}", flush=True)
        else:
            first_path = data_bundle.train_loader.dataset.evn_paths[0]
            vol_size = _read_mrc_vol_size(first_path)
            print(f"[ei-full] auto vol_size={vol_size}  (from {first_path.name})", flush=True)

        physics   = _build_physics(cfg, vol_size, ctx.device)
        transform = Rotate3D(n_trans=1)

        wrapper, model_info = build_ei_model(
            cfg.model_type, cfg.unet_dropout, cfg.drunet_sigma, ctx.device,
        )

        if cfg.pretrained_ckpt is not None:
            ckpt = torch.load(cfg.pretrained_ckpt, map_location=ctx.device, weights_only=True)
            state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
            if any(k.startswith("processor.") for k in state):
                state = {k.removeprefix("processor."): v for k, v in state.items()}
                if rank == 0:
                    print("[ei-full] stripped 'processor.' prefix from checkpoint keys", flush=True)
            wrapper.load_state_dict(state, strict=True)
            if rank == 0:
                print(f"[ei-full] loaded pretrained weights from {cfg.pretrained_ckpt}", flush=True)

        model = distribute(wrapper, ctx, type_object="denoiser",
                           patch_size=tuple(int(v) for v in cfg.patch_size),
                           overlap=tuple(int(v) for v in cfg.overlap),
                           tiling_dims=(-3, -2, -1),
                           max_batch_size=cfg.max_batch_size,
                           checkpoint_batches=cfg.checkpoint_batches)

        if rank == 0:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[ei-full] model={model_info}  params={n_params:,}", flush=True)
            print(f"[ei-full] vol_size={vol_size}  patch_size={cfg.patch_size}  "
                  f"overlap={cfg.overlap}  max_batch_size={cfg.max_batch_size}  "
                  f"checkpoint_batches={cfg.checkpoint_batches}", flush=True)

        losses    = _build_losses(cfg, physics, transform)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))

        trainer = EIFullTrainer(
            model=model, physics=physics, optimizer=optimizer,
            train_dataloader=data_bundle.train_loader,
            eval_dataloader=data_bundle.val_loader if len(data_bundle.val_loader.dataset) > 0 else None,
            epochs=int(cfg.num_epochs), losses=losses, metrics=[],
            online_measurements=False, device=ctx.device, save_path=None,
            ckp_interval=int(cfg.ckp_interval), eval_interval=int(cfg.eval_interval),
            grad_clip=cfg.grad_clip, check_grad=cfg.grad_clip is not None,
            plot_images=False, verbose=rank == 0, show_progress_bar=rank == 0,
            log_train_batch=False, optimizer_step_multi_dataset=False,
        )
        _configure_trainer(trainer, cfg, output_dir, rank,
                           images_subdir="val_images", train_images_subdir="train_images")

        val_ds          = data_bundle.val_loader.dataset
        val_pixel_sizes = _read_pixel_sizes(val_ds.evn_paths, fallback=cfg.pixel_size_angstrom)
        n_paired_val    = sum(1 for p in val_ds.odd_paths if p is not None)
        if rank == 0:
            print(f"[fsc-eval] val pixel sizes (Å/px): {[f'{v:.2f}' for v in val_pixel_sizes]}", flush=True)

        if cfg.eval_fsc and n_paired_val > 0:
            trainer._val_pixel_sizes = val_pixel_sizes
            trainer._fsc_threshold   = float(cfg.fsc_threshold)
            if rank == 0:
                print(f"[fsc-eval] enabled for {n_paired_val} paired val volumes  thr={cfg.fsc_threshold}", flush=True)
        else:
            trainer._val_pixel_sizes = []
            trainer._fsc_threshold   = float(cfg.fsc_threshold)
            if rank == 0:
                msg = "disabled (eval_fsc=False)" if not cfg.eval_fsc else "disabled — no paired ODD val volumes"
                print(f"[fsc-eval] {msg}", flush=True)

        trainer.train()

        if rank == 0 and trainer._ckpt_dir is not None:
            ckpt_path = Path(trainer._ckpt_dir) / "ckp_final.pth"
            torch.save({
                "epoch": cfg.num_epochs,
                "model_state_dict": trainer.model.processor.state_dict(),
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
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=int(cfg.prefetch_factor),
        persistent_workers=bool(cfg.persistent_workers),
        max_train_vols=cfg.max_train_vols,
        max_val_vols=int(cfg.max_val_vols),
        seed=int(cfg.seed),
        normalize=bool(cfg.normalize),
        fallback_tilt_min=cfg.tilt_min,
        fallback_tilt_max=cfg.tilt_max,
    )

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

        physics   = _build_physics(cfg, int(cfg.crop_size), ctx.device)
        transform = Rotate3D(n_trans=1)

        model, model_info = build_ei_model(
            cfg.model_type, cfg.unet_dropout, cfg.drunet_sigma, ctx.device,
        )

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

        losses    = _build_losses(cfg, physics, transform)
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

        trainer.train()

        raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        if rank == 0 and trainer._ckpt_dir is not None:
            ckpt_path = Path(trainer._ckpt_dir) / "ckp_final.pth"
            torch.save({
                "epoch": cfg.num_epochs,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"[ckpt] saved final {ckpt_path}", flush=True)
            plot_metrics(output_dir, save=output_dir / "metrics" / "summary.png")

        # ── Post-training sliding-window inference ───────────────────────────
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
            )
