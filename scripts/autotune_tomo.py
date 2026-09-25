"""Pick the tiling and GPU split of the tomo_ei / unrolled presets with deepinv's AutoTuner.

Needs deepinv on the ``autotune`` branch. Everything runs on one GPU: one training
step with the denoiser replaced by the identity (per probed group size), then one
tile of each candidate size through the real denoiser. Probe on the target GPU
type. ``PYTORCH_CUDA_ALLOC_CONF`` defaults to the job's (slurm.setup)
``expandable_segments:True``.

    python scripts/autotune_tomo.py --config configs/conf_full_eq_tomo.yml \
        --preset tomo_ei unrolled --num-gpus 1 2 4 8

Writes ``<out>/autotune_<preset>.json``.
"""
from __future__ import annotations

import os

# before torch: the allocator reads it once, at CUDA init
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from deepinv.distributed import AutoTuner, DistributedContext
from deepinv.models.base import Denoiser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolcryo.dataset.dataset_full import EIFullDataConfig, build_ei_full_dataloaders  # noqa: E402
from toolcryo.models import build_distributed_denoiser, build_unrolled_model  # noqa: E402
from toolcryo.physics import build_tomography_physics  # noqa: E402
from toolcryo.registry import get_preset  # noqa: E402
from toolcryo.run import RunEIFullConfig  # noqa: E402
from toolcryo.trainer import EIFullTrainer  # noqa: E402
from toolcryo.transform import Rotate3D  # noqa: E402


class AutocastDenoiser(Denoiser):
    """Runs the denoiser under the job's autocast, so the tile probe sees bf16/fp16 activations."""

    def __init__(self, denoiser: torch.nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.denoiser, self.dtype = denoiser, dtype

    def forward(self, x, *args, **kwargs):
        with torch.autocast("cuda", dtype=self.dtype):
            return self.denoiser(x, *args, **kwargs)


def load_cfg(path: Path, preset: str, target_shape) -> RunEIFullConfig:
    conf = yaml.safe_load(path.read_text())
    if conf.get("method") != "equivariant_full":
        raise ValueError(f"{path}: method must be equivariant_full, got {conf.get('method')!r}")
    conf["general"]["preset"] = preset
    if target_shape is not None:
        conf["general"]["target_shape"] = list(target_shape)
    cfg = RunEIFullConfig.from_yaml(conf)
    # weights do not change memory or time; compile would hide the denoiser from the probe
    return cfg.model_copy(update={"pretrained_ckpt": None, "compile": False})


def load_batch(cfg: RunEIFullConfig, device):
    """First training tomogram, collated to batch 1 as the dataloader would."""
    data_cfg = EIFullDataConfig(
        input_dir=cfg.input_dir, num_workers=0, pin_memory=False, prefetch_factor=1,
        persistent_workers=False, max_train_vols=cfg.max_train_vols,
        max_val_vols=int(cfg.max_val_vols), seed=int(cfg.seed),
        train_names=cfg.train_names, val_names=cfg.val_names,
        target_shape=cfg.target_shape, fallback_tilt_min=cfg.tilt_min,
        fallback_tilt_max=cfg.tilt_max, data_source="measurement",
    )
    bundle = build_ei_full_dataloaders(data_cfg, ctx=DistributedContext())
    train_ds, val_ds = bundle.train_loader.dataset, bundle.val_loader.dataset
    evn, odd, params = train_ds[0]
    params = {k: v[None].to(device) for k, v in params.items()}
    return (evn[None].to(device), odd[None].to(device), params,
            train_ds.evn_paths + val_ds.evn_paths, train_ds.odd_paths + val_ds.odd_paths,
            train_ds.evn_paths[0].parent.name)


def build_model(cfg: RunEIFullConfig, make_physics, device):
    """The job's model, untiled: the tuner chooses the tiling."""
    ctx = DistributedContext()
    ctx.device = device
    if cfg.preset == "unrolled":
        model, _ = build_unrolled_model(cfg, make_physics(ctx), ctx)
        model.prior[0].denoiser = model.prior[0].denoiser.processor
    else:
        model, _ = build_distributed_denoiser(cfg, ctx, 0, None)
        model = model.processor
    if cfg.mixed_precision != "off":
        dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
        if cfg.preset == "unrolled":
            model.prior[0].denoiser = AutocastDenoiser(model.prior[0].denoiser, dtype)
        else:
            model = AutocastDenoiser(model, dtype)
    return model.to(device)


def build_optimizer(cfg: RunEIFullConfig, model):
    """Same parameter groups as run_full, so the tuner counts the same Adam state."""
    if cfg.preset == "unrolled" and cfg.train_algo_params:
        stepsize = list(model.params_algo["stepsize"])
        ids = {id(p) for p in stepsize}
        lr_s = cfg.stepsize_learning_rate if cfg.stepsize_learning_rate is not None else cfg.learning_rate
        return torch.optim.Adam([
            {"params": [p for p in model.parameters() if id(p) not in ids], "lr": float(cfg.learning_rate)},
            {"params": stepsize, "lr": float(lr_s)},
        ])
    return torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))


def build_step(cfg: RunEIFullConfig, model, optimizer, x, y, volume_shape, device):
    """``step(model, physics)``: one training step of EIFullTrainer, as the job runs it."""
    preset = get_preset(cfg.preset)
    transform = Rotate3D(n_trans=1, volume_shape=volume_shape)
    trainer = EIFullTrainer(
        model=model, physics=None, optimizer=optimizer, train_dataloader=[], epochs=1,
        losses=preset["losses"](cfg, None, transform), metrics=[],
        online_measurements=False, device=device, save_path=None,
        grad_clip=cfg.grad_clip, check_grad=cfg.grad_clip is not None,
        plot_images=False, verbose=False, show_progress_bar=False,
        log_train_batch=False, optimizer_step_multi_dataset=False,
    )
    trainer._init_trainer_state()
    trainer.setup_train(train=True)
    if cfg.mixed_precision != "off":
        trainer._enable_mixed_precision(dtype=cfg.mixed_precision)
    trainer._forward_strategy = preset["forward"]
    trainer._post_optimizer_step = lambda: preset["post_optimizer_step"](trainer.model)
    # figures would add no-grad denoiser calls the tuner counts as training calls
    trainer._save_train_figures = lambda *a, **k: None
    # a new epoch enters PerfProbe, which resets the CUDA peak the tuner reads
    trainer._current_train_epoch = 0

    def step(model, physics):
        trainer.model = model
        trainer.compute_loss(physics, x, y, train=True, epoch=0, step=True)

    return step, trainer


def as_dict(c, n_alternatives: int = 5) -> dict | None:
    if c is None:
        return None
    d = {k: v for k, v in dataclasses.asdict(c).items() if k != "alternatives"}
    d.update(gpus_used=c.gpus_used, images_per_s=c.images_per_s,
             alternatives=[as_dict(a, 0) for a in c.alternatives[:n_alternatives]])
    return d


def yaml_block(c) -> dict | None:
    """The ``distributed:`` keys for the first config the YAML can express.

    build_distributed_denoiser tiles all three axes, so only configs cutting all
    three map onto the YAML; the others need tiling_dims plumbed through.
    """
    for cand in [c, *c.alternatives] if c is not None else []:
        if len(cand.tiling_dims) == 3:
            return dict(inner_world_size=cand.inner_world_size, patch_size=list(cand.patch_size),
                        overlap=[cand.overlap] * 3, max_batch_size=cand.max_batch_size,
                        checkpoint_batches=cand.checkpoint_batches,
                        is_best=cand is c, peak_mb=cand.peak_mb, step_s=cand.step_s)
    return None


def tune(cfg: RunEIFullConfig, args, device) -> dict:
    x, y, params, evn_paths, odd_paths, tomo = load_batch(cfg, device)

    def make_physics(ctx):
        physics = build_tomography_physics(cfg, evn_paths, odd_paths, device, ctx)
        physics.update(**params)
        return physics

    model = build_model(cfg, make_physics, device)
    optimizer = build_optimizer(cfg, model)
    step, _ = build_step(cfg, model, optimizer, x, y, tuple(params["init_evn"].shape[-3:]), device)
    tuner = AutoTuner(
        model, make_physics, step, overlap=int(max(cfg.overlap)), optimizer=optimizer,
        gpu_memory_gb=args.gpu_memory_gb, patch_sizes=args.patch_sizes,
        memory_fraction=args.memory_fraction,
        physics_scales=cfg.num_operators is not None,
    )

    t0 = time.perf_counter()
    p_always, p_never = tuner.min_gpus(max_gpus=args.max_gpus or max(args.num_gpus))
    single = {n: tuner.best_single(n) for n in args.num_gpus}
    multi = {n: tuner.best_multi(n) for n in args.num_gpus}
    probe_s = time.perf_counter() - t0

    return dict(
        preset=cfg.preset, config=str(args.config), tomogram=tomo,
        volume_shape=list(params["init_evn"].shape[-3:]), sinogram_shape=list(x.shape[-3:]),
        target_shape=cfg.target_shape, num_operators=cfg.num_operators,
        n_iter=cfg.n_iter if cfg.preset == "unrolled" else None,
        eq_weight=cfg.eq_weight, mixed_precision=cfg.mixed_precision,
        probe_gpu=torch.cuda.get_device_name(device),
        gpu_memory_gb=args.gpu_memory_gb, memory_fraction=tuner.memory_fraction,
        budget_mb=tuner.budget / 2**20, probe_s=probe_s,
        min_gpus={"always": p_always, "never": p_never},
        physics_runs={p: r and {**dataclasses.asdict(r), "m_phys": r.m_phys / 2**20, "at_call": r.at_call / 2**20}
                      for p, r in sorted(tuner._runs.items())},
        best_single={n: as_dict(c) for n, c in single.items()},
        best_multi={n: as_dict(c) for n, c in multi.items()},
        yaml_single={n: yaml_block(c) for n, c in single.items()},
        yaml_multi={n: yaml_block(c) for n, c in multi.items()},
    )


def summarize(r: dict) -> str:
    lines = [f"[autotune] {r['preset']}  {r['tomogram']}  vol={r['volume_shape']}  "
             f"budget={r['budget_mb']:.0f} MiB  probes={r['probe_s']:.1f}s",
             f"  min GPUs/image: always={r['min_gpus']['always']}  never={r['min_gpus']['never']}"]
    for kind in ("best_single", "best_multi"):
        for n, c in r[kind].items():
            if c is None:
                lines.append(f"  {kind}({n}): nothing fits")
                continue
            lines.append(
                f"  {kind}({n}): {c['samples_per_step']}x{c['inner_world_size']} GPUs  "
                f"patch={tuple(c['patch_size'])} dims={tuple(c['tiling_dims'])}  "
                f"mb={c['max_batch_size']}  ckpt={c['checkpoint_batches']}  "
                f"{c['peak_mb']:.0f} MiB  {c['step_s']:.2f} s/step  {c['images_per_s']:.2f} img/s")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--preset", nargs="+", choices=["tomo_ei", "unrolled"], default=["tomo_ei", "unrolled"])
    p.add_argument("--num-gpus", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--max-gpus", type=int, default=None, help="min_gpus search bound (default: max --num-gpus)")
    p.add_argument("--gpu-memory-gb", type=float, default=None, help="target GPU memory (default: this GPU)")
    p.add_argument("--memory-fraction", type=float, default=None)
    p.add_argument("--patch-sizes", type=int, nargs="+", default=None)
    p.add_argument("--target-shape", type=int, nargs=3, default=None, help="override general.target_shape")
    p.add_argument("--out", type=Path, default=None, help="default: runs/autotune/<config stem>")
    args = p.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    out = args.out or ROOT / "runs" / "autotune" / args.config.stem
    out.mkdir(parents=True, exist_ok=True)
    for preset in args.preset:
        r = tune(load_cfg(args.config, preset, args.target_shape), args, device)
        print(summarize(r), flush=True)
        path = out / f"autotune_{preset}.json"
        path.write_text(json.dumps(r, indent=2, default=str))
        print(f"[autotune] saved {path}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
