"""Trainers for cryo-ET EI self-supervised training.

Class hierarchy:
  BaseTrainer      — infrastructure (grad accum, AMP, CSV, timing, ckpt) + EI forward pass
    EIFullTrainer  — val: FSC(f(EVN), f(ODD)) + figures; log: resolution histogram
    EIPatchTrainer — train: patch slice figures every _log_every_n_epochs
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path

import deepinv as dinv
import numpy as np
import torch
import torch.nn as nn

from .base_config import amp_dtype_from_str
from .forward import ei_denoiser_forward
from .utils.plot import save_fsc_figure, save_resolution_histogram, save_slice_figure
from .utils.utils import (
    GpuFSC, PerfProbe, append_fsc_row, append_metrics_row, denoise_patches, fsc_resolution, half_set_recon,
    load_mrc_volume, psnr, recon_panels, to_canonical_np,
)


def _znorm_np(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.mean()) / (arr.std() + 1e-8)


class BaseTrainer(dinv.Trainer):

    def _init_trainer_state(self) -> None:
        """Declare all mutable trainer attrs with defaults. Call once before training."""
        # directories
        self._metrics_dir: Path | None = None
        self._images_dir: Path | None = None
        self._train_images_dir: Path | None = None
        self._ckpt_dir: Path | None = None
        # config
        self._grad_accum_steps: int = 1
        self.ckp_interval: int = 10
        self._log_every_n_epochs: int = 1
        self._is_rank0: bool = True
        self._train_sampler = None
        self._scaler = None
        self._autocast = None
        self._amp_dtype = None
        # per-step counters
        self._accum_count: int = 0
        self._current_train_epoch = None
        self._epoch_probe: PerfProbe = PerfProbe()
        self._val_probe: PerfProbe | None = None
        self._train_batch_count: int = 0
        self._val_batch_count: int = 0
        self._block_start_time: float | None = None
        # EI forward pass outputs
        self._last_train_xnet = None
        self._last_train_ynet = None
        # FSC eval (EIFullTrainer)
        self._fsc_threshold: float = 0.143
        self._fsc_split: str = "val"
        self._save_fsc_curves: bool = True
        self._val_pixel_sizes: list = []
        self._val_resolutions: list = []
        self._val_vol_idx: int = 0
        self._val_fsc_epoch = None
        self._fsc_tomo_names: list[str] = []
        self._psnr_refs: list = []          # one Path (or None) per FSC volume
        self._psnr_cache: dict = {}         # vol_idx -> loaded reference array
        self._val_psnr: list = []           # per-volume (psnr_1, psnr_2, std_ratio)
        # figure tracking
        self._train_slice_epoch = None
        self._train_vol_idx: int = 0
        self._train_batch_counter: int = 0
        # patch-position probe (EIPatchTrainer)
        self._patch_probes: list | None = None
        self._patch_probe_wedge = None
        self._patch_probe_dir: Path | None = None
        # forward-pass strategy (how x_net/y_net are computed from a batch + physics)
        self._forward_strategy = ei_denoiser_forward
        # ReduceLROnPlateau, stepped manually from log_metrics_mlops (not
        # self.scheduler — see _build_plateau_scheduler in run.py).
        self._plateau_scheduler = None
        # global epoch of a resumed checkpoint (see setup_train / run.py's
        # _resume_training_state). None = fresh run, start at epoch 0.
        self._resume_epoch: int | None = None
        # hook: called right after optimizer.step() (e.g. clamping trainable
        # algo params — unrolled preset's stepsize must stay positive). No-op
        # by default; harmless for missingwedge_ei.
        self._post_optimizer_step = lambda: None
        # how the displayed reconstruction is formed from the two half-set
        # outputs (set from the preset in run.py; see utils.half_set_recon /
        # utils.unrolled_recon)
        self._recon_strategy = half_set_recon

    def setup_train(self, train: bool = True, **kwargs) -> None:
        """Continue a resumed run on the global epoch timeline.

        dinv.Trainer.setup_train resets ``epoch_start`` to 0, so this has to
        run after it. Keeping the global epoch keeps ckp_/fsc_epoch names, the
        metrics CSV and the ``epoch % eval_interval`` phase continuous across
        the restart.
        """
        super().setup_train(train=train, **kwargs)
        if self._resume_epoch is not None:
            self.epoch_start = int(self._resume_epoch) + 1

    # ------------------------------------------------------------------
    # EI forward pass — f(EVN) and f(ODD) independently
    # ------------------------------------------------------------------

    def forward_pass(self, x, y, physics, train):
        x_net, y_net = self._forward_strategy(self, x, y, physics, train)
        if train:
            self._last_train_xnet = x_net.detach()
            self._last_train_ynet = y_net.detach()
        return x_net, y_net

    def _save_train_figures(self, x, y, epoch, physics) -> None:
        """Hook: called after each train step. Override in subclasses."""

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def compute_loss(self, physics, x, y, train=True, epoch=None, step=False):  # type: ignore[override]
        if train:
            if epoch != self._current_train_epoch:
                self._current_train_epoch = epoch
                self._epoch_probe.__enter__()
                self._train_batch_count = 0
                if self._train_sampler is not None:
                    self._train_sampler.set_epoch(epoch)
            self._train_batch_count += 1
        else:
            self._val_batch_count += 1

        at_window_start = self._accum_count % self._grad_accum_steps == 0
        self._accum_count += 1
        at_window_end = self._accum_count % self._grad_accum_steps == 0

        if train and step and at_window_start:
            self.optimizer.zero_grad(set_to_none=True)

        autocast_ctx = self._autocast or contextlib.nullcontext()
        logs: dict = {}
        loss_total = torch.tensor(0.0)

        with torch.enable_grad() if train else torch.no_grad():
            # This block covers only the model passes reached through
            # forward_pass -> model_inference. The losses call model(...)
            # directly, bypassing it; those passes are covered by _amp_model()
            # below. The two sites are disjoint, not redundant — dropping either
            # one leaves half the step in fp32.
            with autocast_ctx:
                x_net, y_net = self.forward_pass(x, y, physics, train=train)
            if x_net is not None:
                x_net = x_net.float()
            if y_net is not None:
                y_net = y_net.float()

            if train or self.compute_eval_losses:
                loss_total = torch.tensor(0.0, device=x.device)
                for k, loss_fn in enumerate(self.losses):
                    loss = loss_fn(x=x, x_net=x_net, y=y, y_net=y_net,
                                   physics=physics, model=self._amp_model(), epoch=epoch)
                    loss_total = loss_total + loss.mean()
                    meters = self.logs_losses_train[k] if train else self.logs_losses_eval[k]
                    meters.update(loss.detach().cpu().numpy())
                    if len(self.losses) > 1:
                        logs[loss_fn.__class__.__name__] = meters.avg
                meters = self.logs_total_loss_train if train else self.logs_total_loss_eval
                meters.update(loss_total.item())
                logs["TotalLoss"] = meters.avg

        if train:
            is_ddp = isinstance(self.model, nn.parallel.DistributedDataParallel)
            bwd_ctx = self.model.no_sync() if (is_ddp and not at_window_end) else contextlib.nullcontext()
            with bwd_ctx:
                if self._scaler is not None:
                    self._scaler.scale(loss_total / self._grad_accum_steps).backward()
                else:
                    (loss_total / self._grad_accum_steps).backward()

            if step and at_window_end:
                if self._scaler is not None:
                    self._scaler.unscale_(self.optimizer)
                norm = self.check_clip_grad()
                if norm is not None:
                    logs["gradient_norm"] = self.check_grad_val.avg
                if self._scaler is not None:
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                    logs["amp_scale"] = self._scaler.get_scale()
                else:
                    self.optimizer.step()
                self._post_optimizer_step()

            self._save_train_figures(x, y, epoch, physics)
            self._last_train_xnet = self._last_train_ynet = None
            logs.setdefault("gradient_norm", "")

        return loss_total, x_net, logs

    # ------------------------------------------------------------------
    # Epoch-end logging and checkpointing
    # ------------------------------------------------------------------

    def log_metrics_mlops(self, logs: dict, step: int, train: bool = True) -> None:  # type: ignore[override]
        if train and self._plateau_scheduler is not None and "TotalLoss" in logs:
            # All-reduce first: each rank has its own optimizer, and a per-rank
            # local loss could trigger LR drops on different epochs per rank,
            # desyncing the model replicas. Must run before the rank0 return below.
            loss_t = torch.tensor(float(logs["TotalLoss"]), device=self.device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(loss_t, op=torch.distributed.ReduceOp.AVG)
            self._plateau_scheduler.step(loss_t.item())

        if not self._is_rank0:
            return

        if self._metrics_dir is not None:
            row = {"epoch": step, "lr": self.optimizer.param_groups[0]["lr"],
                   **{k: v for k, v in logs.items() if isinstance(v, (int, float, str))}}
            append_metrics_row(self._metrics_dir / ("train_epochs.csv" if train else "val_epochs.csv"), row)

        if train:
            self._epoch_probe.__exit__(None, None, None)
            # astra allocates raw CUDA outside PyTorch's cache; without this it
            # can OOM while PyTorch sits on unused reserved memory.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            n = max(1, self._train_batch_count)
            t, peak_mb = self._epoch_probe.elapsed_s, self._epoch_probe.peak_mb
            if self._block_start_time is None:
                self._block_start_time = time.perf_counter() - t
            if step % self._log_every_n_epochs == 0:
                block_elapsed = time.perf_counter() - self._block_start_time
                loss_str = "  ".join(f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, float))
                # reserved, not just allocated: astra allocates outside PyTorch's
                # caching allocator, so its headroom is total - reserved (see PerfProbe).
                gpu_str = (f"  max_gpu={peak_mb/1024:.2f}"
                           f"(reserved {self._epoch_probe.peak_reserved_mb/1024:.2f})/"
                           f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB"
                           if torch.cuda.is_available() else "")
                n_ep = self._log_every_n_epochs
                block_str = f"  [{n_ep}ep: {block_elapsed:.1f}s, {block_elapsed/n_ep:.2f}s/ep]" if n_ep > 1 else ""
                print(f"[train ep={step}]  {loss_str}  total={t:.1f}s  per_img={t/n:.2f}s{gpu_str}{block_str}", flush=True)
                self._block_start_time = time.perf_counter()
            self._val_probe = PerfProbe()
            self._val_probe.__enter__()
            self._val_batch_count = 0

            # step is the epoch just finished, 0-based, so step+1 are done. The
            # file is named for that; "epoch" stays 0-based because resume does
            # epoch_start = ckpt["epoch"] + 1.
            done = step + 1
            if self._ckpt_dir is not None and done % self.ckp_interval == 0:
                self._ckpt_dir.mkdir(parents=True, exist_ok=True)
                raw_model = self.model.module if isinstance(
                    self.model, nn.parallel.DistributedDataParallel) else self.model
                state = {
                    "epoch": step,
                    "epochs_completed": done,
                    "model_state_dict": getattr(raw_model, "processor", raw_model).state_dict(),
                    "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                    "scheduler": (self._plateau_scheduler.state_dict()
                                  if self._plateau_scheduler is not None else None),
                }
                torch.save(state, self._ckpt_dir / f"ckp_{done:04d}.pth")
                print(f"[ckpt] saved ckp_{done:04d}.pth", flush=True)
        else:
            if self._val_probe is not None and step % self._log_every_n_epochs == 0:
                self._val_probe.__exit__(None, None, None)
                t, n = self._val_probe.elapsed_s, max(1, self._val_batch_count)
                print(f"[val   ep={step}]  total={t:.1f}s  per_img={t/n:.2f}s  n={n}", flush=True)
            self._val_probe = None
            # Same reasoning as the train-side empty_cache() above.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _enable_mixed_precision(self, dtype: str = "fp16", device_type: str | None = None) -> None:
        """Arm autocast (+ GradScaler for fp16). Never called when mixed_precision is "off"."""
        if device_type is None:
            device_type = self.device.type if hasattr(self, "device") else "cuda"
        amp_dtype = amp_dtype_from_str(dtype)
        self._amp_dtype = amp_dtype
        self._autocast = torch.amp.autocast(device_type, dtype=amp_dtype)
        # GradScaler exists to rescue tiny fp16 gradients from underflow via a
        # 65536x loss multiply. bf16 shares fp32's exponent range, so scaling is
        # unnecessary — and it was that multiply that overflowed at native
        # resolution. No scaler for bf16: the trainer already runs a plain
        # .backward() whenever self._scaler is None (the fp32 path).
        self._scaler = torch.amp.GradScaler(device_type) if dtype == "fp16" else None

    def _amp_model(self):
        """``self.model``, wrapped so each call autocasts and returns fp32.

        The equivariance losses run a denoiser pass of their own
        (losses_equivariant_wedge.py:339-340, losses_equivariant_tomo.py:43),
        reached by calling ``model(...)`` directly rather than through
        ``model_inference``. Without this they are the only fp32 model passes in
        the step — roughly half of them.


        A closure, not an ``nn.Module``: wrapping ``self.model`` in a module
        would break the ``isinstance(self.model, DistributedDataParallel)``
        check below and the ``self.model.module`` unwrap in the probe. Reading
        ``self.model`` inside the closure also picks up any DDP/``torch.compile``
        re-wrapping applied after the trainer was built.
        """
        if self._autocast is None:
            return self.model                       # strict no-op on the fp32 path

        def _call(*args, **kwargs):
            with self._autocast:
                out = self.model(*args, **kwargs)
            return out.float()

        return _call

    def plot(self, epoch, physics, x, y, x_net, train=True):  # type: ignore[override]
        """Suppress the default deepinv plot."""


# ---------------------------------------------------------------------------

class EIFullTrainer(BaseTrainer):
    """Full-volume trainer. Val: FSC(f(EVN), f(ODD)) + figures."""

    def _psnr_ref(self, vol_idx: int, shape):
        """Reference volume for ``vol_idx``, canonical order, loaded once.

        Resampled to ``shape`` when the run sets ``target_shape``, then
        z-normalised. A shape that still disagrees means the axis order does, so
        ``psnr`` raises rather than scoring a transposed volume. Rank 0 only.
        """
        if vol_idx in self._psnr_cache:
            return self._psnr_cache[vol_idx]
        path = self._psnr_refs[vol_idx] if vol_idx < len(self._psnr_refs) else None
        ref = None
        if path is not None:
            ref = load_mrc_volume(path, order="native")     # (Y, X, Z) = canonical
            if ref.shape != tuple(shape):
                t = torch.from_numpy(ref)[None, None]
                ref = torch.nn.functional.interpolate(
                    t, size=tuple(shape), mode="trilinear", align_corners=False
                ).squeeze().numpy()
            # z-normalised like every other volume, so std_ratio is std(recon)
            # against a unit-variance reference and 1.0 is the neutral value.
            # dtype=float32: a float16 accumulator overflows to inf here.
            mu = float(np.mean(ref, dtype=np.float32))
            sd = float(np.std(ref, dtype=np.float32))
            ref = ((ref - mu) / (sd + 1e-8)).astype(np.float16)
        self._psnr_cache[vol_idx] = ref
        return ref

    def compute_loss(self, physics, x, y, train=True, epoch=None, step=False):  # type: ignore[override]
        if not train:
            if epoch != self._val_fsc_epoch:
                self._val_fsc_epoch = epoch
                self._val_resolutions = []
                self._val_psnr = []
                self._val_vol_idx = 0

            vol_idx = self._val_vol_idx
            px      = self._val_pixel_sizes[vol_idx] if vol_idx < len(self._val_pixel_sizes) else 1.0

            with torch.no_grad():
                f_evn_t, f_odd_t = self.forward_pass(x, y, physics, train=False)
            if hasattr(self.device, "type") and self.device.type == "cuda":
                torch.cuda.synchronize()

            # Scored apart by FSC (comparable to patch inference); averaged only for display.
            with torch.no_grad():
                r_evn, r_odd = self._recon_strategy(self.model, physics, f_evn_t, f_odd_t)

            if not hasattr(self, "_gpu_fsc"):
                self._gpu_fsc = GpuFSC(device=f_evn_t.device)

            fsc_curve  = self._gpu_fsc(r_evn, r_odd)
            k, res, D  = fsc_resolution(fsc_curve, r_evn.squeeze().shape,
                                        px, self._fsc_threshold)
            self._val_resolutions.append(res)

            # 1-pass f(.) score, before the round trip — guards against the Eq
            # term's collapse mode (2-pass score rising while this one falls).
            has_round_trip = self._recon_strategy is half_set_recon
            res_1 = k_1 = fsc_curve_1 = None
            if has_round_trip:
                fsc_curve_1 = self._gpu_fsc(f_evn_t, f_odd_t)
                k_1, res_1, _ = fsc_resolution(fsc_curve_1, f_evn_t.squeeze().shape,
                                               px, self._fsc_threshold)

            name = (self._fsc_tomo_names[vol_idx] if vol_idx < len(self._fsc_tomo_names)
                    else f"vol{vol_idx:02d}")

            recon_2 = 0.5 * (r_evn + r_odd)
            recon_1 = 0.5 * (f_evn_t + f_odd_t) if has_round_trip else None
            can = lambda t: to_canonical_np(t.squeeze().float().cpu().numpy(), physics)  # noqa: E731

            # Rank 0 only: the epoch row it feeds is rank-0 only too, and no
            # collective runs here, so other ranks skip the read and the reference.
            if self._is_rank0:
                can_2 = can(recon_2)
                ref = self._psnr_ref(vol_idx, can_2.shape)
                if ref is not None:
                    # dtype: the cached ref is fp16 and its default accumulator
                    # overflows to inf on a real volume.
                    self._val_psnr.append((
                        psnr(can(recon_1), ref) if recon_1 is not None else None,
                        psnr(can_2, ref),
                        float(recon_2.std()) / (float(np.std(ref, dtype=np.float32)) + 1e-12),
                    ))
            if self._is_rank0:
                _p = getattr(physics, "physics_evn", physics)   # TomographyEMPair holds the operator
                print(f"[physics] {name}  tilt=[{_p._tilt_min:.1f}, {_p._tilt_max:.1f}]°", flush=True)
            if self._is_rank0 and self._metrics_dir is not None:
                append_fsc_row(self._metrics_dir / "fsc_per_volume.csv",
                               curve=fsc_curve if self._save_fsc_curves else None,
                               curve_1pass=fsc_curve_1 if self._save_fsc_curves else None,
                               mode="train", regime="full", split=self._fsc_split,
                               epoch=epoch, vol_idx=vol_idx, tomo=name,
                               pixel_size=px, n_ref=D,
                               fsc_threshold=self._fsc_threshold,
                               fsc_shell=int(k), fsc_res_angstrom=float(res),
                               **({"fsc_shell_1pass": int(k_1),
                                   "fsc_res_1pass_angstrom": float(res_1)} if has_round_trip else {}))

            recon_t = recon_2
            if self._images_dir is not None:
                save_fsc_figure(self._images_dir, epoch, f"{name}.png",
                                fsc_curve, k, res, f"Epoch {epoch} | {name}",
                                self._fsc_threshold, vol_size=D, pixel_size=px,
                                fsc_curve_1=fsc_curve_1, res_shell_1=k_1,
                                res_angstrom_1=res_1)
                pa, pb, pl = recon_panels(x, y, physics)
                cols, labels = [pa, pb], list(pl)
                if has_round_trip:
                    cols.append(_znorm_np(to_canonical_np(recon_1.squeeze().cpu().numpy(), physics)))
                    labels.append(f"1 pass  f(.)\n{res_1:.1f} Å")
                cols.append(_znorm_np(to_canonical_np(recon_t.squeeze().cpu().numpy(), physics)))
                labels.append(f"2 pass  f(A(f(.)))\n{res:.1f} Å" if has_round_trip else "recon")
                save_slice_figure(
                    self._images_dir, epoch, vol_idx, cols, labels=labels,
                    title=f"Epoch {epoch} | {name} — inference recon",
                    fname=f"{name}_recon.png",
                )

            self._val_vol_idx += 1
            return torch.tensor(0.0, device=y.device), f_evn_t.detach(), {}

        return super().compute_loss(physics, x, y, train=True, epoch=epoch, step=step)

    def _save_train_figures(self, x, y, epoch, physics) -> None:
        if self._fsc_split == "train" and self._val_pixel_sizes:
            return  # FSC eval already reconstructs + plots these same volumes
        if epoch != self._train_slice_epoch:
            self._train_slice_epoch = epoch
            self._train_vol_idx = 0
        vol_idx = self._train_vol_idx
        self._train_vol_idx += 1
        if epoch % self.eval_interval != 0:
            return
        # All ranks must call the (possibly distributed) model; only rank-0 saves.
        with torch.no_grad():
            r_evn, r_odd = self._recon_strategy(
                self.model, physics, self._last_train_xnet, self._last_train_ynet)
        recon_t = 0.5 * (r_evn + r_odd)
        if self._train_images_dir is None:
            return
        pa, pb, pl = recon_panels(x, y, physics)
        save_slice_figure(
            self._train_images_dir, epoch, vol_idx,
            [pa, pb, _znorm_np(to_canonical_np(recon_t.squeeze().cpu().numpy(), physics))],
            labels=[*pl, "recon"],
            title=f"Train Epoch {epoch} | Vol {vol_idx} — inference recon",
            fname=f"vol{vol_idx:02d}_recon.png",
        )

    def log_metrics_mlops(self, logs: dict, step: int, train: bool = True) -> None:  # type: ignore[override]
        if not train and self._val_resolutions:
            res_arr    = np.array(self._val_resolutions)
            mean_res   = float(np.mean(res_arr))
            median_res = float(np.median(res_arr))
            q1_res     = float(np.percentile(res_arr, 25))
            q3_res     = float(np.percentile(res_arr, 75))
            logs.update(fsc_res_angstrom=mean_res, fsc_res_median=median_res,
                        fsc_res_q1=q1_res, fsc_res_q3=q3_res, fsc_split=self._fsc_split)
            if self._images_dir is not None:
                save_resolution_histogram(
                    self._images_dir, step, self._val_resolutions,
                    mean_res, median_res, q1_res, q3_res,
                    threshold_label=str(self._fsc_threshold),
                )
            if self.verbose:
                print(f"[fsc-eval] epoch={step}  mean={mean_res:.1f} Å  median={median_res:.1f} Å  "
                      f"Q1={q1_res:.1f} Å  Q3={q3_res:.1f} Å  (lower=better)", flush=True)

        if not train and self._val_psnr:
            # Mean over volumes, as above. psnr_ref names the file the numbers
            # are against; PSNR to a ground truth and to icecream do not compare.
            p1 = [v[0] for v in self._val_psnr if v[0] is not None]
            logs.update(psnr_2pass=float(np.mean([v[1] for v in self._val_psnr])),
                        std_ratio=float(np.mean([v[2] for v in self._val_psnr])),
                        psnr_ref=(self._psnr_refs[0].name if self._psnr_refs
                                  and self._psnr_refs[0] else ""))
            if p1:
                logs.update(psnr_1pass=float(np.mean(p1)))
            if self.verbose:
                print(f"[psnr] epoch={step}  2-pass={logs['psnr_2pass']:.2f} dB  "
                      f"std_ratio={logs['std_ratio']:.3f}  (higher=better)", flush=True)
        super().log_metrics_mlops(logs, step, train=train)


# ---------------------------------------------------------------------------

class EIPatchTrainer(BaseTrainer):
    """Patch trainer. Val: loss evaluation via base. Train: slice figures."""

    def _save_train_figures(self, x, y, epoch, physics) -> None:
        """Once per log-interval epoch (first batch): denoise the fixed probe
        crops and save an EVN / ODD / predict figure per (tomo, position)."""
        if self._patch_probe_dir is None or not self._patch_probes:
            return
        if epoch != self._train_slice_epoch:
            self._train_slice_epoch = epoch
            self._train_batch_counter = 0
        first = self._train_batch_counter == 0
        self._train_batch_counter += 1
        if not first or epoch % self._log_every_n_epochs != 0:
            return

        model = self.model.module if isinstance(
            self.model, nn.parallel.DistributedDataParallel) else self.model
        was_training = model.training
        model.eval()
        epoch_dir = self._patch_probe_dir / f"epoch{epoch:04d}"
        for tomo_name, evn_crops, odd_crops, origins in self._patch_probes:
            recon_evn = denoise_patches(evn_crops, model, self._patch_probe_wedge, self.device,
                                        amp_dtype=self._amp_dtype)
            recon_odd = denoise_patches(odd_crops, model, self._patch_probe_wedge, self.device,
                                        amp_dtype=self._amp_dtype)
            recon = 0.5 * (recon_evn + recon_odd)
            for j, (d0, h0, w0) in enumerate(origins):
                save_slice_figure(
                    epoch_dir, epoch, j,
                    [evn_crops[j].cpu().numpy(), odd_crops[j].cpu().numpy(), recon[j].numpy()],
                    labels=["EVN", "ODD", "predict"],
                    title=f"{tomo_name} | pos ({d0},{h0},{w0}) | epoch {epoch}",
                    subdir=".",
                    fname=f"{tomo_name}_pos{d0}-{h0}-{w0}.png",
                )
        if was_training:
            model.train()
