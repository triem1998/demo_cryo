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

from .utils.plot import save_fsc_figure, save_resolution_histogram, save_slice_figure
from .utils.utils import GpuFSC, PerfProbe, append_metrics_row, fsc_shell, half_set_recon


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
        self._val_pixel_sizes: list = []
        self._val_resolutions: list = []
        self._val_vol_idx: int = 0
        self._val_fsc_epoch = None
        # figure tracking
        self._train_slice_epoch = None
        self._train_vol_idx: int = 0
        self._train_batch_counter: int = 0

    # ------------------------------------------------------------------
    # EI forward pass — f(EVN) and f(ODD) independently
    # ------------------------------------------------------------------

    def forward_pass(self, x, y, physics, train):
        x_net = self.model_inference(y=x, physics=physics, x=y, train=train)
        y_net = self.model_inference(y=y, physics=physics, x=x, train=train)
        if train:
            self._last_train_xnet = x_net
            self._last_train_ynet = y_net
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
                                   physics=physics, model=self.model, epoch=epoch)
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
                else:
                    self.optimizer.step()

            self._save_train_figures(x, y, epoch, physics)

        return loss_total, x_net, logs

    # ------------------------------------------------------------------
    # Epoch-end logging and checkpointing
    # ------------------------------------------------------------------

    def log_metrics_mlops(self, logs: dict, step: int, train: bool = True) -> None:  # type: ignore[override]
        if not self._is_rank0:
            return

        if self._metrics_dir is not None:
            row = {"epoch": step, "lr": self.optimizer.param_groups[0]["lr"],
                   **{k: v for k, v in logs.items() if isinstance(v, (int, float))}}
            append_metrics_row(self._metrics_dir / ("train_epochs.csv" if train else "val_epochs.csv"), row)

        if train:
            self._epoch_probe.__exit__(None, None, None)
            n = max(1, self._train_batch_count)
            t, peak_mb = self._epoch_probe.elapsed_s, self._epoch_probe.peak_mb
            if self._block_start_time is None:
                self._block_start_time = time.perf_counter() - t
            if step % self._log_every_n_epochs == 0:
                block_elapsed = time.perf_counter() - self._block_start_time
                loss_str = "  ".join(f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, float))
                gpu_str = (f"  max_gpu={peak_mb/1024:.2f}/"
                           f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB"
                           if torch.cuda.is_available() else "")
                n_ep = self._log_every_n_epochs
                block_str = f"  [{n_ep}ep: {block_elapsed:.1f}s, {block_elapsed/n_ep:.2f}s/ep]" if n_ep > 1 else ""
                print(f"[train ep={step}]  {loss_str}  total={t:.1f}s  per_img={t/n:.2f}s{gpu_str}{block_str}", flush=True)
                self._block_start_time = time.perf_counter()
            self._val_probe = PerfProbe()
            self._val_probe.__enter__()
            self._val_batch_count = 0
        else:
            if self._val_probe is not None and step % self._log_every_n_epochs == 0:
                self._val_probe.__exit__(None, None, None)
                t, n = self._val_probe.elapsed_s, max(1, self._val_batch_count)
                print(f"[val   ep={step}]  total={t:.1f}s  per_img={t/n:.2f}s  n={n}", flush=True)
            self._val_probe = None

            if self._ckpt_dir is not None and step % self.ckp_interval == 0:
                self._ckpt_dir.mkdir(parents=True, exist_ok=True)
                state = {
                    "epoch": step,
                    "model_state_dict": getattr(self.model, "processor", self.model).state_dict(),
                    "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                }
                torch.save(state, self._ckpt_dir / f"ckp_{step:04d}.pth")
                print(f"[ckpt] saved ckp_{step:04d}.pth", flush=True)

    def _enable_mixed_precision(self, device_type: str = "cuda") -> None:
        self._scaler = torch.amp.GradScaler(device_type)
        self._autocast = torch.amp.autocast(device_type)

    def plot(self, epoch, physics, x, y, x_net, train=True):  # type: ignore[override]
        """Suppress the default deepinv plot."""


# ---------------------------------------------------------------------------

class EIFullTrainer(BaseTrainer):
    """Full-volume trainer. Val: FSC(f(EVN), f(ODD)) + figures."""

    def compute_loss(self, physics, x, y, train=True, epoch=None, step=False):  # type: ignore[override]
        if not train:
            if epoch != self._val_fsc_epoch:
                self._val_fsc_epoch = epoch
                self._val_resolutions = []
                self._val_vol_idx = 0

            vol_idx = self._val_vol_idx
            px      = self._val_pixel_sizes[vol_idx] if vol_idx < len(self._val_pixel_sizes) else 1.0

            with torch.no_grad():
                f_evn_t = self.model(x)
                f_odd_t = self.model(y)
            if hasattr(self.device, "type") and self.device.type == "cuda":
                torch.cuda.synchronize()

            if not hasattr(self, "_gpu_fsc"):
                self._gpu_fsc = GpuFSC(f_evn_t.shape[-1], device=f_evn_t.device)

            fsc_curve = self._gpu_fsc(f_evn_t, f_odd_t)
            D         = int(f_evn_t.shape[-1])
            k         = fsc_shell(fsc_curve, self._fsc_threshold)
            res       = D * px / max(k, 1)
            self._val_resolutions.append(res)

            # All ranks must call the distributed model; only rank-0 saves figures.
            with torch.no_grad():
                recon_t = half_set_recon(self.model, physics, f_evn_t, f_odd_t)
            if self._images_dir is not None:
                save_fsc_figure(self._images_dir, epoch, f"vol{vol_idx:02d}.png",
                                fsc_curve, k, res, f"Epoch {epoch} | Vol {vol_idx}",
                                self._fsc_threshold, vol_size=D, pixel_size=px)
                save_slice_figure(
                    self._images_dir, epoch, vol_idx,
                    [x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy(),
                     _znorm_np(recon_t.squeeze().cpu().numpy())],
                    labels=["EVN", "ODD", "recon"],
                    title=f"Epoch {epoch} | Vol {vol_idx} — inference recon",
                    fname=f"vol{vol_idx:02d}_recon.png",
                )

            self._val_vol_idx += 1
            return torch.tensor(0.0, device=y.device), f_evn_t.detach(), {}

        return super().compute_loss(physics, x, y, train=True, epoch=epoch, step=step)

    def _save_train_figures(self, x, y, epoch, physics) -> None:
        if epoch != self._train_slice_epoch:
            self._train_slice_epoch = epoch
            self._train_vol_idx = 0
        vol_idx = self._train_vol_idx
        self._train_vol_idx += 1
        if epoch % self.eval_interval != 0:
            return
        # All ranks must call the distributed model; only rank-0 saves figures.
        with torch.no_grad():
            recon_t = half_set_recon(self.model, physics, self._last_train_xnet, self._last_train_ynet)
        if self._train_images_dir is None:
            return
        save_slice_figure(
            self._train_images_dir, epoch, vol_idx,
            [x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy(),
             _znorm_np(recon_t.squeeze().cpu().numpy())],
            labels=["EVN", "ODD", "recon"],
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
                        fsc_res_q1=q1_res, fsc_res_q3=q3_res)
            if self._images_dir is not None:
                save_resolution_histogram(
                    self._images_dir, step, self._val_resolutions,
                    mean_res, median_res, q1_res, q3_res,
                    threshold_label=str(self._fsc_threshold),
                )
            if self.verbose:
                print(f"[fsc-eval] epoch={step}  mean={mean_res:.1f} Å  median={median_res:.1f} Å  "
                      f"Q1={q1_res:.1f} Å  Q3={q3_res:.1f} Å  (lower=better)", flush=True)
        super().log_metrics_mlops(logs, step, train=train)


# ---------------------------------------------------------------------------

class EIPatchTrainer(BaseTrainer):
    """Patch trainer. Val: loss evaluation via base. Train: slice figures."""

    def _save_train_figures(self, x, y, epoch, physics) -> None:
        if self._images_dir is None:
            return
        if epoch != self._train_slice_epoch:
            self._train_slice_epoch = epoch
            self._train_batch_counter = 0
        if self._train_batch_counter == 0 and epoch % self._log_every_n_epochs == 0:
            save_slice_figure(
                self._images_dir, epoch, 0,
                [x[0].squeeze().cpu().numpy(), y[0].squeeze().cpu().numpy(),
                 self._last_train_ynet.detach()[0].squeeze().cpu().numpy(),
                 self._last_train_xnet.detach()[0].squeeze().cpu().numpy()],
                labels=["Input EVN", "Input ODD", "f(EVN)", "f(ODD)"],
                title=f"Train Epoch {epoch} | Batch 0",
                fname="batch0000_raw.png",
            )
        self._train_batch_counter += 1
