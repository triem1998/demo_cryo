"""losses_custom.py — Self-contained EI loss terms (no icecream dependency).

Implements ObsLoss and EqLoss with hardcoded defaults that match
the always-used icecream configuration:
  use_fourier=False, view_as_real=True, eq_use_direct=False,
  no_window=False, min_distance=0.5, criteria=MSELoss

Math:
  A_w(x) = IFFT(fftshift(FFT(x, size=wedge.shape) * wedge))[:crop]  — complex

  _masked_mse(a, b, wedge, win) = |A_w(a-b) * win|².mean() / 2
    (equivalent to MSE(view_as_real(A_w(a)*win), view_as_real(A_w(b)*win)))

  ObsLoss:
    L = _masked_mse(ODD, f(EVN), wedge_in, win)
      + _masked_mse(EVN, f(ODD), wedge_in, win)

  EqLoss  (est1=T(f(EVN)), est2=T(f(ODD))):
    L = _masked_mse(A_ref(est2), f(A_in(est1)), wedge_rot, win)
      + _masked_mse(A_ref(est1), f(A_in(est2)), wedge_rot, win)
"""
from __future__ import annotations

import numpy as np
import torch
from deepinv.loss import Loss

from ..transform import Rotate3D


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _initialize_window(shape: int | tuple[int, int, int]) -> torch.Tensor:
    """Box window: inner 50% along each axis = 1, rest = 0."""
    if isinstance(shape, int):
        shape = (shape, shape, shape)
    D, H, W = shape
    w = np.zeros((D, H, W), dtype=np.float32)
    w[D // 4: -D // 4, H // 4: -H // 4, W // 4: -W // 4] = 1.0
    return torch.from_numpy(w)


def _symmetrize_3d(x: torch.Tensor) -> torch.Tensor:
    """Flip each axis leaving the first row/col unchanged (icecream symmetrize_3D)."""
    x = x.clone()
    x[1:] = torch.flip(x[1:], [0])
    x[:, 1:] = torch.flip(x[:, 1:], [1])
    x[:, :, 1:] = torch.flip(x[:, :, 1:], [2])
    return x


def _symmetrize_and_binarize(w: torch.Tensor) -> torch.Tensor:
    """Symmetrize wedge then binarize at threshold 0.1."""
    w_sym = _symmetrize_3d(w)
    w = (w + w_sym) / 2.0
    w[w > 0.1] = 1.0
    return w


def _apply_wedge(x: torch.Tensor, wedge: torch.Tensor) -> torch.Tensor:
    """Apply wedge mask in Fourier space. Returns COMPLEX tensor same spatial shape as x.

    Steps: zero-pad FFT to wedge.shape → fftshift → multiply wedge
           → ifftshift → IFFT → crop to original spatial shape.
    """
    n1, n2, n3 = x.shape[-3], x.shape[-2], x.shape[-1]
    x_4d = x.reshape(-1, n1, n2, n3)
    X = torch.fft.fftshift(
        torch.fft.fftn(x_4d, s=tuple(wedge.shape), dim=(-3, -2, -1)),
        dim=(-3, -2, -1),
    )
    X = X * wedge[None]
    out = torch.fft.ifftn(torch.fft.ifftshift(X, dim=(-3, -2, -1)), dim=(-3, -2, -1))
    out = out[..., :n1, :n2, :n3]
    return out.reshape(*x.shape[:-3], n1, n2, n3)


def _masked_mse(
    a: torch.Tensor,
    b: torch.Tensor,
    wedge: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    """Fourier-masked MSE: |A_w(a-b) * win|².mean() / 2.

    Equivalent to MSE(view_as_real(A_w(a)*win), view_as_real(A_w(b)*win)).
    """
    diff = _apply_wedge(a - b, wedge) * window
    return diff.abs().pow(2).mean() / 2


def _masked_mse_batch(
    a: torch.Tensor,
    b: torch.Tensor,
    wedge_batch: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    """Per-sample _masked_mse with per-sample wedge (B, M, M, M), averaged over batch."""
    B = a.shape[0]
    loss = torch.zeros(1, device=a.device, dtype=a.dtype)
    for i in range(B):
        loss = loss + _masked_mse(a[i:i+1], b[i:i+1], wedge_batch[i], window)
    return loss / B


def _rotate_wedge(wedge: torch.Tensor, kx: int, ky: int, kz: int, axis: int) -> torch.Tensor:
    """Apply the (kx,ky,kz,axis) cube-symmetry rotation to a 3D wedge (M,M,M)."""
    w = wedge.unsqueeze(0)
    w = torch.rot90(w, k=kx, dims=(2, 3))
    w = torch.rot90(w, k=ky, dims=(1, 3))
    w = torch.rot90(w, k=kz, dims=(1, 2))
    if axis != -1:
        w = torch.flip(w, [axis + 1])
    return w.squeeze(0)


# ---------------------------------------------------------------------------
# 1. ObsLoss
# ---------------------------------------------------------------------------

class ObsLoss(Loss):
    """Cross half-set data-fidelity loss in the Fourier domain.

    L = MSE(view_as_real(A_w(ODD)*win), view_as_real(A_w(f(EVN))*win))
      + MSE(view_as_real(A_w(EVN)*win), view_as_real(A_w(f(ODD))*win))

    where A_w applies the symmetrized+binarized missing-wedge mask.
    """

    def __init__(self, physics, weight: float = 1.0) -> None:
        super().__init__()
        self.weight   = weight
        self._physics = physics

    @property
    def _wedge(self) -> torch.Tensor:
        return _symmetrize_and_binarize(self._physics.mask[:-1, :-1, :-1])

    @property
    def _window(self) -> torch.Tensor:
        return _initialize_window(self._physics._volume_shape)

    def forward(self, x, y, x_net, physics, model, **kwargs) -> torch.Tensor:
        wedge  = self._wedge.to(x.device)
        window = self._window.to(x.device)

        est_evn = x_net
        est_odd = kwargs.get("y_net")
        if est_odd is None:
            est_odd = model(y)

        # Reshape (B,1,D,H,W) → (B,D,H,W) for the wedge operator
        odd_4d = y.reshape(-1, *y.shape[-3:])
        evn_4d = x.reshape(-1, *x.shape[-3:])
        est_evn_4d = est_evn.reshape(-1, *est_evn.shape[-3:])
        est_odd_4d = est_odd.reshape(-1, *est_odd.shape[-3:])

        loss = (
            _masked_mse(odd_4d, est_evn_4d, wedge, window)
            + _masked_mse(evn_4d, est_odd_4d, wedge, window)
        )
        return self.weight * loss


# ---------------------------------------------------------------------------
# 2. EqLoss
# ---------------------------------------------------------------------------

class EqLoss(Loss):
    """Equivariance loss under random cube-symmetry rotation, Fourier domain.

    est1 = T(f(EVN)),  est2 = T(f(ODD))

    L = MSE(view_as_real(A_rot(A_ref(est2))*win), view_as_real(A_rot(f(A_in(est1)))*win))
      + MSE(view_as_real(A_rot(A_ref(est1))*win), view_as_real(A_rot(f(A_in(est2)))*win))

    Rotations are sampled from the subset of the 40-element cubic group where
    the rotated wedge differs from the original by normalized L2 distance > 0.5.
    """

    _MIN_DISTANCE = 0.5

    def __init__(self, physics, transform: Rotate3D, weight: float = 2.0) -> None:
        super().__init__()
        self.weight        = weight
        self._physics      = physics
        self._transform    = transform
        self._valid_k_sets = self._compute_valid_k_sets()
        self._last_tilt_key: object = None

    def _compute_valid_k_sets(self) -> list[int]:
        wedge    = self._physics.mask[:-1, :-1, :-1].float()
        norm_w   = torch.linalg.norm(wedge)
        valid    = []
        for i, (kx, ky, kz, axis) in enumerate(Rotate3D._KSET):
            w_rot = _rotate_wedge(wedge, kx, ky, kz, axis)
            dist  = torch.linalg.norm(w_rot - wedge) / norm_w
            if dist.item() > self._MIN_DISTANCE:
                valid.append(i)
        return valid if valid else list(range(len(Rotate3D._KSET)))

    @property
    def _wedge_full(self) -> torch.Tensor:
        return self._physics.mask

    @property
    def _wedge_input(self) -> torch.Tensor:
        return _symmetrize_and_binarize(self._physics.mask[:-1, :-1, :-1])

    @property
    def _wedge_ref(self) -> torch.Tensor:
        return self._physics.mask_ref

    @property
    def _window(self) -> torch.Tensor:
        return _initialize_window(self._physics._volume_shape)

    def _rotate_batch(self, x: torch.Tensor, k_indices: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self._transform.transform(x[i:i+1], k_idx=int(k_indices[i].item()))
             for i in range(x.shape[0])],
            dim=0,
        )

    def _wedge_rot_batch(self, wedge_full: torch.Tensor, k_indices: torch.Tensor) -> torch.Tensor:
        batch = []
        for i in range(k_indices.shape[0]):
            kx, ky, kz, axis = Rotate3D._KSET[int(k_indices[i].item())]
            w = _rotate_wedge(wedge_full, kx, ky, kz, axis)[:-1, :-1, :-1]
            batch.append(_symmetrize_and_binarize(w))
        return torch.stack(batch, dim=0)

    def forward(self, x, y, x_net, physics, model, **kwargs) -> torch.Tensor:
        current_key = getattr(physics, "tilt_key", None)
        if current_key != self._last_tilt_key:
            self._valid_k_sets = self._compute_valid_k_sets()
            self._last_tilt_key = current_key

        wedge_full  = self._wedge_full.to(x.device)
        wedge_input = self._wedge_input.to(x.device)
        wedge_ref   = self._wedge_ref.to(x.device)
        window      = self._window.to(x.device)

        pool    = self._valid_k_sets
        bsz     = x.shape[0]
        rand_idx   = torch.randint(len(pool), (bsz,), device=x.device)
        k_indices  = torch.tensor([pool[int(i.item())] for i in rand_idx], device=x.device)

        est_evn = x_net
        est_odd = kwargs.get("y_net")
        if est_odd is None:
            est_odd = model(y)

        # est1 = T(f(EVN)), est2 = T(f(ODD))
        est1 = self._rotate_batch(est_evn, k_indices)
        est2 = self._rotate_batch(est_odd, k_indices)

        wedge_rot_batch = self._wedge_rot_batch(wedge_full, k_indices).to(x.device)

        # A_ref and A_input return real (imaginary ≈ 0 for real inputs + symmetric wedge)
        est1_ref = _apply_wedge(est1, wedge_ref).real
        est2_ref = _apply_wedge(est2, wedge_ref).real
        f_A_est1 = model(_apply_wedge(est1, wedge_input).real)
        f_A_est2 = model(_apply_wedge(est2, wedge_input).real)

        loss = (
            _masked_mse_batch(est2_ref, f_A_est1, wedge_rot_batch, window)
            + _masked_mse_batch(est1_ref, f_A_est2, wedge_rot_batch, window)
        )
        return self.weight * loss
