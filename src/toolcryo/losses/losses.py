"""losses.py — EI loss terms for cryo-ET equivariant training.

Implements the two loss terms from icecream's EquivariantTrainer.compute_loss:

1. ``ObsLoss`` — Cross half-set data-fidelity in the Fourier domain
       L_obs = fourier_loss(EVN, f(ODD), W) + fourier_loss(ODD, f(EVN), W)

2. ``EqLoss`` — Equivariance under random cube-symmetry rotations,
   measured in the Fourier domain under the *rotated* wedge.

Both are ``deepinv.loss.Loss`` subclasses so they plug straight into the deepinv
Trainer ``losses`` list.

The ``fourier_loss`` formula (icecream default, ``use_fourier=False, view_as_real=True``):
    FFT both target and estimate to ``mask_size`` (zero-pad),
    apply the binary wedge mask in centred Fourier space,
    IFFT back to real space (take ``.real``),
    crop back to ``crop_size³``,
    compute plain MSE — **no** ``1/sqrt(N³)`` normalisation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from deepinv.loss import Loss

from ..transform import Rotate3D
from ..icecream_orig.utils.utils import (
    fourier_loss as _ic_fourier_loss,
    fourier_loss_batch as _ic_fourier_loss_batch,
    get_measurement as _ic_get_measurement,
)


def _initialize_window(shape: int | tuple[int, int, int]) -> torch.Tensor:
    """Build the 3-D box window used in icecream's ``initialize_window``.

    Inner region ``[s//4 : -s//4]`` along each axis = 1, rest = 0.
    Accepts either a single int (cubic) or a (D, H, W) tuple.
    """
    if isinstance(shape, int):
        shape = (shape, shape, shape)
    D, H, W = shape
    w = np.zeros((D, H, W), dtype=np.float32)
    qd, qh, qw = D // 4, H // 4, W // 4
    w[qd:-qd, qh:-qh, qw:-qw] = 1.0
    return torch.from_numpy(w)


# ---------------------------------------------------------------------------
# Low-level Fourier loss helpers  (exact icecream clones)
# ---------------------------------------------------------------------------

def _fourier_loss(
    target: torch.Tensor,
    estimate: torch.Tensor,
    wedge: torch.Tensor,
    criteria: nn.Module,
    window: torch.Tensor | None = None,
    use_fourier: bool = False,
    view_as_real: bool = True,
) -> torch.Tensor:
    """Delegates to icecream_orig fourier_loss.

    Handles (B, C, D, H, W) by reshaping to (B*C, D, H, W) before the call.
    """
    t = target.reshape(-1, *target.shape[-3:])
    e = estimate.reshape(-1, *estimate.shape[-3:])
    return _ic_fourier_loss(t, e, wedge, criteria, use_fourier=use_fourier, view_as_real=view_as_real, window=window)


def _fourier_loss_batch(
    target: torch.Tensor,
    estimate: torch.Tensor,
    wedge: torch.Tensor,       # (B, M, M, M)  — per-sample rotated wedge
    criteria: nn.Module,
    window: torch.Tensor | None = None,
    use_fourier: bool = False,
    view_as_real: bool = True,
) -> torch.Tensor:
    """Delegates to icecream_orig fourier_loss_batch.

    Handles (B, C, D, H, W) by reshaping to (B*C, D, H, W) before the call.
    """
    t = target.reshape(-1, *target.shape[-3:])
    e = estimate.reshape(-1, *estimate.shape[-3:])
    return _ic_fourier_loss_batch(t, e, wedge, criteria, use_fourier=use_fourier, view_as_real=view_as_real, window=window)


# ---------------------------------------------------------------------------
# Helper: apply wedge mask (icecream's get_measurement)
# ---------------------------------------------------------------------------

def _apply_wedge(x: torch.Tensor, wedge: torch.Tensor) -> torch.Tensor:
    """Delegates to icecream_orig get_measurement.

    Handles (B, C, D, H, W) by reshaping to (B*C, D, H, W) and restoring shape.

    :param x:     (B, [C,] D, H, W)
    :param wedge: (M, M, M)
    :return: same shape as x
    """
    shape = x.shape
    x_4d = x.reshape(-1, *shape[-3:])
    out = _ic_get_measurement(x_4d, wedge)
    return out.reshape(*shape[:-3], *out.shape[-3:])


# ---------------------------------------------------------------------------
# Helpers: wedge mask operations
# ---------------------------------------------------------------------------

def _symmetrize_and_binarize(w: torch.Tensor) -> torch.Tensor:
    """Matches icecream's get_real_binary_filter: symmetrize_3D + average + binarize."""
    from ..icecream_orig.utils.utils import symmetrize_3D
    w_sym = symmetrize_3D(w)
    w = (w + w_sym) / 2.0
    w[w > 0.1] = 1.0
    return w


def _rotate_wedge(wedge: torch.Tensor, kx: int, ky: int, kz: int, axis: int) -> torch.Tensor:
    """Apply the same (kx,ky,kz,axis) rotation used in batch_rot_4vol to a 3D wedge.

    :param wedge: (M, M, M)
    :return: (M, M, M) rotated wedge
    """
    w = wedge.unsqueeze(0)          # (1, M, M, M)  dim1=D, dim2=H, dim3=W
    # icecream: kx→(1,2)=(H,W), ky→(0,2)=(D,W), kz→(0,1)=(D,H)
    w = torch.rot90(w, k=kx, dims=(2, 3))  # (H, W)
    w = torch.rot90(w, k=ky, dims=(1, 3))  # (D, W)
    w = torch.rot90(w, k=kz, dims=(1, 2))  # (D, H)
    if axis != -1:
        w = torch.flip(w, [axis + 1])
    return w.squeeze(0)             # (M, M, M)


# ---------------------------------------------------------------------------
# 1. Data-fidelity loss  (obs_loss in icecream)
# ---------------------------------------------------------------------------

class ObsLoss(Loss):
    """Cross half-set data-fidelity loss in the Fourier domain.

        L = fourier_loss(ODD, A(f(EVN)), W) + fourier_loss(EVN, A(f(ODD)), W)

    :param MissingWedge physics: the MissingWedge physics object (provides ``mask``).
    :param float weight: loss weight (default 1.0).
    """

    def __init__(self, physics, weight: float = 1.0,
                 use_fourier: bool = False, view_as_real: bool = True,
                 no_window: bool = False) -> None:
        super().__init__()
        self.weight           = weight
        self.use_fourier      = use_fourier
        self.view_as_real     = view_as_real
        self.no_window        = no_window
        self._physics  = physics
        self._criteria = nn.MSELoss(reduction="mean")

    @property
    def _wedge_input(self) -> torch.Tensor:
        """(mask_size)³ wedge — matches icecream's get_real_binary_filter(wedge_full[:-1,:-1,:-1])."""
        return _symmetrize_and_binarize(self._physics.mask[:-1, :-1, :-1])

    @property
    def _window(self) -> torch.Tensor | None:
        """Box window matching the actual volume shape, or None if no_window=True."""
        if self.no_window:
            return None
        return _initialize_window(self._physics._volume_shape)

    def forward(
        self,
        x: torch.Tensor,        # EVN patch  (B, 1, D, H, W)
        y: torch.Tensor,        # ODD patch  (same shape)
        x_net: torch.Tensor,
        physics,
        model,
        **kwargs,
    ) -> torch.Tensor:
        wedge  = self._wedge_input.to(x.device)
        _w = self._window
        window = _w.to(x.device) if _w is not None else None

        est_evn = x_net                          # f(EVN)
        est_odd = kwargs.get("y_net")            # f(ODD)
        if est_odd is None:
            est_odd = model(y)

        loss = (
            _fourier_loss(y, est_evn, wedge, self._criteria, window=window,
                          use_fourier=self.use_fourier, view_as_real=self.view_as_real)
            + _fourier_loss(x, est_odd, wedge, self._criteria, window=window,
                            use_fourier=self.use_fourier, view_as_real=self.view_as_real)
        )
        return self.weight * loss


# ---------------------------------------------------------------------------
# 2. Equivariance loss  (equi_loss_est in icecream)
# ---------------------------------------------------------------------------

class EqLoss(Loss):
    """Equivariance loss under random cube-symmetry rotation, Fourier domain.

    For a randomly sampled rotation T (from the 40-element cubic group):

        est_1_rot, est_2_rot = T(f(EVN)), T(f(ODD))
        L_eq = fourier_loss_batch(est_2_ref, f(A(est_1_rot)), wedge_rot)
             + fourier_loss_batch(est_1_ref, f(A(est_2_rot)), wedge_rot)

    :param MissingWedge physics: provides mask buffers.
    :param Rotate3D transform: the 3D rotation transform (for sampling k_idx).
    :param float weight: loss weight (default 2.0).
    """

    def __init__(
        self,
        physics,
        transform: Rotate3D,
        weight: float = 2.0,
        min_distance: float = 0.5,
        use_fourier: bool = False,
        view_as_real: bool = True,
        eq_use_direct: bool = False,
        no_window: bool = False,
    ) -> None:
        super().__init__()
        self.weight           = weight
        self.use_fourier      = use_fourier
        self.eq_use_direct    = eq_use_direct
        self.view_as_real     = view_as_real
        self.no_window        = no_window
        self._physics      = physics
        self._transform    = transform
        self._criteria     = nn.MSELoss(reduction="mean")
        self._min_distance = min_distance
        self._valid_k_sets = self._compute_valid_k_sets(min_distance)

    def _compute_valid_k_sets(self, min_distance: float) -> list[int]:
        """Return indices into ``Rotate3D._KSET`` where the rotated wedge differs
        from the original by ``distance > min_distance``.

        Mirrors icecream's ``generate_all_cube_symmetries_torch`` filter.
        Volumes are always cubic so all 40 rotations are shape-preserving.
        """
        wedge = self._physics.mask[:-1, :-1, :-1].float()
        norm_w = torch.linalg.norm(wedge)
        valid = []
        for i, (kx, ky, kz, axis) in enumerate(Rotate3D._KSET):
            w_rot = _rotate_wedge(wedge, kx, ky, kz, axis)
            dist = torch.linalg.norm(w_rot - wedge) / norm_w
            if dist.item() > min_distance:
                valid.append(i)
        return valid if valid else list(range(len(Rotate3D._KSET)))

    @property
    def _wedge_full(self) -> torch.Tensor:
        """Full (mask_size)³ wedge — used for rotation."""
        return self._physics.mask

    @property
    def _wedge_input(self) -> torch.Tensor:
        """(mask_size)³ wedge — matches icecream's get_real_binary_filter(wedge_full[:-1,:-1,:-1])."""
        return _symmetrize_and_binarize(self._physics.mask[:-1, :-1, :-1])

    @property
    def _wedge_ref(self) -> torch.Tensor:
        """(D,H,W) wedge at native volume resolution — matches icecream's ``wedge_ref`` for A_ref."""
        return self._physics.mask_ref

    @property
    def _window(self) -> torch.Tensor | None:
        """Box window matching the actual volume shape, or None if no_window=True."""
        if self.no_window:
            return None
        return _initialize_window(self._physics._volume_shape)

    def _rotate_batch_per_sample(self, x: torch.Tensor, k_indices: torch.Tensor) -> torch.Tensor:
        """Rotate each sample in a batch with its own k-index (icecream batch_rot behavior)."""
        out = []
        for i in range(x.shape[0]):
            out.append(self._transform.transform(x[i:i + 1], k_idx=int(k_indices[i].item())))
        return torch.cat(out, dim=0)

    def _rotate_wedge_batch(
        self,
        wedge_full: torch.Tensor,
        k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate full wedge per sample, then crop + symmetrize + binarize.

        Matches icecream's get_real_binary_filters_batch: symmetrize_3D then binarize.
        """
        w_batch = []
        for i in range(k_indices.shape[0]):
            kx, ky, kz, axis = Rotate3D._KSET[int(k_indices[i].item())]
            w_rot = _rotate_wedge(wedge_full, kx, ky, kz, axis)
            w_rot = w_rot[:-1, :-1, :-1]
            w_rot = _symmetrize_and_binarize(w_rot)
            w_batch.append(w_rot)
        return torch.stack(w_batch, dim=0)

    def _paired_eq_loss(
        self,
        est_1: torch.Tensor,
        est_2: torch.Tensor,
        model: nn.Module,
        wedge_full: torch.Tensor,
        wedge_ref: torch.Tensor,
        wedge_input: torch.Tensor,
        window: torch.Tensor,
        k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Icecream-equivalent paired equivariant term (cross-coupled).

        Matches eq_trainer compute_loss:
          est_1_rot, est_2_rot = batch_rot_4vol(...)
          wedge_rot = rotated_wedge[:-1,:-1,:-1]
          est_1_ref = A_ref(est_1_rot), est_2_ref = A_ref(est_2_rot)
          est_1_rot_est = f(A_input(est_1_rot)), est_2_rot_est = f(A_input(est_2_rot))
          L_eq = fourier_loss_batch(est_2_ref, est_1_rot_est, wedge_rot)
               + fourier_loss_batch(est_1_ref, est_2_rot_est, wedge_rot)
        """
        est_1_rot = self._rotate_batch_per_sample(est_1, k_indices)
        est_2_rot = self._rotate_batch_per_sample(est_2, k_indices)
        wedge_rot_batch = self._rotate_wedge_batch(wedge_full, k_indices)

        est_1_ref = _apply_wedge(est_1_rot, wedge_ref)
        est_2_ref = _apply_wedge(est_2_rot, wedge_ref)

        est_1_rot_inp = _apply_wedge(est_1_rot, wedge_input)
        est_2_rot_inp = _apply_wedge(est_2_rot, wedge_input)

        est_1_rot_est = model(est_1_rot_inp)
        est_2_rot_est = model(est_2_rot_inp)

        if self.eq_use_direct:
            # Plain spatial MSE, no Fourier masking (icecream eq_use_direct=True branch)
            return self._criteria(est_2_ref, est_1_rot_est) + self._criteria(est_1_ref, est_2_rot_est)
        return (
            _fourier_loss_batch(est_2_ref, est_1_rot_est, wedge_rot_batch, self._criteria, window=window,
                                use_fourier=self.use_fourier, view_as_real=self.view_as_real)
            + _fourier_loss_batch(est_1_ref, est_2_rot_est, wedge_rot_batch, self._criteria, window=window,
                                  use_fourier=self.use_fourier, view_as_real=self.view_as_real)
        )

    def forward(
        self,
        x: torch.Tensor,        # EVN patch
        y: torch.Tensor,        # ODD patch
        x_net: torch.Tensor,    # f(EVN), pre-computed by forward_pass
        physics,
        model: nn.Module,
        **kwargs,
    ) -> torch.Tensor:
        # Lazily refresh rotation-index cache when physics angles change.
        current_key = getattr(physics, "tilt_key", None)
        if current_key != getattr(self, "_last_tilt_key", None):
            self._valid_k_sets = self._compute_valid_k_sets(self._min_distance)
            self._last_tilt_key = current_key

        wedge_full  = self._wedge_full.to(x.device)
        wedge_ref   = self._wedge_ref.to(x.device)
        wedge_input = self._wedge_input.to(x.device)
        _w = self._window
        window      = _w.to(x.device) if _w is not None else None

        pool = self._valid_k_sets
        bsz = x.shape[0]
        rand_idx = torch.randint(len(pool), (bsz,), device=x.device)
        k_indices = torch.tensor([pool[int(i.item())] for i in rand_idx], device=x.device)

        est_evn = x_net                          # f(EVN)
        est_odd = kwargs.get("y_net")            # f(ODD)
        if est_odd is None:
            est_odd = model(y)

        loss = self._paired_eq_loss(
            est_evn, est_odd, model,
            wedge_full, wedge_ref, wedge_input, window, k_indices,
        )
        return self.weight * loss
