"""MissingWedge — deepinv LinearPhysics wrapping icecream's Fourier wedge mask.

Mask construction matches icecream's initialize_wedge exactly:
    - get_wedge_3d_new(mask_size) → shape (mask_size+1)³
    - symmetrize: (mask + flipped_mask) / 2
    - binarize:   values > 0.1 → 1
    - keep full (mask_size+1)³ as ``mask`` (icecream ``wedge_full``)

Forward operator A matches icecream's get_measurement exactly:
    1. fftn(x, s=mask.shape)  — zero-pad volume to mask.shape before FFT
    2. fftshift                — center DC before masking
    3. multiply by binary wedge mask
    4. ifftshift + ifftn.real  — back to real space at mask_size
    5. crop output to original x shape

Volumes are always cubic (centre-cropped to min-dim before reaching here):
    When wedge_double_size=True  (icecream default):
        mask_size = 2 * crop_size, and ``mask`` has shape (mask_size+1)³
    When wedge_double_size=False:
        mask_size = crop_size      →  no zero-padding, mask applied at native size

"""
from __future__ import annotations

import numpy as np
import torch
import deepinv as dinv

from .icecream_orig.utils.utils import symmetrize_3D


class MissingWedge(dinv.physics.LinearPhysics):
    """Missing-wedge Fourier-mask operator for cryo-ET.

    Matches icecream's EquivariantTrainer exactly for cubic volumes.
    When ``volume_shape`` is provided (non-cubic volumes), the wedge is built
    at ``max(volume_shape)`` cubic and then **cropped** to the actual volume
    dimensions.  Cropping the centred Fourier grid is physically correct: it
    retains the low-frequency region that corresponds to the spatial resolution
    of each axis.

    :param float tilt_max: Maximum tilt angle in degrees (default 60).
    :param float tilt_min: Minimum tilt angle in degrees (default -60).
    :param int crop_size: Cubic patch side length — used for wedge_ref and window.
    :param tuple[int,int,int] | None volume_shape: Actual (D, H, W) of the
        volume the mask will be applied to.  ``None`` = cubic (legacy behaviour).
    :param bool use_spherical_support: Enforce spherical support in Fourier space (default True).
    :param bool wedge_double_size: Build mask at 2× the cubic side (icecream default True).
    :param float wedge_low_support: Radius² of a low-frequency ball forced to 1 in the input wedge
        (icecream ``wedge_low_support``). 0 = no leakage (icecream default); 0.1 = 10% low-freq kept.
    :param float ref_wedge_support: Same parameter for the reference wedge used in EqLoss
        (icecream ``ref_wedge_support``). 1.0 = full unit sphere set to 1 (icecream default),
        meaning the equivariance reference sees the complete spectrum.
    :param str device: Device string (default 'cpu').
    """

    def __init__(
        self,
        tilt_max: float = 60.0,
        tilt_min: float = -60.0,
        crop_size: int = 72,
        volume_shape: tuple[int, int, int] | None = None,
        use_spherical_support: bool = True,
        wedge_double_size: bool = True,
        wedge_low_support: float = 0.0,
        ref_wedge_support: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        # ── Resolve effective spatial dimensions ─────────────────────────
        if volume_shape is not None:
            D, H, W = int(volume_shape[0]), int(volume_shape[1]), int(volume_shape[2])
        else:
            D = H = W = crop_size  # cubic — legacy behaviour

        self._volume_shape = (D, H, W)
        self._wedge_double_size = wedge_double_size
        self._wedge_low_support = wedge_low_support
        self._ref_wedge_support = ref_wedge_support
        self._use_spherical_support = use_spherical_support
        self._tilt_min = float(tilt_min)
        self._tilt_max = float(tilt_max)

        mask, mask_ref = self._build_masks(tilt_max, tilt_min, torch.device(device))
        self.register_buffer("mask", mask)
        self.register_buffer("mask_ref", mask_ref)

    def _build_masks(
        self, tilt_max: float, tilt_min: float, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (mask, mask_ref) tensors on ``device`` using pure torch (no NumPy).

        Called once at init and again by update_angles when the tilt range changes.
        Runs on whatever device is passed — GPU when available, CPU otherwise.
        """
        if device is None:
            device = self.mask.device
        D, H, W = self._volume_shape
        max_dim = max(D, H, W)
        mask_size = max_dim * 2 if self._wedge_double_size else max_dim

        mask = self._make_wedge_3d_torch(
            mask_size, tilt_max, tilt_min,
            self._wedge_low_support, self._use_spherical_support, device,
        )
        if not self._wedge_double_size:
            # Crop to (D+1, H+1, W+1) only when using native size.
            # For wedge_double_size=True the full (2*max_dim+1)³ mask must be
            # preserved so fourier_loss can zero-pad volumes to mask_size before FFT.
            full_side = mask_size + 1
            dD = (full_side - (D + 1)) // 2
            dH = (full_side - (H + 1)) // 2
            dW = (full_side - (W + 1)) // 2
            mask = mask[dD: dD + D + 1, dH: dH + H + 1, dW: dW + W + 1]

        mask_ref = self._make_wedge_3d_torch(
            max_dim, tilt_max, tilt_min,
            self._ref_wedge_support, self._use_spherical_support, device,
        )
        full_side_ref = max_dim + 1
        dD_r = (full_side_ref - (D + 1)) // 2
        dH_r = (full_side_ref - (H + 1)) // 2
        dW_r = (full_side_ref - (W + 1)) // 2
        mask_ref = mask_ref[dD_r: dD_r + D + 1, dH_r: dH_r + H + 1, dW_r: dW_r + W + 1]

        return mask, mask_ref[:-1, :-1, :-1]

    @staticmethod
    def _make_wedge_3d_torch(
        mask_size: int,
        tilt_max: float,
        tilt_min: float,
        low_support: float,
        use_spherical: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Pure-torch equivalent of get_wedge_3d_new + symmetrize_3D + binarize.

        Everything runs on ``device`` — no NumPy allocation, no CPU round-trip.
        Uses broadcasting for r² so only one (S,S,S) array is materialised.
        """
        S = mask_size + 1
        tan_max = float(np.tan(np.deg2rad(tilt_max)))
        tan_min = float(np.tan(np.deg2rad(tilt_min)))
        lin = torch.linspace(-1.0, 1.0, S, device=device)

        # ── 2D wedge slice (matches get_wedge_new, rotation=0) ──────────
        # np.meshgrid(x,y) 'xy' default: xx[i,j]=x[j] (axis1), yy[i,j]=y[i] (axis0)
        xx2 = lin.unsqueeze(0).expand(S, S)   # (S,S) — x varies along axis 1
        yy2 = lin.unsqueeze(1).expand(S, S)   # (S,S) — y varies along axis 0
        radius = 10.0 if use_spherical else 2.0
        w2d = (xx2 ** 2 + yy2 ** 2 < radius).float()
        w2d = w2d.clone()
        w2d[yy2 > tan_max * xx2] = 0.0
        w2d[yy2 < tan_min * xx2] = 0.0
        w2d = w2d.T.contiguous()                                      # matches wedge.T in icecream
        w2d = w2d + torch.flip(torch.flip(w2d, [0]), [1])            # flipud + fliplr + add

        # ── 3D: broadcast 2D wedge, apply spherical ball ─────────────────
        # r² computed with broadcasting — only one (S,S,S) tensor materialised
        r_sq = lin.view(1, S, 1) ** 2 + lin.view(S, 1, 1) ** 2 + lin.view(1, 1, S) ** 2
        ball = (r_sq < 1.0).float() if use_spherical else torch.ones(S, S, S, device=device)
        # w2d (S,S) → (1,S,S): broadcast along axis 0 (matches np right-align)
        wedge_3d = w2d.unsqueeze(0) * ball
        if low_support > 0.0:
            wedge_3d[r_sq < low_support] = 1.0

        # symmetrize + binarize
        mask_sym = symmetrize_3D(wedge_3d)
        wedge_3d = (wedge_3d + mask_sym) / 2.0
        wedge_3d[wedge_3d > 0.1] = 1.0
        return wedge_3d

    @property
    def tilt_key(self) -> tuple[float, float]:
        """Current (tilt_min, tilt_max) for change detection."""
        return (self._tilt_min, self._tilt_max)

    def update_parameters(self, tilt_min=None, tilt_max=None, **kwargs) -> None:
        """deepinv hook — called by ``Physics.update(**params)`` each training step.

        When the dataloader returns ``(evn, odd, {"tilt_min": t, "tilt_max": t})``,
        deepinv extracts the dict and calls this method so the wedge is rebuilt
        in-place before the loss is computed.

        With batch_size > 1 (patch training), the DataLoader collates scalar tensors
        into shape (B,). We reduce to a scalar by taking the mean across the batch
        — all patches in a batch share the same physics.
        """
        if tilt_min is not None and tilt_max is not None:
            if hasattr(tilt_min, "numel") and tilt_min.numel() > 1:
                tilt_min = tilt_min.float().mean()
                tilt_max = tilt_max.float().mean()
            self.update_angles(float(tilt_min), float(tilt_max))

    def update_angles(self, tilt_min: float, tilt_max: float) -> None:
        """Rebuild wedge masks in-place for a new tilt range.

        All losses that hold a reference to this physics object will automatically
        see the new mask on their next forward pass (they access buffers via
        properties, not cached copies). EqLoss_icecream's _valid_k_sets cache
        must be refreshed separately via loss.refresh_valid_k_sets().
        """
        if tilt_min == self._tilt_min and tilt_max == self._tilt_max:
            return  # angles unchanged — skip expensive _build_masks

        new_mask, new_mask_ref = self._build_masks(tilt_max, tilt_min, self.mask.device)
        self.mask.copy_(new_mask)
        self.mask_ref.copy_(new_mask_ref)

        self._tilt_min = float(tilt_min)
        self._tilt_max = float(tilt_max)

    # ------------------------------------------------------------------
    # deepinv LinearPhysics interface
    # ------------------------------------------------------------------

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply missing-wedge mask in Fourier space.

        Matches icecream's ``get_measurement``:
          - zero-pads x to mask shape via ``fftn(s=mask.shape)``
          - centers DC with fftshift before masking
          - crops output back to original x spatial shape

        :param torch.Tensor x: Input tensor of shape (B, C, D, H, W).
        :return: Wedge-masked volume, same shape as x.
        """
        # Remember original spatial size for output crop
        D, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
        mask_dims = tuple(self.mask.shape)  # (mask_size+1,)*3

        # Zero-pad to mask shape, shift DC to center, apply mask
        X = torch.fft.fftshift(
            torch.fft.fftn(x, s=mask_dims, dim=(-3, -2, -1)),
            dim=(-3, -2, -1),
        )
        # mask may have shape (mask_size+1,)³ — broadcast over batch/channel dims
        X_masked = X * self.mask

        # Back to real space, then crop to original volume size
        out = torch.fft.ifftn(
            torch.fft.ifftshift(X_masked, dim=(-3, -2, -1)),
            dim=(-3, -2, -1),
        ).real
        return out[..., :D, :H, :W]

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Adjoint = A (self-adjoint because mask is real and binary)."""
        return self.A(y)
