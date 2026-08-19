"""TomographyEM — the astra-toolbox forward-projection operator.

This module is the astra backend and nothing else. Its pure-torch counterpart
is ``tomography_torch.py``; everything that builds, pairs or normalises either
of them — and the choice between them — lives in ``tomography_build.py``, the
only module that knows both.
"""
from __future__ import annotations

import torch
import deepinv as dinv


class TomographyEM(dinv.physics.LinearPhysics):
    """Real single-axis-tilt forward-projection operator for electron tomography.

    Wraps ``deepinv.physics.TomographyWithAstra`` (3D parallel-beam, astra-toolbox
    backend) to project a reconstructed volume down to its tilt series.

    Volumes are expected **already in astra's ``(n_slices, n_rows, n_cols)``
    order**, where ``n_slices`` is the rotation-invariant (tilt) axis — i.e.
    ``(Y, Z, X)`` for this dataset's MRC files, which ``load_fbp_init``
    produces directly. Reordering is done once at load time (a real numpy copy)
    rather than per call, so ``A``/``A_adjoint`` contain no ``permute``: a
    permute is a zero-copy stride relabel whose *backward* hands astra and NCCL
    non-contiguous gradients, which both reject. Same arrangement as demo_tomo,
    which passes ``TomographyWithAstra`` straight through for the same reason.

    :param tuple[int,int,int] volume_shape: ``(n_slices, n_rows, n_cols)`` shape
        of the input volume.
    :param angles_deg: 1D array/tensor of tilt angles in degrees (e.g. read from
        a ``.tlt`` file).
    :param tuple[int,int] | None detector_shape: Real detector pixel grid
        ``(V, N)``. Decoupled from ``volume_shape`` — it is a property of the
        camera, not of the reconstruction geometry. Defaults to
        ``(volume_shape[0], volume_shape[2])``.
    :param float pixel_spacing: Isotropic voxel size of the object grid
        (default 1.0 — absolute units cancel out for correlation-based
        calibration; only matters if you need physical units).
    :param float detector_spacing: Isotropic detector pixel size (default 1.0).
    :param float angle_sign: Multiplier applied to ``angles_deg`` before passing
        to astra (astra internally negates angles too) — a calibration knob for
        the rotation-direction sign convention. Default 1.0.
    :param bool normalize: Forwarded to ``TomographyWithAstra`` (default False,
        so ``A`` returns physically-meaningful line-integral units rather than a
        unit-norm-rescaled operator — needed to compare against real tilt series).
    :param str device: Must be a CUDA device (astra-toolbox backend requirement).
    """

    def __init__(
        self,
        volume_shape: tuple[int, int, int],
        angles_deg,
        detector_shape: tuple[int, int] | None = None,
        pixel_spacing: float = 1.0,
        detector_spacing: float = 1.0,
        angle_sign: float = 1.0,
        normalize: bool = False,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        if torch.device(device).type != "cuda":
            raise ValueError(
                f"TomographyEM requires a CUDA device (astra-toolbox backend), got device={device!r}."
            )
        self.volume_shape = tuple(int(s) for s in volume_shape)

        if detector_shape is None:
            detector_shape = (self.volume_shape[0], self.volume_shape[2])
        self.detector_shape = tuple(int(s) for s in detector_shape)

        angles = torch.as_tensor(angles_deg, dtype=torch.float32) * float(angle_sign)
        self.n_angles = int(angles.numel())
        # Logging only — pre-sign-flip, so it matches the .tlt file. Same
        # private names MissingWedge uses, so one log line reads either physics.
        self._tilt_min = float(torch.as_tensor(angles_deg).min())
        self._tilt_max = float(torch.as_tensor(angles_deg).max())

        self.xray = dinv.physics.TomographyWithAstra(
            img_size=self.volume_shape,
            angles=angles,
            n_detector_pixels=self.detector_shape,
            angular_range=(0, 180),  # unused: angles is an explicit tensor
            detector_spacing=detector_spacing,
            pixel_spacing=pixel_spacing,
            geometry_type="parallel",
            normalize=normalize,
            device=torch.device(device),
        )

    # ------------------------------------------------------------------
    # deepinv LinearPhysics interface — straight delegation, no axis
    # bookkeeping (see the class docstring). The ``.contiguous()`` calls are
    # free no-ops for already-contiguous inputs (they return the same object)
    # and only guard astra's hard `assert data.is_contiguous()`.
    # ------------------------------------------------------------------

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward projection: volume (B,C,*volume_shape) -> sinogram (B,C,V,A,N).

        :param torch.Tensor x: Input volume of shape (B, C, *volume_shape).
        :return: Sinogram of shape (B, C, V, A, N) — V/N are the detector grid
            (``detector_shape``), A is ``n_angles``.
        """
        return self.xray.A(x.contiguous())

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Approximate adjoint (pixel-driven back-projection): sinogram -> volume.

        :param torch.Tensor y: Sinogram of shape (B, C, V, A, N).
        :return: Volume of shape (B, C, *volume_shape).
        """
        return self.xray.A_adjoint(y.contiguous())

    def fbp(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Filtered back-projection reconstruction: sinogram -> volume.

        The sinogram is centred before filtering.  deepinv's ramp filter zero-pads
        each detector line to twice its length; cryo-ET sinograms carry a large DC
        pedestal (mean >> std), so zero-padding manufactures a step edge that the
        ramp filter amplifies into stripe artifacts of amplitude comparable to the
        signal itself.  Centring removes the step.  The volume's absolute DC level
        is not recoverable from a limited-angle tilt series anyway.

        :param torch.Tensor y: Sinogram of shape (B, C, V, A, N).
        :return: Volume of shape (B, C, *volume_shape).
        """
        return self.fbp_raw(y - y.mean(dim=(-3, -2, -1), keepdim=True))

    def fbp_raw(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """FBP without the DC centring — the piece ``ShardedTomography`` composes,
        since the centring must be done once on the *global* sinogram."""
        return self.xray.fbp(y)
