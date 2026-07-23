"""TomographyEM / TomographyEMPair — real tomography physics for the
tomography-domain methods (unrolled today, reusable by future
tomography-domain presets).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import deepinv as dinv

from ..utils.utils import load_mrc_volume


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
        return self.xray.fbp(y - y.mean(dim=(-3, -2, -1), keepdim=True))


# ---------------------------------------------------------------------------
# A tomogram's two half-set (EVN/ODD) TomographyEM operators — split1/split2
# use different interleaved tilt angles, so each half gets its own operator —
# bundled with their FBP-init volumes. Method-agnostic physics: consumed by
# the unrolled preset today, reusable by future tomography-domain presets.
# ---------------------------------------------------------------------------

# Calibrated for this dataset's acquisition convention (see
# scripts/test_tomography_em.py) — the tilt sign matches IMOD's convention
# negated. The tilt axis is Y, which load_fbp_init puts first.
_TOMO_ANGLE_SIGN = -1.0


@dataclass
class TomographyEMPair:
    """A tomogram's two half-set (EVN/ODD) TomographyEM operators + FBP inits.

    Each rank runs the full operator locally — the tilt angles are *not* sharded
    across ranks. Only the denoiser is tiled (models.py::build_unrolled_model),
    so no per-iteration physics collective is needed.
    """
    physics_evn: "TomographyEM"
    physics_odd: "TomographyEM"
    init_evn: torch.Tensor
    init_odd: torch.Tensor
    # Full (train + val, concatenated) path lists + build params, kept so
    # ``update()`` can lazily rebuild physics_evn/odd for whichever tomogram
    # the current batch is — see CryoEIFullDataset.index_offset.
    evn_paths: list
    odd_paths: list
    device: object
    target_shape: tuple | None
    _tomo_idx: int = 0

    def update(self, tomo_idx=None, **kwargs) -> None:
        """Rebuild physics_evn/odd + their FBP inits for a different tomogram.

        ``deepinv.Trainer`` calls ``physics.update(**params)`` before every
        forward pass (train and eval) — mirrors how ``MissingWedge`` rebuilds
        its wedge mask per volume (dataset_full.py), except a TomographyEM
        operator is calibrated to one tomogram's actual tilt angles, so a
        "rebuild" here means re-reading that tomogram's .tlt file + FBP
        volume, not just recomputing a mask from two floats.
        """
        if tomo_idx is None:
            return
        if hasattr(tomo_idx, "numel"):
            tomo_idx = int(tomo_idx.flatten()[0].item())
        if tomo_idx == self._tomo_idx:
            return
        evn_path, odd_path = self.evn_paths[tomo_idx], self.odd_paths[tomo_idx]
        self.physics_evn, self.init_evn = build_one_tomography_em(
            evn_path.parent, "split1", evn_path, self.device, self.target_shape)
        self.physics_odd, self.init_odd = build_one_tomography_em(
            odd_path.parent, "split2", odd_path, self.device, self.target_shape)
        self._tomo_idx = tomo_idx


def load_fbp_init(
    path: Path, device, target_shape: tuple[int, int, int] | None,
) -> torch.Tensor:
    """Load a precomputed FBP volume as the PGD iteration's ``x_init``.

    Physics rather than dataset code on purpose: this is set up once alongside
    the operator, not fetched per batch, and the z-normalisation below is what
    puts ``A(x)`` on the same scale as the z-normalised sinogram. The generic
    "read an MRC and reorient it" step is shared — ``utils.utils.load_mrc_volume``.
    """
    # (Y, Z, X), no crop — the reorder is a real copy done once here rather than
    # as a permute inside A()/A_adjoint(); see the TomographyEM docstring.
    vol_np = load_mrc_volume(path, order="astra")
    vol = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0)  # (1, 1, Y, Z, X)
    if target_shape is not None:
        # Local-testing only — pure downsample, no crop, kept consistent with
        # CryoEIFullDataset._resample_sinogram's matching resample of the tilt
        # series' (ny, nx).
        # target_shape is given in the canonical (Y, X, Z) order that every
        # preset's config uses; permute it to this operator's astra (Y, Z, X).
        ty, tx, tz = (int(s) for s in target_shape)
        vol = torch.nn.functional.interpolate(
            vol, size=(ty, tz, tx), mode="trilinear", align_corners=False,
        )
    # Z-normalize (centre + unit std), matching CryoEIFullDataset._load_and_prepare.
    # Centring matters as much as scaling: a volume with a non-zero mean projects
    # to a constant offset in A(x) that can never match a centred sinogram.
    return ((vol - vol.mean()) / (vol.std() + 1e-8)).to(device)


def build_one_tomography_em(
    tomo_dir: Path, split: str, vol_path: Path, device,
    target_shape: tuple[int, int, int] | None,
) -> tuple["TomographyEM", torch.Tensor]:
    ang_matches = sorted(tomo_dir.glob(f"angles_*_{split}.tlt"))
    if not ang_matches:
        raise FileNotFoundError(f"build_unrolled_physics: no angles_*_{split}.tlt in {tomo_dir}")
    angles = np.loadtxt(str(ang_matches[0]))

    init = load_fbp_init(vol_path, device, target_shape)
    volume_shape = tuple(init.shape[-3:])

    physics = TomographyEM(
        volume_shape=volume_shape,
        angles_deg=angles,
        angle_sign=_TOMO_ANGLE_SIGN,
        normalize=True,
        device=str(device),
    )
    return physics, init
