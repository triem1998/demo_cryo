"""losses_equivariant_tomo.py — equivariance loss for the ``tomo_ei`` preset
(true tomography physics).

For a randomly sampled shape-preserving rotation T (see
``Rotate3D(volume_shape=...)`` — the astra volume is not a cube, so only a
subset of the 40-element cubic group applies):

    x_rot = T(x_net)
    L_eq += MSE( f( fbp( A(x_rot) ) ),  x_rot )

and symmetrically for the ODD branch. Unlike ``losses_equivariant_wedge.py``'s
Fourier-wedge ``EqLoss``, the wedge here cannot be rotated directly in frequency space —
``A`` is volume→sinogram, not a frequency mask — so the rotated volume is
re-simulated through the real geometry and reconstructed via ``fbp`` instead.
This is also what makes the term physically meaningful: it imprints the
*rotated* missing-angle pattern from the actual acquisition geometry, not an
idealised wedge.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from deepinv.loss import Loss


class EqLoss(Loss):
    """Equivariance loss for ``tomo_ei``.

    :param Rotate3D transform: shape-preserving rotation sampler (built with
        ``volume_shape=physics.physics_evn.volume_shape`` in run.py).
    :param float weight: loss weight (default 1.0).
    """

    def __init__(self, transform, weight: float = 1.0) -> None:
        super().__init__()
        self._transform = transform
        self.weight = weight
        self._criteria = nn.MSELoss(reduction="mean")

    def _term(self, x_net: torch.Tensor, tomo_physics, model) -> torch.Tensor:
        params = self._transform.get_params(x_net)
        x_rot = self._transform.transform(x_net, **params)
        recon = model(tomo_physics.fbp(tomo_physics.A(x_rot)))
        return self._criteria(recon, x_rot)

    def forward(
        self,
        x_net: torch.Tensor,    # reconstruction from EVN, pre-computed by forward_pass
        physics,                # TomographyEMPair container (physics/__init__.py)
        model: nn.Module,
        **kwargs,
    ) -> torch.Tensor:
        y_net = kwargs["y_net"]  # reconstruction from ODD, pre-computed by forward_pass
        loss = (
            self._term(x_net, physics.physics_evn, model)
            + self._term(y_net, physics.physics_odd, model)
        )
        return self.weight * loss
