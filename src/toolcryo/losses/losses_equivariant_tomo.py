"""Equivariance loss for the ``tomo_ei`` preset.

``A`` is volume->sinogram, not a frequency mask, so the rotated volume is
re-simulated through the real geometry and reconstructed: ``P = fbp o A``.

``eq_cross_coupled=True`` (default) draws **one** rotation T for both halves and
takes each half's target from the other (icecream's
``EquivariantTrainer.compute_loss``)::

    x_rot, y_rot = T(x_net), T(y_net)
    L_eq = MSE(f(P(x_rot)), y_rot) + MSE(f(P(y_rot)), x_rot)

``False`` restores the self-coupled form: a rotation per half, each its own
target.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from deepinv.loss import Loss


class EqLoss(Loss):
    """Cross-coupled equivariance loss for ``tomo_ei``.

    :param Rotate3D transform: shape-preserving rotation sampler (built with
        ``volume_shape=physics.physics_evn.volume_shape`` in run.py).
    :param float weight: loss weight (default 1.0).
    :param bool cross_coupled: ``True`` (default) shares one rotation and takes
        each half's target from the *other* half; ``False`` restores the
        self-coupled form, each half with its own rotation and its own target.
    """

    def __init__(self, transform, weight: float = 1.0,
                 cross_coupled: bool = True) -> None:
        super().__init__()
        self._transform = transform
        self.weight = weight
        self.cross_coupled = cross_coupled
        self._criteria = nn.MSELoss(reduction="mean")

    def _term(self, x_net: torch.Tensor, tomo_physics, model) -> torch.Tensor:
        """One self-coupled half: its own rotation, and itself as the target."""
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

        if not self.cross_coupled:
            return self.weight * (
                self._term(x_net, physics.physics_evn, model)
                + self._term(y_net, physics.physics_odd, model)
            )

        # One rotation for both halves — see module docstring.
        k = self._transform.get_params(x_net)["k_idx"]
        x_rot = self._transform.transform(x_net, k_idx=k)
        y_rot = self._transform.transform(y_net, k_idx=k)

        pe, po = physics.physics_evn, physics.physics_odd
        loss = (self._criteria(model(pe.fbp(pe.A(x_rot))), y_rot)
                + self._criteria(model(po.fbp(po.A(y_rot))), x_rot))
        return self.weight * loss
