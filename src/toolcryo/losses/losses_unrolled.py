"""losses_unrolled.py — self-supervised data-fidelity loss for the unrolled preset.

Cross half-set consistency in the measurement (sinogram) domain, mirroring
losses.py's ObsLoss structure but using the real TomographyEM operators
(each half-set's own operator, since split1/split2 use different interleaved
tilt angles) instead of a synthetic Fourier wedge mask:

    L = MSE(physics_odd.A(f(x)), y) + MSE(physics_evn.A(f(y)), x)

where x/y are the EVN/ODD real sinograms and f(x)/f(y) are the unrolled
model's reconstructions from each. No equivariance term (v1 scope) and no
cropping — everything operates at native resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from deepinv.loss import Loss


class ObsLoss(Loss):
    """Cross half-set data-fidelity loss in the measurement domain.

    :param float weight: loss weight (default 1.0).
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight
        self._criteria = nn.MSELoss(reduction="mean")

    def forward(
        self,
        x: torch.Tensor,        # EVN sinogram
        y: torch.Tensor,        # ODD sinogram
        x_net: torch.Tensor,    # reconstruction from x, pre-computed by forward_pass
        physics,                # TomographyEMPair container (physics/__init__.py)
        **kwargs,
    ) -> torch.Tensor:
        y_net = kwargs["y_net"]  # reconstruction from y, pre-computed by forward_pass
        loss = (
            self._criteria(physics.physics_odd.A(x_net), y)
            + self._criteria(physics.physics_evn.A(y_net), x)
        )
        return self.weight * loss
