"""losses_unrolled.py — self-supervised data-fidelity loss for the unrolled preset.

Cross half-set consistency in the measurement (sinogram) domain, mirroring
losses_equivariant_wedge.py's ObsLoss structure but using the real TomographyEM operators
(each half-set's own operator, since split1/split2 use different interleaved
tilt angles) instead of a synthetic Fourier wedge mask:

    L = MSE(physics_odd.A(f(x)), y) + MSE(physics_evn.A(f(y)), x)

where x/y are the EVN/ODD real sinograms and f(x)/f(y) are the unrolled
model's reconstructions from each. No equivariance term (v1 scope) and no
cropping — everything operates at native resolution.

Also reused as-is by ``tomo_ei`` (build_tomo_ei_losses) — same cross
half-set data-fidelity structure applies whether the reconstruction comes
from PGD-unfolding or a plain denoiser. ``tomo_ei``'s equivariance term
lives separately in ``losses_equivariant_tomo.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from deepinv.loss import Loss


def _as_sinogram(pred) -> torch.Tensor:
    """Reassemble a sharded ``A(x)`` back into one ``(B, C, V, A, N)`` sinogram.

    With ``num_operators`` set, the physics is a *stack* of per-angle-subset
    operators, so ``A`` returns one measurement per shard (a ``TensorList``)
    rather than a tensor. The shards are contiguous and in ascending angle
    order, so concatenating on the angle axis rebuilds exactly the sinogram the
    unsharded operator would have produced — which keeps this loss numerically
    identical whatever ``num_operators`` is set to. Pass-through otherwise.
    """
    return pred if torch.is_tensor(pred) else torch.cat(list(pred), dim=3)


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
            self._criteria(_as_sinogram(physics.physics_odd.A(x_net)), y)
            + self._criteria(_as_sinogram(physics.physics_evn.A(y_net)), x)
        )
        return self.weight * loss
