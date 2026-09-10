"""EI loss-list construction from a run config.

One sibling module per physics family, each holding its own Obs/Eq pair:
``losses_equivariant_wedge.py`` (FFT wedge) and ``losses_equivariant_tomo.py``
(real tomography). This is the thin construction layer.
"""
from __future__ import annotations

from ..base_config import RunEIBaseConfig
from .losses_equivariant_wedge import EqLoss, ObsLoss
from .losses_equivariant_tomo import EqLoss as TomoEqLoss, ObsLoss as TomoObsLoss

__all__ = ["build_ei_losses", "build_tomo_losses"]


def build_ei_losses(cfg: RunEIBaseConfig, physics, transform) -> list:
    return [
        ObsLoss(physics, weight=1.0,
                use_fourier=False, view_as_real=True, no_window=False),
        EqLoss(physics, transform, weight=float(cfg.eq_weight),
               use_fourier=False, view_as_real=True, eq_use_direct=False, no_window=False),
    ]


def build_tomo_losses(cfg: RunEIBaseConfig, physics=None, transform=None) -> list:
    """Obs, plus Eq when ``eq_weight > 0`` — shared by ``unrolled`` and ``tomo_ei``.

    Eq is skipped, not zero-weighted: it costs an extra pass per half.
    ``cfg.preset`` only selects how ``EqLoss`` calls the model.
    """
    losses = [TomoObsLoss(weight=1.0, gain=str(cfg.obs_gain),
                          ramp=bool(cfg.obs_ramp))]
    if float(cfg.eq_weight) > 0.0:
        losses.append(TomoEqLoss(transform, weight=float(cfg.eq_weight),
                                 unrolled=str(cfg.preset) == "unrolled",
                                 noise=float(cfg.eq_noise),
                                 scale_free=bool(cfg.eq_scale_free)))
    return losses
