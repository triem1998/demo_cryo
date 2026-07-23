"""EI loss-list construction from a run config.

Heavy loss classes live in the sibling modules (``losses.py``, ``losses_custom.py``,
``losses_unrolled.py``); this ``__init__`` is the thin construction layer, one
builder per preset.
"""
from __future__ import annotations

from ..base_config import RunEIBaseConfig
from .losses import EqLoss, ObsLoss
from .losses_custom import EqLoss as EqLossCustom, ObsLoss as ObsLossCustom
from .losses_unrolled import ObsLoss as UnrolledObsLoss

__all__ = ["build_ei_losses", "build_tomography_losses"]


def build_ei_losses(cfg: RunEIBaseConfig, physics, transform) -> list:
    if str(cfg.loss_type) == "custom":
        return [
            ObsLossCustom(physics, weight=1.0),
            EqLossCustom(physics, transform, weight=float(cfg.eq_weight)),
        ]
    return [
        ObsLoss(physics, weight=1.0,
                use_fourier=False, view_as_real=True, no_window=False),
        EqLoss(physics, transform, weight=float(cfg.eq_weight),
               use_fourier=False, view_as_real=True, eq_use_direct=False, no_window=False),
    ]


def build_tomography_losses(cfg: RunEIBaseConfig, physics=None, transform=None) -> list:
    """Obs-only cross-half-set consistency loss (no equivariance term)."""
    return [UnrolledObsLoss(weight=1.0)]
