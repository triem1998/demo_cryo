"""EI loss-list construction from a run config.

Heavy loss classes live in the sibling modules (``losses_equivariant_wedge.py``,
``losses_unrolled.py``, ``losses_equivariant_tomo.py``); this ``__init__`` is
the thin construction layer, one builder per preset.
"""
from __future__ import annotations

from ..base_config import RunEIBaseConfig
from .losses_equivariant_wedge import EqLoss, ObsLoss
from .losses_unrolled import ObsLoss as UnrolledObsLoss
from .losses_equivariant_tomo import EqLoss as TomoEqLoss

__all__ = ["build_ei_losses", "build_tomography_losses", "build_tomo_ei_losses"]


def build_ei_losses(cfg: RunEIBaseConfig, physics, transform) -> list:
    return [
        ObsLoss(physics, weight=1.0,
                use_fourier=False, view_as_real=True, no_window=False),
        EqLoss(physics, transform, weight=float(cfg.eq_weight),
               use_fourier=False, view_as_real=True, eq_use_direct=False, no_window=False),
    ]


def build_tomography_losses(cfg: RunEIBaseConfig, physics=None, transform=None) -> list:
    """Obs-only cross-half-set consistency loss (no equivariance term)."""
    return [UnrolledObsLoss(weight=1.0)]


def build_tomo_ei_losses(cfg: RunEIBaseConfig, physics=None, transform=None) -> list:
    """True-physics EI: Obs (reused from the unrolled preset) + optional Eq.

    Eq is skipped entirely (not just zero-weighted) when eq_weight<=0 — each
    Eq term costs a full A + fbp + denoiser pass per half, unlike the cheap
    FFT-based EqLoss in build_ei_losses.
    """
    losses = [UnrolledObsLoss(weight=1.0)]
    if float(cfg.eq_weight) > 0.0:
        losses.append(TomoEqLoss(transform, weight=float(cfg.eq_weight)))
    return losses
