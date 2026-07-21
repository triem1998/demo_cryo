"""EI loss-list construction from a run config."""
from __future__ import annotations

from ..base_config import RunEIBaseConfig
from ..losses.losses import EqLoss, ObsLoss
from ..losses.losses_custom import EqLoss as EqLossCustom, ObsLoss as ObsLossCustom


def build_losses(cfg: RunEIBaseConfig, physics, transform) -> list:
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
