"""Named presets bundling a physics/model/loss builder triple."""
from .physics import build_physics
from .models import build_ei_model
from .losses import build_losses

PRESETS = {
    "missingwedge_ei": {
        "physics": build_physics,
        "model": build_ei_model,
        "losses": build_losses,
    },
}


def get_preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(f"Unknown method preset {name!r}. Available: {list(PRESETS)}")
    return PRESETS[name]
