"""Named presets bundling a physics/model/loss builder triple.

Pure registration — every builder is a thin construction function living
beside its heavy classes (physics/__init__.py, models.py, losses/__init__.py)
or a small standalone glue function (forward.py). Nothing is defined here.
"""
from .physics import build_missingwedge_physics, build_tomography_physics
from .models import build_ei_model, build_unrolled_model, clamp_stepsize
from .losses import build_ei_losses, build_tomography_losses
from .forward import ei_denoiser_forward, unrolled_forward
from .utils.utils import half_set_recon, unrolled_recon

PRESETS = {
    "missingwedge_ei": {
        "physics": build_missingwedge_physics,
        "model": build_ei_model,
        "losses": build_ei_losses,
        "forward": ei_denoiser_forward,
        "post_optimizer_step": lambda model: None,
        "recon": half_set_recon,
    },
    # Non-uniform call signatures vs missingwedge_ei (physics/model take the
    # TomographyEMPair container instead of crop_size) — run_full branches on
    # preset name for the actual build calls.
    "unrolled": {
        "physics": build_tomography_physics,
        "model": build_unrolled_model,
        "losses": build_tomography_losses,
        "forward": unrolled_forward,
        "post_optimizer_step": clamp_stepsize,
        "recon": unrolled_recon,
    },
}


def get_preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(f"Unknown method preset {name!r}. Available: {list(PRESETS)}")
    return PRESETS[name]
