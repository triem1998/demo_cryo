"""Forward-pass strategies: how (x_net, y_net) are computed from a batch + physics."""

from .physics import split_sinogram


def ei_denoiser_forward(trainer, x, y, physics, train):
    """Default EI double-pass: f(EVN) and f(ODD) independently."""
    x_net = trainer.model_inference(y=x, physics=physics, x=y, train=train)
    y_net = trainer.model_inference(y=y, physics=physics, x=x, train=train)
    return x_net, y_net


def tomo_ei_forward(trainer, x, y, physics, train):
    """True-physics EI: denoise each half's FBP volume directly (no unfolding).

    x/y are the EVN/ODD real sinograms (used by the Obs loss, not here);
    physics is the TomographyEMPair container — its FBP-init volumes are the
    denoiser's input, matching missingwedge_ei's ei_denoiser_forward but with
    the wedge-crop replaced by a real FBP reconstruction per half.
    """
    x_net = trainer.model_inference(y=physics.init_evn, physics=physics.physics_evn, train=train)
    y_net = trainer.model_inference(y=physics.init_odd, physics=physics.physics_odd, train=train)
    return x_net, y_net


def unrolled_forward(trainer, x, y, physics, train):
    """Unrolled reconstruction from real measurements.

    x/y are the EVN/ODD sinograms; physics is the TomographyEMPair container
    (separate operators + FBP inits for each half, see physics/__init__.py).
    The PGD uses a plain L2 + the raw operator, so the sinogram is passed
    straight through. When the angles are sharded it must be split to match — a
    distributed operator consumes one measurement per shard, not one whole
    sinogram (see physics/tomography_build.py::split_sinogram). Both paths end
    up unit spectral norm, so the stepsize is the same either way.
    """
    if physics.num_operators is not None:
        x = split_sinogram(x, physics.num_operators)
        y = split_sinogram(y, physics.num_operators)
    x_net = trainer.model_inference(
        y=x, physics=physics.physics_evn, init=physics.init_evn, train=train)
    y_net = trainer.model_inference(
        y=y, physics=physics.physics_odd, init=physics.init_odd, train=train)
    return x_net, y_net
