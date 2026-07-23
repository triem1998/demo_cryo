"""Forward-pass strategies: how (x_net, y_net) are computed from a batch + physics."""


def ei_denoiser_forward(trainer, x, y, physics, train):
    """Default EI double-pass: f(EVN) and f(ODD) independently."""
    x_net = trainer.model_inference(y=x, physics=physics, x=y, train=train)
    y_net = trainer.model_inference(y=y, physics=physics, x=x, train=train)
    return x_net, y_net


def unrolled_forward(trainer, x, y, physics, train):
    """Unrolled reconstruction from real measurements.

    x/y are the EVN/ODD sinograms; physics is the TomographyEMPair container
    (separate operators + FBP inits for each half, see physics/__init__.py).
    The PGD uses a plain L2 + the full raw operator, so the sinogram is passed
    straight through — each rank runs the whole projection locally.

    No per-tomogram stepsize rescaling: the operators are built with
    ``normalize=True`` (unit spectral norm), so every tomogram's operator has
    the same norm and the stepsize is valid throughout.
    """
    x_net = trainer.model_inference(
        y=x, physics=physics.physics_evn, init=physics.init_evn, train=train)
    y_net = trainer.model_inference(
        y=y, physics=physics.physics_odd, init=physics.init_odd, train=train)
    return x_net, y_net
