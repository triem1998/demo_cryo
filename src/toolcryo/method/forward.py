"""Forward-pass strategies: how (x_net, y_net) are computed from a batch + physics."""


def ei_denoiser_forward(trainer, x, y, physics, train):
    """Default EI double-pass: f(EVN) and f(ODD) independently."""
    x_net = trainer.model_inference(y=x, physics=physics, x=y, train=train)
    y_net = trainer.model_inference(y=y, physics=physics, x=x, train=train)
    return x_net, y_net
