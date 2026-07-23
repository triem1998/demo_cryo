import torch
from deepinv.models.base import Denoiser
from .unet3d_bf import UNet3D as UNet3D


class IceCreamUNetWrapper(Denoiser):
    """Wraps icecream's UNet3D so deepinv's model_inference(y, physics) works.

    deepinv calls model(y, physics) — the physics object would land on
    UNet3D's pos_enc argument and silently corrupt behaviour.  This wrapper
    absorbs physics (and any other deepinv kwargs) and forwards only the
    tensor to the underlying UNet.

    Both presets tile this wrapper with ``distribute(..., type_object="denoiser")``
    — missingwedge_ei on the bare wrapper, unrolled on ``prior.denoiser`` inside
    the PGD. deepinv's tiling *only* activates when the target is a
    ``deepinv.models.base.Denoiser`` instance, so this class must subclass it.
    """

    def __init__(self, unet: torch.nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor, physics=None, **kwargs) -> torch.Tensor:
        return self.unet(x)
