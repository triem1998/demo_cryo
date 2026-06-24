import torch
from .unet3d_bf import UNet3D as UNet3D


class IceCreamUNetWrapper(torch.nn.Module):
    """Wraps icecream's UNet3D so deepinv's model_inference(y, physics) works.

    deepinv calls model(y, physics) — the physics object would land on
    UNet3D's pos_enc argument and silently corrupt behaviour.  This wrapper
    absorbs physics (and any other deepinv kwargs) and forwards only the
    tensor to the underlying UNet.

    Also used with deepinv's distribute(): pass type_object="denoiser" so
    distribute accepts a non-Denoiser nn.Module, then wrap the backbone with
    this class so each tiled patch call ignores the physics arg forwarded
    by DistributedProcessing._apply_op.
    """

    def __init__(self, unet: torch.nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor, physics=None, **kwargs) -> torch.Tensor:
        return self.unet(x)
