"""Denoiser model construction from a run config."""
from __future__ import annotations

import torch

_UNET_F_MAPS = 64
_UNET_NUM_LEVELS = 4
_DRUNET_NB = 4


class DRUNetWrapper(torch.nn.Module):
    """Wraps DRUNet so model(x) works — injects a fixed sigma as a float.

    DRUNet.forward(x, sigma) requires a noise level.  Passing sigma as a
    Python float uses the 3D-safe branch: torch.ones((B,1,*x.shape[2:]))*sigma,
    unlike the tensor branch which hard-codes 2D expand calls.
    """

    def __init__(self, drunet: torch.nn.Module, sigma: float = 0.0) -> None:
        super().__init__()
        self.drunet = drunet
        self.sigma = sigma

    def forward(self, x: torch.Tensor, physics=None, **kwargs) -> torch.Tensor:
        return self.drunet(x, self.sigma)


def build_ei_model(
    model_type: str,
    unet_dropout: float,
    drunet_sigma: float,
    device,
) -> tuple[torch.nn.Module, str]:
    """Build IceCreamUNetWrapper (unet) or DRUNetWrapper (drunet) on *device*."""
    import deepinv as dinv
    from ..icecream_orig.models import IceCreamUNetWrapper
    from ..icecream_orig.models.unet3d_bf import UNet3D as _IceCreamUNet3D

    if model_type == "unet":
        _inner = _IceCreamUNet3D(
            in_channels=1,
            out_channels=1,
            f_maps=_UNET_F_MAPS,
            num_levels=_UNET_NUM_LEVELS,
            layer_order="cr",
            use_bias=False,
            dropout_prob=unet_dropout,
        ).to(device)
        model = IceCreamUNetWrapper(_inner)
        info = f"unet  f_maps={_UNET_F_MAPS}  num_levels={_UNET_NUM_LEVELS}  dropout={unet_dropout}"
    elif model_type == "drunet":
        _nc = tuple(_UNET_F_MAPS * (2 ** i) for i in range(4))
        _inner = dinv.models.DRUNet(
            in_channels=1,
            out_channels=1,
            nc=_nc,
            nb=_DRUNET_NB,
            pretrained="download_2d",
            pretrained_2d_isotropic=False,
            dim=3,
        ).to(device)
        model = DRUNetWrapper(_inner, sigma=drunet_sigma)
        info = f"drunet  nc={_nc}  nb={_DRUNET_NB}  sigma={drunet_sigma}  init=pretrained_2d"
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'unet' or 'drunet'.")
    return model, info
