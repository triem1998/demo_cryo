"""Denoiser wrapper classes + model construction from a run config.

Heavy wrapper classes (``DRUNetWrapper``, ``TrainableSigmaDRUNet``) and their
``cfg -> model`` builders live together here — the denoiser *architectures*
themselves (UNet3D, DRUNet) are vendored in ``icecream_orig``/``deepinv``, not
redefined here, only wrapped.
"""
from __future__ import annotations

import torch
from deepinv.distributed import DistributedContext, distribute
from deepinv.distributed.framework import DistributedReplicatedParameters
from deepinv.models.base import Denoiser
from deepinv.optim import PGD
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP

from .physics import TomographyEMPair

_UNET_F_MAPS = 64
_UNET_NUM_LEVELS = 4
_DRUNET_NB = 4


class DRUNetWrapper(Denoiser):
    """Wraps DRUNet so ``model(x)`` works — injects a fixed sigma as a float.

    DRUNet.forward(x, sigma) requires a noise level.  Passing sigma as a
    Python float uses the 3D-safe branch: torch.ones((B,1,*x.shape[2:]))*sigma,
    unlike the tensor branch which hard-codes 2D expand calls.

    The second positional slot (``physics``) is absorbed and ignored — for
    ``missingwedge_ei`` deepinv calls ``model(x, physics)`` with a real physics
    object there; for the PGD/PnP path ``PnP.prox`` calls
    ``denoiser(x, sigma_denoiser)`` with a sigma scalar there. This wrapper
    ignores that slot entirely and always uses its own fixed ``self.sigma`` —
    see ``TrainableSigmaDRUNet`` for the variant that makes sigma trainable.

    Subclasses ``deepinv.models.base.Denoiser`` (not just ``nn.Module``): for
    ``missingwedge_ei``'s direct ``distribute(wrapper, ctx, type_object="denoiser")``
    call this makes no difference (that path accepts any ``nn.Module``), but
    for ``unrolled``'s ``distribute(pgd_model, ctx, ...)`` on the whole PGD, the
    internal denoiser-tiling step only wraps ``prior.denoiser`` when it is a
    ``Denoiser`` instance — required for that tiling to activate at all.
    """

    def __init__(self, drunet: torch.nn.Module, sigma: float = 0.0) -> None:
        super().__init__()
        self.drunet = drunet
        self.sigma = sigma

    def forward(self, x: torch.Tensor, physics=None, **kwargs) -> torch.Tensor:
        return self.drunet(x, self.sigma)


class TrainableSigmaDRUNet(Denoiser):
    """DRUNet wrapper for the PGD/PnP path that makes ``g_param`` trainable.

    ``PnP.prox`` calls ``denoiser(x, sigma_denoiser)`` positionally — inside
    the unrolled model this second slot always holds a plain sigma scalar. But
    ``DRUNetWrapper`` (shared with the ``missingwedge_ei`` preset, where that
    *same* slot receives a real ``Physics`` object instead) always ignores it
    and uses its own fixed ``self.sigma``. To make ``g_param``
    (=``sigma_denoiser``) an actually-trainable PGD parameter, this wraps the
    *raw* DRUNet directly and uses whatever sigma is passed in — used only by
    ``build_unrolled_model`` below (drunet + train_algo_params) so
    ``DRUNetWrapper``/``missingwedge_ei`` are untouched. Subclasses ``Denoiser``
    for the same reason as ``DRUNetWrapper`` above (denoiser-tiling gate).
    """

    def __init__(self, drunet: torch.nn.Module) -> None:
        super().__init__()
        self.drunet = drunet

    def forward(self, x: torch.Tensor, sigma=None, **kwargs) -> torch.Tensor:
        return self.drunet(x, sigma)


def build_ei_model(model_type, unet_dropout, drunet_sigma, device) -> tuple[torch.nn.Module, str]:
    """Build IceCreamUNetWrapper (unet) or DRUNetWrapper (drunet) on *device*."""
    import deepinv as dinv
    from .icecream_orig.models import IceCreamUNetWrapper
    from .icecream_orig.models.unet3d_bf import UNet3D as _IceCreamUNet3D

    if model_type == "unet":
        _inner = _IceCreamUNet3D(
            in_channels=1, out_channels=1, f_maps=_UNET_F_MAPS,
            num_levels=_UNET_NUM_LEVELS, layer_order="cr", use_bias=False,
            dropout_prob=unet_dropout,
        ).to(device)
        model = IceCreamUNetWrapper(_inner)
        info = f"unet  f_maps={_UNET_F_MAPS}  num_levels={_UNET_NUM_LEVELS}  dropout={unet_dropout}"
    elif model_type == "drunet":
        _nc = tuple(_UNET_F_MAPS * (2 ** i) for i in range(4))
        _inner = dinv.models.DRUNet(
            in_channels=1, out_channels=1, nc=_nc, nb=_DRUNET_NB,
            pretrained="download_2d", pretrained_2d_isotropic=False, dim=3,
        ).to(device)
        model = DRUNetWrapper(_inner, sigma=drunet_sigma)
        info = f"drunet  nc={_nc}  nb={_DRUNET_NB}  sigma={drunet_sigma}  init=pretrained_2d"
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'unet' or 'drunet'.")
    return model, info


def _kernels_native_to_astra(state: dict) -> dict:
    """Re-express a denoiser trained in native ``(Y, X, Z)`` for astra ``(Y, Z, X)``.

    ``missingwedge_ei``/patch train on native-order volumes, but the PGD
    iterate — and therefore the PnP denoiser inside it — is in astra order
    (``TomographyEM`` loads volumes in astra order and never permutes inside
    ``A``/``A_adjoint``). Feeding those weights a transposed volume is not
    merely suboptimal: the missing wedge is strongly anisotropic along Z, so a
    prior trained to repair it becomes actively wrong, which in a PGD loop
    diverges to inf/nan gradients.

    For a network built only from convolutions, pointwise nonlinearities and
    isotropic pooling, permuting the *kernels'* spatial axes is exactly
    equivalent to permuting the volume: ``P(D(v)) == D'(P(v))``. The volume swap
    is spatial axes 1<->2, i.e. dims 3 and 4 of a ``(out, in, kD, kH, kW)``
    weight. Applied once at load, so nothing enters the autograd graph.

    The equivalence needs every conv to be symmetric across the two swapped
    axes; asymmetric kernels/strides would silently break it, so that is
    checked rather than assumed.
    """
    out = {}
    for k, v in state.items():
        if v.ndim != 5:
            out[k] = v
            continue
        if v.shape[3] != v.shape[4]:
            raise ValueError(
                f"Cannot convert pretrained denoiser to astra axis order: weight {k!r} has "
                f"shape {tuple(v.shape)}, which is asymmetric across the swapped spatial axes "
                f"(dims 3 and 4). The kernel-permute equivalence only holds for isotropic "
                f"convolutions — this architecture needs a different conversion."
            )
        out[k] = v.transpose(3, 4).contiguous()
    return out


def build_unrolled_model(cfg, physics: TomographyEMPair, ctx: DistributedContext) -> tuple:
    """Build a distributed PGD-unfold model with a PnP(denoiser) prior.

    Reuses ``build_ei_model`` as the prior. ``cfg.train_algo_params`` (default
    ``True``) makes ``stepsize`` (+ ``g_param`` for drunet only, via
    ``TrainableSigmaDRUNet``; inert for unet, which has no sigma input)
    learned jointly with the denoiser. ``cfg.init_stepsize`` is used directly:
    the operators are built with ``normalize=True`` (unit spectral norm), so
    there is no operator-norm division and nothing to rescale per tomogram.

    The denoiser inside ``PnP`` is tiled across ranks (``distribute(...,
    type_object="denoiser")``); the physics itself is not distributed — each
    rank runs the full operator.

    ``cfg.pretrained_ckpt`` accepts two different checkpoint shapes,
    auto-detected from their top-level keys:
      - a bare-denoiser checkpoint (e.g. from ``missingwedge_ei``, or a
        hand-extracted denoiser) — loaded onto the denoiser alone, before the
        drunet adapter swap and before ``PGD``/``distribute()`` exist.
      - a full ``unrolled`` checkpoint (keys start with ``"prior."`` or
        ``"params_algo."``) — loaded onto the fully-built distributed model
        *after* ``distribute()``, restoring the denoiser weights *and*
        ``stepsize``/``g_param`` together, since it's saved from that exact
        shape. Requires ``n_iter``/``model_type``/``train_algo_params`` to
        match the checkpoint's own run (those determine state_dict shape);
        ``patch_size``/``overlap``/``checkpoint_batches`` are tiling-only and
        free to differ.
    """
    denoiser, info = build_ei_model(cfg.model_type, cfg.unet_dropout, cfg.drunet_sigma, ctx.device)

    full_ckpt_state = None
    if cfg.pretrained_ckpt is not None:
        ckpt = torch.load(cfg.pretrained_ckpt, map_location=ctx.device, weights_only=True)
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        is_full_unrolled_ckpt = any(
            k.startswith("prior.") or k.startswith("params_algo.") for k in state
        )
        if is_full_unrolled_ckpt:
            # Denoiser + stepsize (+ g_param) — deferred until the model is
            # fully built below, since that's the shape this was saved from.
            full_ckpt_state = state
        else:
            # Bare-denoiser checkpoint: same IceCreamUNetWrapper/DRUNetWrapper
            # class missingwedge_ei uses, at this point (before the drunet
            # adapter swap below) — loads directly.
            if any(k.startswith("processor.") for k in state):
                state = {k.removeprefix("processor."): v for k, v in state.items()}
            state = _kernels_native_to_astra(state)
            denoiser.load_state_dict(state, strict=True)
            if ctx.rank == 0:
                print(f"[unrolled] loaded pretrained denoiser weights from {cfg.pretrained_ckpt} "
                      f"(bare denoiser -> assumed missingwedge_ei/patch native (Y,X,Z) order; "
                      f"conv kernels permuted to this preset's astra (Y,Z,X) order)", flush=True)

    is_drunet = cfg.model_type == "drunet"
    train_algo = bool(cfg.train_algo_params)
    if train_algo and is_drunet:
        denoiser = TrainableSigmaDRUNet(denoiser.drunet)
    trainable_params = (["stepsize"] + (["g_param"] if is_drunet else [])) if train_algo else []

    n_iter = int(cfg.n_iter)
    model = PGD(
        stepsize=[float(cfg.init_stepsize)] * n_iter,
        sigma_denoiser=float(cfg.drunet_sigma),
        beta=[1.0] * n_iter,
        trainable_params=trainable_params,
        data_fidelity=L2(),
        max_iter=n_iter,
        prior=PnP(denoiser=denoiser),
        unfold=True,
    )
    # Denoiser-only distribution: the PGD keeps its plain L2 + plain physics, so
    # each rank runs the full projection locally (no per-iteration physics
    # all-reduce), while the denoiser is tiled for memory. Trainable
    # algo params (stepsize/g_param) get a cross-rank grad-sync, since the tiled
    # denoiser makes each rank's contribution differ. Same pieces
    # _distribute_base_optim does, minus the data-fidelity distribution.
    model = model.to(ctx.device)
    model.prior[0].denoiser = distribute(
        model.prior[0].denoiser, ctx, type_object="denoiser",
        patch_size=tuple(int(v) for v in cfg.patch_size),
        overlap=tuple(int(v) for v in cfg.overlap),
        tiling_dims=(-3, -2, -1),
        max_batch_size=cfg.max_batch_size,
        checkpoint_batches=cfg.checkpoint_batches,
    )
    algo_params = [p for v in model.params_algo.values()
                   if isinstance(v, torch.nn.ParameterList) for p in v]
    if algo_params:
        model._deepinv_dist_sync = DistributedReplicatedParameters(ctx, algo_params, average=True)

    if full_ckpt_state is not None:
        try:
            model.load_state_dict(full_ckpt_state, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load full unrolled checkpoint {cfg.pretrained_ckpt!r} into the "
                f"current model — its state_dict shape depends on n_iter/model_type/"
                f"train_algo_params matching the checkpoint's own run config. "
                f"Original error: {e}"
            ) from e
        if ctx.rank == 0:
            print(f"[unrolled] loaded full unrolled checkpoint (denoiser+stepsize"
                  f"{'+g_param' if is_drunet else ''}) from {cfg.pretrained_ckpt}", flush=True)

    return model, (f"unrolled(PGD, physics local + denoiser tiled, "
                   f"trainable_params={trainable_params}) "
                   f"n_iter={n_iter} stepsize={cfg.init_stepsize} prior={info}")


def clamp_stepsize(model, eps: float = 1e-8) -> None:
    """Keep a trainable stepsize positive after each optimizer step (no-op when
    stepsize isn't trainable). Wired as ``_post_optimizer_step`` for the
    unrolled preset (registry.py); harmless for missingwedge_ei.
    """
    stepsize = model.params_algo["stepsize"]
    if isinstance(stepsize, torch.nn.ParameterList):
        with torch.no_grad():
            for s in stepsize:
                s.data.clamp_(min=eps)
