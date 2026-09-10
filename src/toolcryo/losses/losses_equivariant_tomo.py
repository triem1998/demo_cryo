"""Loss terms for the real-tomography presets (``unrolled``, ``tomo_ei``).

``A`` is volume -> sinogram, not a frequency mask — the counterpart to
``losses_equivariant_wedge.py`` for ``missingwedge_ei``.

``ObsLoss``, cross half-set data fidelity. The halves are a *dose* split (same
angles, independent noise), so this is Noise2Noise in the measurement domain::

    L = MSE(c_odd * physics_odd.A(f(x)), y) + MSE(c_evn * physics_evn.A(f(y)), x)

``obs_gain`` rescales ``a = A(x_net)`` onto ``y``'s scale — they are normalised
independently::

    none                  c = 1
    znorm                 z-norm both operands, in-graph
    leastsq_xnet          <a, y> / <a, a>, refit each step; leaves amplitude
                          free, so the model can shrink
    leastsq_xnet_frozen   fit once per tomogram, then held

``obs_ramp`` weights the residual by ``|k|`` across the detector, so fine detail
counts more than coarse. Without it blur is cheap. Changes the loss scale.

``EqLoss``, equivariance under cube-symmetry rotations, re-simulated through the
real geometry. One rotation is shared by both halves, and each half's target is
the *other* half's reconstruction, so the target's noise is independent::

    L_eq = MSE(f(P(x_rot)), y_rot) + MSE(f(P(y_rot)), x_rot)

``eq_scale_free`` z-norms both MSE operands (Eq scores shape, Obs owns
amplitude). ``eq_noise`` noises ``A(T x_net)`` before reconstruction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from deepinv.loss import Loss
from deepinv.utils.tensorlist import TensorList


def _ramp_half(t: torch.Tensor) -> torch.Tensor:
    """Multiply by ``sqrt(|k|)`` along the detector axis; the caller squares.

    ``w`` has unit mean square (a white residual keeps its scale), ``w(0) = 0``.
    """
    n = t.shape[-1]
    w = torch.fft.rfftfreq(n, device=t.device, dtype=torch.float32).abs().sqrt()
    w = w / w.pow(2).mean().sqrt().clamp_min(1e-12)
    return torch.fft.irfft(torch.fft.rfft(t.float(), dim=-1) * w, n=n, dim=-1).to(t.dtype)


def _as_sinogram(pred) -> torch.Tensor:
    """Concatenate a sharded ``A(x)`` (one ``TensorList`` entry per angle subset)
    into one ``(B, C, V, A, N)`` sinogram; pass-through otherwise. Shards are
    contiguous and ascending, so the result is identical to the unsharded one.
    """
    return pred if torch.is_tensor(pred) else torch.cat(list(pred), dim=3)


class ObsLoss(Loss):
    """Cross half-set data-fidelity loss.

    :param float weight: loss weight (default 1.0).
    :param str gain: scale calibration, one of :attr:`GAINS` — see :meth:`_gains`.
    :param bool ramp: weight the residual by ``|k|`` across the detector axis,
        see :func:`_ramp_half`.
    """

    #: Accepted ``obs_gain`` values.
    GAINS = ("none", "znorm", "leastsq_xnet", "leastsq_xnet_frozen")

    def __init__(self, weight: float = 1.0, gain: str = "none",
                 ramp: bool = False) -> None:
        super().__init__()
        if gain not in self.GAINS:
            raise ValueError(f"obs_gain must be one of {self.GAINS}, got {gain!r}.")
        self.weight = weight
        self.gain = gain
        self.ramp = ramp
        self._gain_cache = None     # (init_evn, init_odd, c_odd, c_evn)

    def _gains(self, physics, x, y, a_odd_net, a_evn_net):
        """Least-squares ``c`` per half, fitted to ``a = A(x_net)`` under
        ``no_grad``. ``leastsq_xnet_frozen`` caches it on the ``init_*`` tensors,
        which ``TomographyEMPair.update()`` rebinds per tomogram.
        """
        if self.gain == "none":
            return 1.0, 1.0
        c = self._gain_cache
        if (c is not None and self.gain == "leastsq_xnet_frozen"
                and c[0] is physics.init_evn and c[1] is physics.init_odd):
            return c[2], c[3]
        with torch.no_grad():
            c_odd = (a_odd_net * y).sum() / ((a_odd_net * a_odd_net).sum() + 1e-8)
            c_evn = (a_evn_net * x).sum() / ((a_evn_net * a_evn_net).sum() + 1e-8)
        if self.gain == "leastsq_xnet_frozen":
            self._gain_cache = (physics.init_evn, physics.init_odd, c_odd, c_evn)
        return c_odd, c_evn

    def forward(
        self,
        x: torch.Tensor,        # EVN sinogram
        y: torch.Tensor,        # ODD sinogram
        x_net: torch.Tensor,    # reconstruction from x, pre-computed by forward_pass
        physics,                # TomographyEMPair container (physics/__init__.py)
        **kwargs,
    ) -> torch.Tensor:
        y_net = kwargs["y_net"]  # reconstruction from y, pre-computed by forward_pass
        pe, po = physics.physics_evn, physics.physics_odd
        # Projected first so the gains reuse these rather than re-running A.
        a_odd = _as_sinogram(po.A(x_net))
        a_evn = _as_sinogram(pe.A(y_net))
        if self.gain == "znorm":
            # In-graph: c is not the least-squares optimum here, so detaching
            # would change the gradient.
            zn = lambda t: (t - t.mean()) / (t.std() + 1e-8)   # noqa: E731
            r_odd, r_evn = zn(a_odd) - zn(y), zn(a_evn) - zn(x)
        else:
            c_odd, c_evn = self._gains(physics, x, y, a_odd, a_evn)
            r_odd, r_evn = c_odd * a_odd - y, c_evn * a_evn - x
        if self.ramp:
            r_odd, r_evn = _ramp_half(r_odd), _ramp_half(r_evn)
        return self.weight * ((r_odd ** 2).mean() + (r_evn ** 2).mean())


class EqLoss(Loss):
    """Equivariance loss for ``tomo_ei`` and ``unrolled``.

    :param Rotate3D transform: shape-preserving rotation sampler (built with
        ``volume_shape=physics.physics_evn.volume_shape`` in run.py).
    :param float weight: loss weight (default 1.0).
    :param bool unrolled: ``f`` is the measurement-conditioned PGD net rather
        than a plain denoiser — see :meth:`_recon`.
    :param float noise: multiple of the measured half-set noise level to add to
        the simulated measurement; ``0`` (default) leaves it clean.
    :param bool scale_free: z-normalise both MSE operands — see :meth:`_mse`.
    """

    #: Reduced over batch/channel and both detector axes, keeping the tilt-angle
    #: axis of a ``(B, C, V, A, N)`` sinogram.
    _PER_ANGLE = (0, 1, 2, 4)

    def __init__(self, transform, weight: float = 1.0,
                 unrolled: bool = False,
                 noise: float = 0.0, scale_free: bool = False) -> None:
        super().__init__()
        self._transform = transform
        self.weight = weight
        self.unrolled = unrolled
        self.noise = noise
        self.scale_free = scale_free
        self._criteria = nn.MSELoss(reduction="mean")

    def _noise_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Noise-to-signal ratio per tilt angle, from the half-set difference.

        ``x - y`` cancels the object, leaving variance ``2 sigma^2``; subtracting
        that from ``var(y)`` gives the clean signal variance. Dimensionless, so
        it transfers to ``y_sim``'s scale. Per angle: high tilts are noisier.
        """
        var_n = (x - y).var(dim=self._PER_ANGLE, keepdim=True) / 2.0
        var_s = (y.var(dim=self._PER_ANGLE, keepdim=True) - var_n).clamp_min(1e-12)
        return (var_n / var_s).sqrt()

    def _add_noise(self, y_sim, ratio: torch.Tensor):
        """``y_sim + eps``, with ``eps`` scaled to ``y_sim``'s own per-angle std.

        Scaling by ``y_sim``'s std is load-bearing: a fixed sigma would let a
        louder model face less relative noise. Sharded ``A`` returns contiguous
        ascending angle ranges, so each shard slices ``ratio`` by its own count.
        """
        def _one(t, r):
            return t + self.noise * r * t.std(dim=self._PER_ANGLE, keepdim=True) * torch.randn_like(t)

        if torch.is_tensor(y_sim):
            return _one(y_sim, ratio)
        out, a0 = [], 0
        for part in y_sim:
            a1 = a0 + part.shape[3]
            out.append(_one(part, ratio[..., a0:a1, :]))
            a0 = a1
        return TensorList(out)

    def _mse(self, est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE, scale-free when ``scale_free``. Checkpointed: z-norm otherwise
        pins 8 full volumes (17.2 GB), recomputed for ~0.15% of a step. Pure —
        ``eq_noise`` is drawn upstream in :meth:`_recon`."""
        if not self.scale_free:
            return self._criteria(est, target)
        return checkpoint(self._mse_scale_free, est, target, use_reentrant=False)

    def _mse_scale_free(self, est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Statistics in-graph: a detached z-norm is value-invariant, not
        gradient-invariant, so it would leave the shrink channel open."""
        zn = lambda t: (t - t.mean()) / (t.std() + 1e-8)   # noqa: E731
        return self._criteria(zn(est), zn(target))

    def _recon(self, v_rot: torch.Tensor, tomo_physics, model, ratio) -> torch.Tensor:
        """``f`` applied to a measurement simulated from ``v_rot``.

        ``tomo_ei``'s denoiser takes the FBP volume; the unrolled net also gets
        the sinogram, with an init built from it by the same ``fbp`` used at
        inference. Noise goes in before ``fbp``, for the same reason. ``A``
        already returns the sharded ``TensorList``, so no ``split_sinogram``.
        """
        y_sim = tomo_physics.A(v_rot)
        if ratio is not None:
            y_sim = self._add_noise(y_sim, ratio)
        init = tomo_physics.fbp(y_sim)
        if self.unrolled:
            return model(y_sim, tomo_physics, init=init)
        return model(init)

    def forward(
        self,
        x_net: torch.Tensor,    # reconstruction from EVN, pre-computed by forward_pass
        physics,                # TomographyEMPair container (physics/__init__.py)
        model: nn.Module,
        **kwargs,
    ) -> torch.Tensor:
        y_net = kwargs["y_net"]  # reconstruction from ODD, pre-computed by forward_pass

        # One estimate serves both halves: equal dose, so equal sigma.
        ratio = None
        if self.noise > 0.0:
            if kwargs.get("x") is None or kwargs.get("y") is None:
                raise ValueError(
                    "EqLoss(noise>0) estimates the noise level from the EVN/ODD "
                    "sinograms, so it must be called with x= and y= (the trainer "
                    "always passes them; a direct call may not).")
            ratio = self._noise_ratio(kwargs["x"], kwargs["y"])

        # One rotation for both halves — see module docstring.
        k = self._transform.get_params(x_net)["k_idx"]
        x_rot = self._transform.transform(x_net, k_idx=k)
        y_rot = self._transform.transform(y_net, k_idx=k)

        # Independent draws per half: shared eps would correlate each term's
        # input with the other half's target.
        pe, po = physics.physics_evn, physics.physics_odd
        loss = (self._mse(self._recon(x_rot, pe, model, ratio), y_rot)
                + self._mse(self._recon(y_rot, po, model, ratio), x_rot))
        return self.weight * loss
