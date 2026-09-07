"""Cross half-set data-fidelity loss (unrolled + tomo_ei).

    L = MSE(c_odd * physics_odd.A(f(x)), y) + MSE(c_evn * physics_evn.A(f(y)), x)

``x``/``y`` are the EVN/ODD sinograms, ``f(x)``/``f(y)`` the reconstructions from
each; every half is re-projected through its own operator, at native resolution.
``obs_gain`` sets the calibration of ``a = A(x_net)`` against its target, and
is refitted on every call except where noted::

    none                  c = 1.
    znorm                 both operands z-normalised, residual zn(a) - zn(y);
                          the only mode kept in the graph, and the only one that
                          also rescales the target.
    leastsq_xnet          c = <a, y> / <a, a>, under no_grad.
    leastsq_xnet_frozen   same, computed once per tomogram and cached on
                          ``physics.init_*``.

``obs_ramp`` multiplies the residual by ``sqrt(|k|)`` along the detector axis
before it is squared, so the squared residual carries the full ramp ``|k|``.
``|k|`` is ``rfftfreq(n)`` for ``n`` the detector length, normalised to unit
mean square.
"""
from __future__ import annotations

import torch
from deepinv.loss import Loss


def _ramp_half(t: torch.Tensor) -> torch.Tensor:
    """Multiply by ``sqrt(|k|)`` along the detector axis; the caller squares.

    ``w`` is normalised to unit mean square, so a white residual keeps its scale.
    ``w(0) = 0``, so DC is dropped. Transformed in fp32 and cast back.
    """
    n = t.shape[-1]
    w = torch.fft.rfftfreq(n, device=t.device, dtype=torch.float32).abs().sqrt()
    w = w / w.pow(2).mean().sqrt().clamp_min(1e-12)
    return torch.fft.irfft(torch.fft.rfft(t.float(), dim=-1) * w, n=n, dim=-1).to(t.dtype)


def _as_sinogram(pred) -> torch.Tensor:
    """Reassemble a sharded ``A(x)`` back into one ``(B, C, V, A, N)`` sinogram.

    With ``num_operators`` set, the physics is a *stack* of per-angle-subset
    operators, so ``A`` returns one measurement per shard (a ``TensorList``)
    rather than a tensor. The shards are contiguous and in ascending angle
    order, so concatenating on the angle axis rebuilds exactly the sinogram the
    unsharded operator would have produced — which keeps this loss numerically
    identical whatever ``num_operators`` is set to. Pass-through otherwise.
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
        """Least-squares ``c`` per half, fitted to ``a = A(x_net)``.

        ``leastsq_xnet_frozen`` computes it once and caches it on the ``init_*``
        tensors, which ``TomographyEMPair.update()`` rebinds per tomogram;
        ``leastsq_xnet`` refits on every call. Taken under ``no_grad``, so ``c``
        is a constant of the step and carries no gradient.
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
