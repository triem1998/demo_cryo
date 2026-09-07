"""``tomo_ei`` losses — ``ObsLoss`` (``obs_gain`` / ``obs_ramp``), ``EqLoss`` —
and the ``psnr`` val metric.

Toy linear operators, no astra, no dataset: pure CPU, so this also runs on an
AMD/ROCm box.

The last section is a report on real data instead of a test — it needs the local
dataset and skips without it::

    python tests/test_losses.py           # report only
    pytest tests/test_losses.py -s        # tests, then the report
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from toolcryo.losses.losses_equivariant_tomo import EqLoss
from toolcryo.losses.losses_unrolled import ObsLoss, _ramp_half
from toolcryo.utils.utils import psnr
from toolcryo.transform import Rotate3D

CONFIG = ROOT / "configs" / "conf_tomo_ei_local.yml"
MSE = torch.nn.MSELoss()


class _ToyPhysics:
    """``A`` = scale (counting its calls), ``fbp(A(v)) == v * gain``."""

    def __init__(self, scale: float, gain: float = 1.0) -> None:
        self.scale, self.gain, self.n_A = scale, gain, 0

    def A(self, x):
        self.n_A += 1
        return x * self.scale

    def fbp(self, y):
        return y * (self.gain / self.scale)


class _ToyPair:
    def __init__(self) -> None:
        g = torch.Generator().manual_seed(0)
        shape = (1, 1, 6, 6, 6)                    # cubic: all 24 rotations valid
        # distinct scales and gains per half, so operator mix-ups show up
        self.physics_evn = _ToyPhysics(2.0, gain=0.7)
        self.physics_odd = _ToyPhysics(3.0, gain=1.3)
        self.init_evn = torch.randn(shape, generator=g)
        self.init_odd = torch.randn(shape, generator=g)

    def swap_tomogram(self) -> None:
        """What ``TomographyEMPair.update()`` does: rebind both init tensors."""
        self.init_evn = torch.randn_like(self.init_evn)
        self.init_odd = torch.randn_like(self.init_odd)


def _batch():
    g = torch.Generator().manual_seed(2)
    shape = (1, 1, 6, 6, 6)
    return (torch.randn(shape, generator=g),   # x     EVN sinogram
            torch.randn(shape, generator=g),   # y     ODD sinogram
            torch.randn(shape, generator=g),   # x_net f(EVN)
            torch.randn(shape, generator=g))   # y_net f(ODD)


# --------------------------------------------------------------------------- #
# ObsLoss
# --------------------------------------------------------------------------- #
def test_none_is_a_true_no_op_and_costs_nothing():
    physics = _ToyPair()
    x, y, x_net, y_net = _batch()
    loss = ObsLoss(weight=1.0, gain="none")(
        x=x, y=y, x_net=x_net, y_net=y_net, physics=physics)

    po, pe = physics.physics_odd, physics.physics_evn
    # exactly the pre-calibration formula...
    assert torch.allclose(loss, MSE(po.A(x_net), y) + MSE(pe.A(y_net), x))
    # ...and only the estimate's projection is formed: 1 in the loss, 1 above.
    assert pe.n_A == 2 and po.n_A == 2


def test_frozen_gain_is_held_and_invalidated_on_tomogram_swap():
    """``A(x_net)`` changes every step, so a frozen ``c`` is only meaningful if
    it is genuinely held — and only correct if it is dropped when the volume
    changes, which ``TomographyEMPair.update()`` signals by rebinding ``init_*``."""
    physics = _ToyPair()
    x, y, x_net, y_net = _batch()
    crit = ObsLoss(weight=1.0, gain="leastsq_xnet_frozen")

    crit(x=x, y=y, x_net=x_net, y_net=y_net, physics=physics)
    first = crit._gain_cache[2].clone()
    crit(x=x, y=y, x_net=3.0 * x_net, y_net=3.0 * y_net, physics=physics)
    assert torch.allclose(crit._gain_cache[2], first)

    physics.swap_tomogram()
    crit(x=x, y=y, x_net=x_net, y_net=y_net, physics=physics)
    # rekeyed to the new tomogram. The VALUE need not move: c is fitted to
    # A(x_net), and x_net is unchanged here — only the cache key is.
    assert crit._gain_cache[0] is physics.init_evn
    assert crit._gain_cache[1] is physics.init_odd


def test_live_gain_refits_every_step():
    physics = _ToyPair()
    x, y, x_net, y_net = _batch()
    crit = ObsLoss(weight=1.0, gain="leastsq_xnet")
    c1, _ = crit._gains(physics, x, y, physics.physics_odd.A(x_net),
                        physics.physics_evn.A(y_net))
    c2, _ = crit._gains(physics, x, y, physics.physics_odd.A(3.0 * x_net),
                        physics.physics_evn.A(3.0 * y_net))
    assert torch.allclose(c2, c1 / 3.0, rtol=1e-4)     # the flat direction


def test_gain_is_detached_but_the_loss_still_reaches_the_model():
    """At the least-squares optimum dL/dc = 0, so detaching ``c`` is exact, and
    it stops the model moving its own denominator."""
    physics = _ToyPair()
    x, y, x_net, y_net = _batch()
    xn = x_net.clone().requires_grad_(True)
    yn = y_net.clone().requires_grad_(True)
    crit = ObsLoss(gain="leastsq_xnet_frozen")
    c_odd, c_evn = crit._gains(physics, x, y,
                               physics.physics_odd.A(xn), physics.physics_evn.A(yn))
    assert not c_odd.requires_grad and not c_evn.requires_grad

    crit(x=x, y=y, x_net=xn, y_net=yn, physics=physics).backward()
    assert xn.grad is not None and xn.grad.norm() > 0


@pytest.mark.parametrize("gain", ["znorm", "leastsq_xnet", "leastsq_xnet_frozen"])
def test_calibrated_gains_make_the_loss_scale_invariant(gain):
    """Scale ``x_net`` by ``s`` and ``c`` scales by ``1/s``, so the product is
    untouched. It breaks the moment ``c`` stops being refitted from ``A(x_net)``
    — which is what would let the network optimise its own amplitude instead."""
    physics = _ToyPair()
    x, y, x_net, y_net = _batch()
    base = ObsLoss(weight=1.0, gain=gain)(
        x=x, y=y, x_net=x_net, y_net=y_net, physics=physics)

    for s in (0.1, 7.0):
        # frozen caches per tomogram, so give each scale a fresh instance
        scaled = ObsLoss(weight=1.0, gain=gain)(
            x=x, y=y, x_net=s * x_net, y_net=s * y_net, physics=physics)
        assert torch.allclose(base, scaled, atol=1e-6), f"{gain} broke at s={s}"


def test_unknown_gain_is_rejected():
    """``leastsq``/``std`` fitted c to A(init) and are retired; the config's
    Literal refuses them, ObsLoss the GAINS check."""
    for g in ("leastsq", "std", "lstsq"):
        with pytest.raises(ValueError, match="obs_gain must be one of"):
            ObsLoss(gain=g)


def test_ramp_reweights_toward_fine_detail_at_the_same_scale():
    """A high-frequency residual must cost more than an equal-energy coarse one,
    while a white residual keeps its scale — ``w`` is normalised to unit mean
    square. Fails if the filter axis is wrong, if the exponent is 1 instead of
    1/2, or if the weighting is not applied at all."""
    n = 64
    k = torch.arange(n, dtype=torch.float32)
    lo = torch.sin(2 * torch.pi * k / n).expand(1, 1, 1, 1, n)        # 1 cycle
    hi = torch.sin(2 * torch.pi * k * 16 / n).expand(1, 1, 1, 1, n)   # 16 cycles
    assert torch.allclose(lo.pow(2).mean(), hi.pow(2).mean(), atol=1e-5)
    assert _ramp_half(hi).pow(2).mean() > 3 * _ramp_half(lo).pow(2).mean()

    white = torch.randn(1, 1, 2, 3, 256, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(_ramp_half(white).pow(2).mean(),
                          white.pow(2).mean(), rtol=0.05)


def test_ramp_is_off_by_default_and_changes_the_loss_when_on():
    physics = _ToyPair()
    kw = dict(zip(("x", "y", "x_net", "y_net"), _batch())) | {"physics": physics}
    assert ObsLoss(gain="none").ramp is False
    assert not torch.allclose(ObsLoss(gain="none", ramp=True)(**kw),
                              ObsLoss(gain="none", ramp=False)(**kw))


# --------------------------------------------------------------------------- #
# EqLoss
# --------------------------------------------------------------------------- #
def _model(v):
    """Deliberately asymmetric, so swapping operands changes the value."""
    return v * 3.0 + 0.25


def _eq_fixtures():
    _, _, x_net, y_net = _batch()
    return x_net, y_net, _ToyPair(), Rotate3D(n_trans=1, volume_shape=x_net.shape[-3:])


def test_cross_coupled_shares_one_rotation_and_swaps_the_targets():
    """Pins the whole formula: one k for both halves, each half re-simulated
    through its own operator, and the *other* half's reconstruction as target —
    so the target's noise is independent of the estimate's. The self-coupled
    form (EVN against EVN) is satisfiable by the model merely becoming
    predictable on its own output."""
    x_net, y_net, physics, tr = _eq_fixtures()

    torch.manual_seed(7)
    loss = EqLoss(tr, weight=2.0)(x_net=x_net, y_net=y_net, physics=physics, model=_model)

    torch.manual_seed(7)
    k = tr.get_params(x_net)["k_idx"]        # one draw, reused for both halves
    x_rot, y_rot = tr.transform(x_net, k_idx=k), tr.transform(y_net, k_idx=k)
    pe, po = physics.physics_evn, physics.physics_odd
    expected = (MSE(_model(pe.fbp(pe.A(x_rot))), y_rot)
                + MSE(_model(po.fbp(po.A(y_rot))), x_rot))
    assert torch.allclose(loss, 2.0 * expected)


def test_self_coupled_branch_restores_the_previous_form():
    """``eq_cross_coupled=False`` must be a *true* revert, not a third variant:
    each half samples its own rotation (``_term`` calls ``get_params`` per half)
    and is its own target."""
    x_net, y_net, physics, tr = _eq_fixtures()

    torch.manual_seed(7)
    loss = EqLoss(tr, weight=1.0, cross_coupled=False)(
        x_net=x_net, y_net=y_net, physics=physics, model=_model)

    torch.manual_seed(7)
    k1 = tr.get_params(x_net)["k_idx"]       # first draw: the EVN half
    k2 = tr.get_params(y_net)["k_idx"]       # second draw: the ODD half
    x_rot, y_rot = tr.transform(x_net, k_idx=k1), tr.transform(y_net, k_idx=k2)
    pe, po = physics.physics_evn, physics.physics_odd
    expected = (MSE(_model(pe.fbp(pe.A(x_rot))), x_rot)
                + MSE(_model(po.fbp(po.A(y_rot))), y_rot))
    assert torch.allclose(loss, expected)


# --------------------------------------------------------------------------- #
# psnr
# --------------------------------------------------------------------------- #
def test_psnr_is_scale_free_and_ranks_by_error():
    """Both operands are z-normalised, so PSNR sees shape and not amplitude —
    which is why the CSV carries ``std_ratio`` beside it. A shape mismatch is an
    axis-order error, not something to score."""
    import numpy as np
    g = np.random.default_rng(0)
    ref = g.standard_normal((4, 5, 6)).astype(np.float32)

    assert psnr(ref, ref) > 100.0
    assert psnr(7.0 * ref + 3.0, ref) == pytest.approx(psnr(ref, ref), abs=1e-3)
    assert psnr(ref + 0.1 * g.standard_normal(ref.shape), ref) > \
           psnr(ref + 1.0 * g.standard_normal(ref.shape), ref)
    with pytest.raises(ValueError, match="shape mismatch"):
        psnr(ref[:3], ref)


# --------------------------------------------------------------------------- #
# Measured scales on real data — a report, not an assertion
# --------------------------------------------------------------------------- #
def _grad_norm(model, term):
    model.zero_grad(set_to_none=True)
    term.backward(retain_graph=True)
    sq = sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return sq ** 0.5


def report():
    """Print each term's raw MSE and ``||d term / d theta||`` on a real tomogram.

    Loads ``configs/conf_tomo_ei_local.yml`` and builds the real astra/torch
    operators and denoiser. The gradient norm is the only column comparable
    across terms; the MSE column is not.
    """
    from deepinv.distributed import DistributedContext

    from main import load_config
    from toolcryo.run import RunEIFullConfig
    from toolcryo.dataset.dataset_full import EIFullDataConfig, build_ei_full_dataloaders
    from toolcryo.models import build_distributed_denoiser
    from toolcryo.registry import get_preset

    cfg = RunEIFullConfig.from_yaml(load_config(str(CONFIG)))

    with DistributedContext(seed=int(cfg.seed), seed_offset=False, cleanup=True) as ctx:
        data_cfg = EIFullDataConfig(
            input_dir=cfg.input_dir, num_workers=0, pin_memory=False,
            prefetch_factor=None, persistent_workers=False,
            max_train_vols=cfg.max_train_vols, max_val_vols=int(cfg.max_val_vols),
            seed=int(cfg.seed), train_names=cfg.train_names, val_names=cfg.val_names,
            target_shape=cfg.target_shape,
            fallback_tilt_min=cfg.tilt_min, fallback_tilt_max=cfg.tilt_max,
            data_source="measurement", crop_size=cfg.crop_size,
            normalize_crops=bool(cfg.normalize_crops),
        )
        bundle = build_ei_full_dataloaders(data_cfg)
        train_ds = bundle.train_loader.dataset

        preset = get_preset("tomo_ei")
        physics = preset["physics"](cfg, train_ds.evn_paths, train_ds.odd_paths,
                                    ctx.device, ctx)
        model, info = build_distributed_denoiser(
            cfg, ctx, int(ctx.rank), None, permute_native_to_astra=False)
        tr = Rotate3D(n_trans=1, volume_shape=physics.physics_evn.volume_shape)

        x, y, _ = next(iter(bundle.train_loader))          # EVN / ODD sinograms
        x, y = x.to(ctx.device), y.to(ctx.device)
        pe, po = physics.physics_evn, physics.physics_odd
        x_net, y_net = model(physics.init_evn), model(physics.init_odd)

        print(f"\nmodel={info}  (random init: no local pretrained ckpt)")
        print(f"volume={tuple(physics.init_evn.shape[-3:])}  sinogram={tuple(y.shape)}\n")
        print("raw scales (std)")
        for n, t in [("init_evn (data)", physics.init_evn),
                     ("x_net = f(init_evn)", x_net),
                     ("y (ODD sinogram)", y),
                     ("A_odd(x_net)", po.A(x_net)),
                     ("P_odd(x_net) = fbp(A(.))", po.fbp(po.A(x_net)))]:
            print(f"  {n:<30} {t.std().item():>9.4f}")

        terms = {
            "Obs gain=none": ObsLoss(gain="none"),
            "Obs gain=znorm": ObsLoss(gain="znorm"),
            "Obs gain=leastsq_xnet": ObsLoss(gain="leastsq_xnet"),
            "Obs gain=none ramp=True": ObsLoss(gain="none", ramp=True),
            "Eq cross-coupled": EqLoss(tr, weight=1.0),
        }
        print(f"\n  {'term':<30} {'value':>12} {'||grad||':>12}")
        print("  " + "-" * 56)
        out = {}
        for name, crit in terms.items():
            v = crit(x=x, y=y, x_net=x_net, y_net=y_net, physics=physics, model=model)
            out[name] = (v.item(), _grad_norm(model, v))
            print(f"  {name:<30} {out[name][0]:>12.5g} {out[name][1]:>12.4g}")

        eq_v, eq_g = out["Eq cross-coupled"]
        obs_v, obs_g = out["Obs gain=none"]
        print(f"\n  Obs/Eq   value = {obs_v / eq_v:8.1f}   grad = {obs_g / eq_g:8.1f}\n")
        return out


def test_report_runs():
    if not (ROOT / "dataset" / "empiar-11830").exists() or not CONFIG.exists():
        pytest.skip("local dataset or config not available")
    out = report()
    assert all(v > 0 for v, _ in out.values())


if __name__ == "__main__":
    report()
