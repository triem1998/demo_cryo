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

import numpy as np
from deepinv.distributed import DistributedContext
from deepinv.utils.tensorlist import TensorList

from main import load_config
from toolcryo.dataset.dataset_full import EIFullDataConfig, build_ei_full_dataloaders
from toolcryo.losses import build_tomo_losses
from toolcryo.losses.losses_equivariant_tomo import EqLoss, ObsLoss, _ramp_half
from toolcryo.models import build_distributed_denoiser
from toolcryo.registry import get_preset
from toolcryo.run import RunEIFullConfig
from toolcryo.transform import Rotate3D
from toolcryo.utils.utils import psnr

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
    """A frozen ``c`` must be genuinely held, and dropped when the volume changes
    (``TomographyEMPair.update()`` rebinds ``init_*``)."""
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
    """Scale ``x_net`` by ``s`` and ``c`` scales by ``1/s``, leaving the product
    untouched. Breaks if ``c`` stops being refitted from ``A(x_net)``."""
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
    while a white residual keeps its scale. Catches a wrong axis or exponent."""
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


def test_eq_shares_one_rotation_and_swaps_the_targets():
    """Pins the formula: one k for both halves, each through its own operator, the
    *other* half as target — so the target's noise is independent."""
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


class _RecordingUnrolled:
    """Stands in for the PGD net: ``f(y, physics, init=...)``, recording both."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, y, physics, init=None):
        self.calls.append((y, physics, init))
        return _model(init)


def test_unrolled_hands_f_the_simulated_sinogram_and_an_init_built_from_it():
    """The unrolled net is measurement-conditioned: it needs ``A(x_rot)`` *and* an
    init built from that sinogram by ``fbp``, the map the deployed net uses."""
    x_net, y_net, physics, tr = _eq_fixtures()
    model = _RecordingUnrolled()

    torch.manual_seed(7)
    EqLoss(tr, weight=1.0, unrolled=True)(
        x_net=x_net, y_net=y_net, physics=physics, model=model)

    torch.manual_seed(7)
    k = tr.get_params(x_net)["k_idx"]
    rots = [tr.transform(x_net, k_idx=k), tr.transform(y_net, k_idx=k)]
    halves = [physics.physics_evn, physics.physics_odd]

    assert len(model.calls) == 2
    for (y_seen, p_seen, init_seen), v_rot, p in zip(model.calls, rots, halves):
        assert p_seen is p                                  # each half its own operator
        assert torch.allclose(y_seen, p.A(v_rot))           # the simulated sinogram
        assert torch.allclose(init_seen, p.fbp(p.A(v_rot)))  # init from that sinogram


def test_unrolled_false_is_bit_identical_to_the_denoiser_form():
    """Regression pin for the merge: the flag defaults off, and off must
    reproduce the pre-change value exactly, both couplings."""
    x_net, y_net, physics, tr = _eq_fixtures()
    kw = dict(x_net=x_net, y_net=y_net, physics=physics, model=_model)

    torch.manual_seed(7)
    default = EqLoss(tr, weight=1.0)(**kw)
    torch.manual_seed(7)
    explicit = EqLoss(tr, weight=1.0, unrolled=False)(**kw)
    assert torch.equal(default, explicit)


# --------------------------------------------------------------------------- #
# EqLoss — eq_noise
# --------------------------------------------------------------------------- #
def test_zero_noise_is_bit_identical_and_never_touches_the_sinograms():
    """The default must reproduce the clean form exactly, so runs predating
    ``eq_noise`` stay comparable."""
    x, y, _, _ = _batch()
    x_net, y_net, physics, tr = _eq_fixtures()

    torch.manual_seed(7)
    clean = EqLoss(tr, weight=1.0)(x_net=x_net, y_net=y_net, physics=physics, model=_model)
    torch.manual_seed(7)
    zero = EqLoss(tr, weight=1.0, noise=0.0)(
        x=x, y=y, x_net=x_net, y_net=y_net, physics=physics, model=_model)
    assert torch.equal(clean, zero)


def test_noise_ratio_recovers_a_known_noise_to_signal_ratio():
    """``x - y`` cancels the signal. The ``var(y) - sigma^2`` correction is what
    keeps the estimate unbiased."""
    g = torch.Generator().manual_seed(3)
    signal = torch.randn((1, 1, 64, 4, 64), generator=g) * 5.0   # per-angle common signal
    sigma = 2.0
    x = signal + sigma * torch.randn(signal.shape, generator=g)
    y = signal + sigma * torch.randn(signal.shape, generator=g)

    _, _, _, tr = _eq_fixtures()
    ratio = EqLoss(tr)._noise_ratio(x, y)

    assert ratio.shape == (1, 1, 1, 4, 1)          # one value per tilt angle
    assert torch.allclose(ratio, torch.full_like(ratio, sigma / 5.0), rtol=0.1)


def test_noise_scales_with_the_flag_and_is_drawn_independently_per_half():
    """The multiplier must reach the sinogram, and the halves must not share eps —
    that independence is what cross-coupling uses."""
    _, _, physics, tr = _eq_fixtures()
    x, y, x_net, y_net = _batch()
    crit = EqLoss(tr, noise=1.0)
    ratio = crit._noise_ratio(x, y)

    pe = physics.physics_evn
    clean = pe.A(x_net)
    torch.manual_seed(0)
    a = crit._add_noise(clean, ratio)
    b = crit._add_noise(clean, ratio)
    assert not torch.allclose(a, b)                       # independent draws

    crit.noise = 2.0
    torch.manual_seed(0)
    doubled = crit._add_noise(clean, ratio)
    torch.manual_seed(0)
    crit.noise = 1.0
    single = crit._add_noise(clean, ratio)
    assert torch.allclose(doubled - clean, 2.0 * (single - clean))


def test_sharded_noise_slices_the_ratio_to_match_each_shard():
    """A misaligned shard slice applies another angle's noise level and is invisible
    in the loss, so pin it: angles 0-1 get ratio 0, angles 2-4 a large one."""
    _, _, _, tr = _eq_fixtures()
    crit = EqLoss(tr, noise=1.0)
    ratio = torch.tensor([0.0, 0.0, 5.0, 5.0, 5.0]).reshape(1, 1, 1, 5, 1)
    shards = TensorList([torch.randn(1, 1, 6, 2, 6), torch.randn(1, 1, 6, 3, 6)])

    out = crit._add_noise(shards, ratio)

    assert len(out) == 2
    assert torch.equal(out[0], shards[0])            # ratio 0 -> untouched
    assert not torch.allclose(out[1], shards[1])     # ratio 5 -> visibly noised


def test_noise_needs_the_sinograms_and_says_so():
    """The trainer always passes x/y; a direct call may not. Fail loudly rather
    than silently training with a clean y_sim."""
    x_net, y_net, physics, tr = _eq_fixtures()
    with pytest.raises(ValueError, match="x= and y="):
        EqLoss(tr, noise=1.0)(x_net=x_net, y_net=y_net, physics=physics, model=_model)


# --------------------------------------------------------------------------- #
# EqLoss — eq_scale_free
# --------------------------------------------------------------------------- #
def _model_h(v):
    """Homogeneous stand-in for the real (bias-free) UNet: f(a*v) == a*f(v)."""
    return v * 3.0


def test_scale_free_eq_has_zero_gradient_along_the_output_scale():
    """Assert the *gradient*: a detached z-norm is value-invariant and still leaks
    the shrink direction, so a value-only test passes for the broken form."""
    x_net, y_net, physics, tr = _eq_fixtures()
    lam = torch.tensor(1.0, requires_grad=True)

    torch.manual_seed(7)
    loss = EqLoss(tr, weight=1.0, scale_free=True)(
        x_net=lam * x_net, y_net=lam * y_net, physics=physics, model=_model_h)
    grad = torch.autograd.grad(loss, lam)[0]
    assert grad.abs().item() < 1e-5


def test_checkpointed_scale_free_matches_the_uncheckpointed_form():
    """The checkpoint is a pure memory trade. Assert the GRADIENT: checkpointing
    re-runs the forward, so impurity in the region shows up only there."""
    x_net, y_net, physics, tr = _eq_fixtures()
    xn = x_net.clone().requires_grad_(True)
    crit = EqLoss(tr, weight=1.0, scale_free=True)
    kw = dict(y_net=y_net, physics=physics, model=_model_h)

    torch.manual_seed(7)
    a = crit(x_net=xn, **kw)
    ga = torch.autograd.grad(a, xn)[0]

    crit._mse = crit._mse_scale_free          # bypass the checkpoint
    torch.manual_seed(7)
    b = crit(x_net=xn, **kw)
    gb = torch.autograd.grad(b, xn)[0]

    assert torch.equal(a, b)
    assert torch.allclose(ga, gb, atol=1e-6)


def test_scale_free_is_off_by_default_and_the_default_still_sees_scale():
    """The default must be a true no-op, and with it off the amplitude channel is
    still open."""
    x_net, y_net, physics, tr = _eq_fixtures()
    assert EqLoss(tr).scale_free is False

    kw = dict(physics=physics, model=_model_h)
    torch.manual_seed(7)
    at_1 = EqLoss(tr, weight=1.0)(x_net=x_net, y_net=y_net, **kw)
    torch.manual_seed(7)
    at_01 = EqLoss(tr, weight=1.0)(x_net=0.1 * x_net, y_net=0.1 * y_net, **kw)
    assert not torch.allclose(at_1, at_01)          # off: shrinking changes it

    torch.manual_seed(7)
    sf_1 = EqLoss(tr, weight=1.0, scale_free=True)(x_net=x_net, y_net=y_net, **kw)
    torch.manual_seed(7)
    sf_01 = EqLoss(tr, weight=1.0, scale_free=True)(
        x_net=0.1 * x_net, y_net=0.1 * y_net, **kw)
    assert torch.allclose(sf_1, sf_01, atol=1e-5)   # on: shrinking is invisible


# --------------------------------------------------------------------------- #
# build_tomo_losses
# --------------------------------------------------------------------------- #
class _Cfg:
    obs_gain, obs_ramp = "none", False
    eq_noise, eq_scale_free = 0.0, False

    def __init__(self, preset, eq_weight):
        self.preset, self.eq_weight = preset, eq_weight


@pytest.mark.parametrize("preset", ["unrolled", "tomo_ei"])
def test_eq_is_skipped_entirely_at_zero_weight(preset):
    """Skipped, not zero-weighted — the term costs an extra A+fbp+denoiser per
    half (tomo_ei) or a whole extra unroll per half (unrolled)."""
    assert len(build_tomo_losses(_Cfg(preset, 0.0), transform=None)) == 1
    assert len(build_tomo_losses(_Cfg(preset, 2.0), transform=None)) == 2


def test_the_preset_selects_how_eq_calls_the_model():
    """One builder for both presets; ``cfg.preset`` is the only difference."""
    assert build_tomo_losses(_Cfg("unrolled", 2.0), transform=None)[1].unrolled is True
    assert build_tomo_losses(_Cfg("tomo_ei", 2.0), transform=None)[1].unrolled is False


# --------------------------------------------------------------------------- #
# psnr
# --------------------------------------------------------------------------- #
def test_psnr_is_scale_free_and_ranks_by_error():
    """Both operands are z-normalised, so PSNR sees shape not amplitude — which is
    why the CSV carries ``std_ratio`` beside it."""
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
    The gradient norm is the only column comparable across terms."""
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
