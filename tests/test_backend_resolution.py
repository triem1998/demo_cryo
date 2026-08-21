"""Backend selection for the tomography operator.

``resolve_tomography_backend`` is the only place that decides between the astra
operator and the pure-torch one, and it is pure policy — no GPU, no astra, no
dataset — so it is tested here rather than inside the operator suite. This file
is also what proves the fallback on a machine where astra cannot run at all
(AMD/ROCm), where every other backend test is skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from toolcryo.physics import (  # noqa: E402
    TOMOGRAPHY_BACKENDS, TomographyEM, TomographyEMPair, TomographyEMTorch,
    resolve_tomography_backend,
)
from toolcryo.physics import tomography_build  # noqa: E402


def test_backend_map():
    assert set(TOMOGRAPHY_BACKENDS) == {"astra", "torch", "torch_exact"}
    assert TOMOGRAPHY_BACKENDS["astra"] is TomographyEM
    assert TOMOGRAPHY_BACKENDS["torch_exact"] is TomographyEMTorch


@pytest.mark.parametrize("backend,mode", [("torch", "fast"), ("torch_exact", "exact")])
def test_torch_backend_adjoint_mode(backend, mode):
    """``torch`` reproduces astra's approximate back-projector so that flipping
    ``auto`` from astra to torch leaves the gradient unchanged; ``torch_exact``
    opts into the true transpose."""
    op = TOMOGRAPHY_BACKENDS[backend](volume_shape=(4, 4, 4), angles_deg=[0.0], device="cpu")
    assert isinstance(op, TomographyEMTorch)
    assert op.adjoint_mode == mode


@pytest.mark.parametrize("backend", ["astra", "torch"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_backend_is_honoured(backend, device):
    """An explicit choice is never second-guessed — including astra on CPU,
    which then fails loudly in the constructor rather than silently swapping."""
    assert resolve_tomography_backend(backend, device) == backend


def test_auto_on_cpu_is_torch():
    assert resolve_tomography_backend("auto", "cpu") == "torch"


#: See the note in ``test_tomography_torch.py``: ROCm answers True here too, and
#: ``auto`` is *meant* to return "torch" there — that is
#: ``test_auto_falls_back_on_rocm`` below, not this test.
CUDA_ASTRA = torch.cuda.is_available() and torch.version.hip is None


@pytest.mark.skipif(not CUDA_ASTRA, reason="needs a real (non-ROCm) CUDA device")
def test_auto_on_cuda_is_astra():
    assert resolve_tomography_backend("auto", "cuda") == "astra"


def test_auto_falls_back_on_rocm(monkeypatch):
    """A ROCm build reports device type 'cuda' but ships no astra kernels."""
    monkeypatch.setattr(torch.version, "hip", "6.2.0", raising=False)
    assert resolve_tomography_backend("auto", "cuda") == "torch"


def test_auto_falls_back_without_astra(monkeypatch):
    monkeypatch.setattr(tomography_build.importlib.util, "find_spec", lambda name: None)
    assert resolve_tomography_backend("auto", "cuda") == "torch"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="tomography_backend"):
        resolve_tomography_backend("astra-toolbox", "cuda")


def test_pair_defaults_to_astra():
    """The dataclass default is what an un-migrated caller gets — it must stay
    the astra operator so existing behaviour is unchanged."""
    assert TomographyEMPair.__dataclass_fields__["backend"].default == "astra"
