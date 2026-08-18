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
    assert TOMOGRAPHY_BACKENDS == {"astra": TomographyEM, "torch": TomographyEMTorch}


@pytest.mark.parametrize("backend", ["astra", "torch"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_backend_is_honoured(backend, device):
    """An explicit choice is never second-guessed — including astra on CPU,
    which then fails loudly in the constructor rather than silently swapping."""
    assert resolve_tomography_backend(backend, device) == backend


def test_auto_on_cpu_is_torch():
    assert resolve_tomography_backend("auto", "cpu") == "torch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
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
