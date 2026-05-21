"""Service package initialization side effects."""

import importlib
import os

import src.service


def test_service_defaults_cuda_device_order_when_unset(monkeypatch):
    """With ``CUDA_DEVICE_ORDER`` unset, importing the service package defaults
    it to ``PCI_BUS_ID`` (mirroring the CLI in ``src/cli.py``) so
    ``REFINERY_DEVICE=cuda:N`` maps to the ``nvidia-smi`` index N rather than
    CUDA's default FASTEST_FIRST ordering."""
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    importlib.reload(src.service)
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_service_respects_explicit_cuda_device_order(monkeypatch):
    """An operator's explicit ``CUDA_DEVICE_ORDER`` is preserved (the package
    uses ``os.environ.setdefault``, which only fills in an unset value)."""
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    importlib.reload(src.service)
    assert os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
