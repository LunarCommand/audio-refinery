"""Service package initialization side effects."""

import os

import src.service  # noqa: F401  — importing the package applies the CUDA_DEVICE_ORDER default


def test_service_package_pins_cuda_device_order():
    """Importing ``src.service`` defaults ``CUDA_DEVICE_ORDER`` to ``PCI_BUS_ID``
    (mirroring the CLI in ``src/cli.py``) so ``REFINERY_DEVICE=cuda:N`` selects
    the same physical GPU as ``nvidia-smi`` index N, rather than CUDA's default
    FASTEST_FIRST ordering. The autouse fixtures in ``conftest.py`` do not touch
    this variable, so by the time any test runs the package default is in effect.
    """
    assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
