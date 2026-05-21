"""Service package initialization side effects."""

import os

import src.service


def test_service_package_pins_cuda_device_order():
    """Importing ``src.service`` applies the ``CUDA_DEVICE_ORDER=PCI_BUS_ID``
    default (mirroring the CLI in ``src/cli.py``) so ``REFINERY_DEVICE=cuda:N``
    selects the GPU at ``nvidia-smi`` index N, rather than CUDA's default
    FASTEST_FIRST ordering. The autouse fixtures in ``conftest.py`` do not touch
    this variable, so by the time any test runs the package default is in effect.
    """
    # Reference the side-effect import so it isn't flagged unused; importing the
    # package is what applies the default.
    assert src.service.__name__ == "src.service"
    assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
