"""Audio Refinery HTTP service mode.

This package provides the long-lived HTTP service that wraps the same core
pipeline as the CLI. See `_reqs/service-mode.md` and `_plans/service-mode-plan.md`.
"""

import os

# Pin CUDA's device enumeration to PCI bus order, matching the CLI
# (`src/cli.py`). Without this, CUDA's default FASTEST_FIRST ordering can make
# `REFINERY_DEVICE=cuda:N` select a different physical GPU than `nvidia-smi`
# index N on multi-GPU hosts. Set here in the service package __init__ so it
# runs before any submodule imports torch and creates a CUDA context.
# `setdefault` respects an explicit operator override.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
