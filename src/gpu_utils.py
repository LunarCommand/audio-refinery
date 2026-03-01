"""GPU availability utilities — queries nvidia-smi for active compute processes."""

import contextlib
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

_TFLOPS_TABLE_PATH = Path(__file__).parent / "gpu_tflops.toml"


@dataclass
class GpuProcess:
    pid: int
    used_memory_mib: int


@dataclass
class GpuInfo:
    name: str
    vram_mib: int
    sm_clock_mhz: int


def query_compute_processes(device_index: int) -> list[GpuProcess]:
    """Return active compute processes on GPU *device_index* via nvidia-smi.

    Returns an empty list if the device is free, if nvidia-smi is unavailable,
    or if the device index is invalid.  Never raises.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
            "-i",
            str(device_index),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    processes: list[GpuProcess] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            with contextlib.suppress(ValueError):
                processes.append(GpuProcess(pid=int(parts[0]), used_memory_mib=int(parts[1])))
    return processes


def load_tflops_table() -> dict[str, float]:
    """Load the GPU FP16 TFLOPS lookup table from gpu_tflops.toml.

    Returns an empty dict if the file is missing or malformed — callers fall
    back to the heuristic in that case.
    """
    try:
        with open(_TFLOPS_TABLE_PATH, "rb") as f:
            data = tomllib.load(f)
        return {k: float(v) for k, v in data.get("tflops", {}).items()}
    except Exception:
        return {}


def lookup_tflops(name: str, table: dict[str, float]) -> float | None:
    """Return the FP16 TFLOPS for *name* from *table*, or None if not found.

    Tolerates the presence or absence of the "NVIDIA " prefix that nvidia-smi
    may or may not include depending on driver version and GPU SKU — some data
    center drivers report e.g. "A100-SXM4-80GB" rather than
    "NVIDIA A100-SXM4-80GB". One entry in the TOML covers both variants.
    """
    if name in table:
        return table[name]
    alt = name[len("NVIDIA ") :] if name.startswith("NVIDIA ") else f"NVIDIA {name}"
    return table.get(alt)


def detect_gpu_order() -> tuple[str, ...]:
    """Return CUDA device strings ordered best-first using nvidia-smi.

    Ranking priority:
      1. GPUs listed in gpu_tflops.toml are ranked by FP16 TFLOPS — the most
         reliable proxy for Demucs/WhisperX/Pyannote inference throughput.
         These always rank above unlisted GPUs.
      2. Unlisted GPUs fall back to (rounded VRAM GB DESC, max SM clock DESC).
         This heuristic is unreliable across GPU generations; add any cloud
         GPUs you use to gpu_tflops.toml for correct ordering.

    Falls back to ("cuda:0",) if nvidia-smi is unavailable or returns no GPUs.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,clocks.max.sm,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return ("cuda:0",)
        tflops_table = load_tflops_table()
        gpus: list[tuple[int, int, int, str]] = []
        for line in result.stdout.strip().splitlines():
            # maxsplit=3 so GPU names containing commas are kept intact.
            parts = [p.strip() for p in line.split(",", maxsplit=3)]
            if len(parts) != 4:
                continue
            try:
                gpus.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3]))
            except ValueError:
                continue
        if not gpus:
            return ("cuda:0",)

        def _sort_key(g: tuple[int, int, int, str]) -> tuple:
            idx, mem_mib, sm_clock, name = g
            tflops = lookup_tflops(name, tflops_table)
            if tflops is not None:
                # Tier 1: known GPU — rank by TFLOPS.
                return (1, tflops, 0, 0)
            # Tier 0: unknown GPU — rank by rounded VRAM then SM clock.
            return (0, 0.0, round(mem_mib / 1024), sm_clock)

        gpus.sort(key=_sort_key, reverse=True)
        return tuple(f"cuda:{g[0]}" for g in gpus)
    except Exception:
        return ("cuda:0",)


def query_gpu_info(device_index: int) -> GpuInfo | None:
    """Return name, VRAM, and max SM clock for GPU *device_index* via nvidia-smi.

    Returns None if the device is unavailable or output cannot be parsed.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,clocks.max.sm",
            "--format=csv,noheader,nounits",
            "-i",
            str(device_index),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    # maxsplit=2 keeps GPU names that contain commas intact.
    parts = [p.strip() for p in result.stdout.strip().split(",", maxsplit=2)]
    if len(parts) != 3:
        return None
    try:
        return GpuInfo(name=parts[0], vram_mib=int(parts[1]), sm_clock_mhz=int(parts[2]))
    except ValueError:
        return None


def query_gpu_temperature(device_index: int) -> int | None:
    """Return the current GPU temperature in °C, or None if unavailable."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            str(device_index),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None
