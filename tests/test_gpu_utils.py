"""Tests for GPU availability utilities (gpu_utils.py)."""

from unittest.mock import MagicMock, patch

from src.gpu_utils import GpuProcess, detect_gpu_order, lookup_tflops, query_compute_processes

# ---------------------------------------------------------------------------
# query_compute_processes — nvidia-smi output parsing
# ---------------------------------------------------------------------------


def test_free_gpu_returns_empty_list():
    """Empty nvidia-smi output means the GPU is free."""
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = query_compute_processes(0)
    assert result == []


def test_single_process_parsed():
    """A single nvidia-smi row is parsed into one GpuProcess."""
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="12345, 4096\n", stderr="")
        result = query_compute_processes(0)
    assert result == [GpuProcess(pid=12345, used_memory_mib=4096)]


def test_multiple_processes_parsed():
    """Multiple nvidia-smi rows are parsed into multiple GpuProcess entries."""
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345, 4096\n67890, 8192\n",
            stderr="",
        )
        result = query_compute_processes(0)
    assert result == [
        GpuProcess(pid=12345, used_memory_mib=4096),
        GpuProcess(pid=67890, used_memory_mib=8192),
    ]


def test_nvidia_smi_unavailable_returns_empty():
    """Non-zero returncode from nvidia-smi (e.g. not installed) returns empty list."""
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="command not found")
        result = query_compute_processes(0)
    assert result == []


def test_correct_device_index_passed():
    """The -i argument forwarded to nvidia-smi matches the requested device index."""
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        query_compute_processes(1)
    cmd = mock_run.call_args[0][0]
    assert "-i" in cmd
    assert cmd[cmd.index("-i") + 1] == "1"


def test_malformed_line_is_skipped():
    """Lines that cannot be parsed (missing fields, non-numeric) are silently ignored."""
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not_a_number, 4096\n12345, 8192\n",
            stderr="",
        )
        result = query_compute_processes(0)
    # Only the valid line is returned.
    assert result == [GpuProcess(pid=12345, used_memory_mib=8192)]


# ---------------------------------------------------------------------------
# lookup_tflops — prefix-tolerant table lookup
# ---------------------------------------------------------------------------

_TABLE = {"NVIDIA GeForce RTX 4090": 82.6, "NVIDIA A100-SXM4-80GB": 78.0}


def test_lookup_tflops_exact_match():
    """Exact name match returns the correct value."""
    assert lookup_tflops("NVIDIA GeForce RTX 4090", _TABLE) == 82.6


def test_lookup_tflops_strips_nvidia_prefix():
    """Driver-reported name without 'NVIDIA ' prefix still resolves."""
    assert lookup_tflops("A100-SXM4-80GB", _TABLE) == 78.0


def test_lookup_tflops_adds_nvidia_prefix():
    """TOML entry without 'NVIDIA ' prefix resolves against driver name with it."""
    table_no_prefix = {"A100-SXM4-80GB": 78.0}
    assert lookup_tflops("NVIDIA A100-SXM4-80GB", table_no_prefix) == 78.0


def test_lookup_tflops_unknown_returns_none():
    """GPU not in the table (under any prefix variant) returns None."""
    assert lookup_tflops("NVIDIA H200", _TABLE) is None


# ---------------------------------------------------------------------------
# detect_gpu_order — TFLOPS lookup table (tier 1)
# ---------------------------------------------------------------------------

# nvidia-smi output format: "index, memory_mib, sm_clock_mhz, GPU Name"
_3090_ROW = "0, 24576, 2100, NVIDIA GeForce RTX 3090"
_4090_ROW = "1, 24564, 3105, NVIDIA GeForce RTX 4090"
_UNKNOWN_ROW = "2, 49152, 1800, NVIDIA SomeUnknownGPU"


def test_tflops_table_ranks_4090_above_3090():
    """4090 (82.6 TFLOPS) ranks above 3090 (35.6 TFLOPS) when both are in the table."""
    table = {"NVIDIA GeForce RTX 3090": 35.6, "NVIDIA GeForce RTX 4090": 82.6}
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=f"{_3090_ROW}\n{_4090_ROW}\n", stderr="")
        with patch("src.gpu_utils.load_tflops_table", return_value=table):
            result = detect_gpu_order()
    assert result == ("cuda:1", "cuda:0")


def test_known_gpu_ranks_above_unknown_gpu():
    """A GPU in the TFLOPS table always ranks above one that isn't, regardless of VRAM/clock."""
    # The unknown GPU has more VRAM and a higher SM clock — heuristic would pick it first.
    # The lookup table overrides this.
    table = {"NVIDIA GeForce RTX 4090": 82.6}
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=f"{_4090_ROW}\n{_UNKNOWN_ROW}\n", stderr="")
        with patch("src.gpu_utils.load_tflops_table", return_value=table):
            result = detect_gpu_order()
    assert result == ("cuda:1", "cuda:2")


def test_tflops_table_missing_falls_back_to_heuristic():
    """Empty TFLOPS table (e.g. file not found) falls back to VRAM/clock heuristic."""
    # cuda:0 = 3090 (24576 MiB, 2100 MHz), cuda:1 = 4090 (24564 MiB, 3105 MHz).
    # Both round to 24 GB; SM clock picks 4090 first.
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=f"{_3090_ROW}\n{_4090_ROW}\n", stderr="")
        with patch("src.gpu_utils.load_tflops_table", return_value={}):
            result = detect_gpu_order()
    assert result == ("cuda:1", "cuda:0")


def test_heuristic_vram_rounding_prevents_12mib_gap_from_misranking():
    """24576 MiB and 24564 MiB both round to 24 GB — SM clock decides, not raw MiB."""
    # Without rounding, 24576 > 24564 would put 3090 first. With rounding both
    # are 24 GB and SM clock (3105 > 2100) correctly puts 4090 first.
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=f"{_3090_ROW}\n{_4090_ROW}\n", stderr="")
        with patch("src.gpu_utils.load_tflops_table", return_value={}):
            result = detect_gpu_order()
    assert result == ("cuda:1", "cuda:0")


def test_heuristic_higher_vram_wins_across_tiers():
    """Unknown GPU with genuinely more VRAM (different tier) ranks above lower-VRAM unknown."""
    # cuda:1 has 48 GB vs cuda:0's 24 GB — should rank first regardless of clock.
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=f"{_3090_ROW}\n{_UNKNOWN_ROW}\n", stderr="")
        with patch("src.gpu_utils.load_tflops_table", return_value={}):
            result = detect_gpu_order()
    assert result == ("cuda:2", "cuda:0")


# ---------------------------------------------------------------------------
# detect_gpu_order — edge cases and failure modes
# ---------------------------------------------------------------------------


def test_detect_gpu_order_single_gpu():
    """A single GPU is returned as-is."""
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=f"{_4090_ROW}\n", stderr="")
        with patch("src.gpu_utils.load_tflops_table", return_value={}):
            result = detect_gpu_order()
    assert result == ("cuda:1",)


def test_detect_gpu_order_nvidia_smi_failure_falls_back():
    """Non-zero returncode falls back to ('cuda:0',)."""
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        result = detect_gpu_order()
    assert result == ("cuda:0",)


def test_detect_gpu_order_empty_output_falls_back():
    """Empty nvidia-smi output falls back to ('cuda:0',)."""
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = detect_gpu_order()
    assert result == ("cuda:0",)


def test_detect_gpu_order_exception_falls_back():
    """Any exception (e.g. FileNotFoundError) falls back to ('cuda:0',)."""
    with patch("src.gpu_utils.subprocess.run", side_effect=FileNotFoundError):
        result = detect_gpu_order()
    assert result == ("cuda:0",)


def test_detect_gpu_order_skips_malformed_lines():
    """Malformed lines are ignored; valid lines are still ranked correctly."""
    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=f"not_valid\n{_4090_ROW}\n", stderr="")
        with patch("src.gpu_utils.load_tflops_table", return_value={}):
            result = detect_gpu_order()
    assert result == ("cuda:1",)


# ---------------------------------------------------------------------------
# _warn_if_gpu_busy — CLI integration
# ---------------------------------------------------------------------------


def test_warn_aborts_when_user_declines(tmp_path):
    """_warn_if_gpu_busy exits 0 when the user answers 'n'."""

    from click.testing import CliRunner

    from src.cli import _warn_if_gpu_busy

    runner = CliRunner()

    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="9999, 4096", stderr="")
        with runner.isolated_filesystem(), runner.isolation(input="n\n"):
            try:
                _warn_if_gpu_busy(["cuda:0"])
                aborted = False
            except SystemExit as exc:
                aborted = exc.code == 0
    assert aborted, "Expected sys.exit(0) when user declines"


def test_warn_continues_when_user_confirms():
    """_warn_if_gpu_busy returns normally when the user answers 'y'."""
    from click.testing import CliRunner

    from src.cli import _warn_if_gpu_busy

    runner = CliRunner()

    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="9999, 4096", stderr="")
        with runner.isolation(input="y\n"):
            # Should not raise.
            _warn_if_gpu_busy(["cuda:0"])


def test_warn_skips_cpu_device():
    """_warn_if_gpu_busy never calls nvidia-smi for CPU devices."""
    from src.cli import _warn_if_gpu_busy

    with patch("src.gpu_utils.subprocess.run") as mock_run:
        _warn_if_gpu_busy(["cpu"])
    mock_run.assert_not_called()


def test_warn_skips_when_gpu_free():
    """_warn_if_gpu_busy returns without prompting when all GPUs are free."""
    from click.testing import CliRunner

    from src.cli import _warn_if_gpu_busy

    runner = CliRunner()

    with patch("src.gpu_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        # No input supplied — would error if a prompt were shown.
        with runner.isolation(input=""):
            _warn_if_gpu_busy(["cuda:0", "cuda:1"])
