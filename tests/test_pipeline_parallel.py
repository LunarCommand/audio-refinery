"""Tests for pipeline-parallel CLI command and supporting utilities."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.cli import cli
from src.pipeline import partition_ids

# ---------------------------------------------------------------------------
# partition_ids
# ---------------------------------------------------------------------------


def test_partition_even() -> None:
    ids = ["a", "b", "c", "d", "e", "f"]
    parts = partition_ids(ids, n=2)
    assert isinstance(parts, list)
    assert parts[0] == ["a", "c", "e"]
    assert parts[1] == ["b", "d", "f"]


def test_partition_odd() -> None:
    ids = ["a", "b", "c", "d", "e"]
    parts = partition_ids(ids, n=2)
    assert parts[0] == ["a", "c", "e"]
    assert parts[1] == ["b", "d"]


def test_partition_single() -> None:
    parts = partition_ids(["only"], n=2)
    assert parts[0] == ["only"]
    assert parts[1] == []


def test_partition_empty() -> None:
    parts = partition_ids([], n=2)
    assert parts[0] == []
    assert parts[1] == []


def test_partition_three_workers() -> None:
    ids = ["a", "b", "c", "d", "e", "f"]
    parts = partition_ids(ids, n=3)
    assert isinstance(parts, list)
    assert len(parts) == 3
    assert parts[0] == ["a", "d"]
    assert parts[1] == ["b", "e"]
    assert parts[2] == ["c", "f"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 64)
    return path


def _make_base_dir(tmp_path: Path, n_files: int = 4) -> Path:
    """Create a minimal base-dir with n WAV files in extracted/."""
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True)
    for i in range(n_files):
        _make_wav(extracted / f"audio_file{i:02d}.wav")
    return tmp_path


def _successful_proc() -> MagicMock:
    proc = MagicMock()
    proc.pid = 12345
    proc.wait.return_value = 0
    proc.poll.return_value = 0
    return proc


def _failed_proc(rc: int = 1) -> MagicMock:
    proc = MagicMock()
    proc.pid = 99999
    proc.wait.return_value = rc
    proc.poll.return_value = rc
    return proc


def _write_worker_summary(path: Path, processed: int = 2, skipped: int = 0, failed: int = 0) -> None:
    """Write a minimal pipeline summary JSON for a worker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "run_at": "2026-01-01T00:00:00+00:00",
        "total_discovered": processed + skipped,
        "total_time_seconds": 100.0,
        "avg_time_per_file_seconds": 50.0,
        "stages": {
            "separation": {
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
                "stage_time_seconds": 10.0,
                "avg_time_per_file_seconds": 5.0,
            },
            "diarization": {
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
                "stage_time_seconds": 20.0,
                "avg_time_per_file_seconds": 10.0,
            },
            "transcription": {
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
                "stage_time_seconds": 70.0,
                "avg_time_per_file_seconds": 35.0,
            },
        },
        "failures": [],
    }
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# pipeline-parallel — manifest file creation
# ---------------------------------------------------------------------------


def test_manifest_files_written(tmp_path: Path) -> None:
    """After launch, manifest_0.txt and manifest_1.txt must exist with interleaved IDs."""
    base = _make_base_dir(tmp_path, n_files=4)
    # files will be: file00, file01, file02, file03 (sorted, all same size)
    # W0 gets even-indexed positions: [file00, file02]
    # W1 gets odd-indexed positions:  [file01, file03]
    runner = CliRunner()

    proc_0 = _successful_proc()
    proc_1 = _successful_proc()

    # Write dummy summary files so the aggregation step doesn't error.
    def fake_popen(cmd, stdout, stderr):
        # Detect which worker this is by the --device value in its command.
        summary_path = None
        for i, arg in enumerate(cmd):
            if arg == "--summary-file":
                summary_path = Path(cmd[i + 1])
                break
        if summary_path:
            _write_worker_summary(summary_path)
        if "--device" in cmd and "cuda:0" in cmd:
            return proc_0
        return proc_1

    with patch("src.cli.detect_gpu_order", return_value=("cuda:0", "cuda:1")):
        with patch("src.cli.subprocess.Popen", side_effect=fake_popen):
            with patch("sys.argv", ["audio-refinery"]):
                result = runner.invoke(
                    cli,
                    ["pipeline-parallel", "--base-dir", str(base)],
                    input="y\n",  # answer the RAM disk confirmation if /mnt/fast_scratch is absent
                )

    assert result.exit_code == 0, result.output

    manifest_0 = base / "manifests" / "manifest_0.txt"
    manifest_1 = base / "manifests" / "manifest_1.txt"
    assert manifest_0.exists(), "manifest_0.txt not created"
    assert manifest_1.exists(), "manifest_1.txt not created"

    ids_0 = [line for line in manifest_0.read_text().splitlines() if line]
    ids_1 = [line for line in manifest_1.read_text().splitlines() if line]

    # Combined should equal all discovered IDs.
    assert sorted(ids_0 + ids_1) == ["file00", "file01", "file02", "file03"]
    # No ID should appear in both manifests.
    assert set(ids_0).isdisjoint(set(ids_1))
    # W0 gets even-indexed positions (0, 2); W1 gets odd-indexed (1, 3).
    assert ids_0 == ["file00", "file02"]
    assert ids_1 == ["file01", "file03"]


def test_no_source_dir_exits(tmp_path: Path) -> None:
    """pipeline-parallel must exit 1 when <base>/extracted does not exist."""
    runner = CliRunner()
    result = runner.invoke(cli, ["pipeline-parallel", "--base-dir", str(tmp_path)])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# pipeline-parallel — worker args
# ---------------------------------------------------------------------------


def test_worker_device_args(tmp_path: Path) -> None:
    """Each worker must receive the correct --device argument."""
    base = _make_base_dir(tmp_path, n_files=2)
    runner = CliRunner()

    captured_cmds: list[list[str]] = []

    def fake_popen(cmd, stdout, stderr):
        captured_cmds.append(list(cmd))
        for i, arg in enumerate(cmd):
            if arg == "--summary-file":
                _write_worker_summary(Path(cmd[i + 1]))
                break
        return _successful_proc()

    with patch("src.cli.subprocess.Popen", side_effect=fake_popen), patch("sys.argv", ["audio-refinery"]):
        result = runner.invoke(
            cli,
            ["pipeline-parallel", "--base-dir", str(base), "--device", "cuda:0", "--device", "cuda:1"],
            input="y\n",
        )

    assert result.exit_code == 0, result.output
    assert len(captured_cmds) == 2

    # One worker gets cuda:0 and the other cuda:1.
    all_devices = [cmd[cmd.index("--device") + 1] for cmd in captured_cmds]
    assert "cuda:0" in all_devices
    assert "cuda:1" in all_devices


def test_worker_manifest_and_summary_args(tmp_path: Path) -> None:
    """Each worker must receive --manifest and --summary-file pointing to worker-specific paths."""
    base = _make_base_dir(tmp_path, n_files=2)
    runner = CliRunner()

    captured_cmds: list[list[str]] = []

    def fake_popen(cmd, stdout, stderr):
        captured_cmds.append(list(cmd))
        for i, arg in enumerate(cmd):
            if arg == "--summary-file":
                _write_worker_summary(Path(cmd[i + 1]))
                break
        return _successful_proc()

    with patch("src.cli.detect_gpu_order", return_value=("cuda:0", "cuda:1")):
        with patch("src.cli.subprocess.Popen", side_effect=fake_popen):
            with patch("sys.argv", ["audio-refinery"]):
                result = runner.invoke(
                    cli,
                    ["pipeline-parallel", "--base-dir", str(base)],
                    input="y\n",
                )

    assert result.exit_code == 0, result.output
    assert len(captured_cmds) == 2

    for cmd in captured_cmds:
        # Every worker command must include a --manifest arg.
        assert "--manifest" in cmd, f"--manifest missing from {cmd}"
        # Every worker command must include a --summary-file arg.
        assert "--summary-file" in cmd, f"--summary-file missing from {cmd}"

    # Worker 0 manifest and worker 1 manifest must differ.
    manifests = [cmd[cmd.index("--manifest") + 1] for cmd in captured_cmds]
    assert manifests[0] != manifests[1]

    # Worker 0 summary and worker 1 summary must differ.
    summaries = [cmd[cmd.index("--summary-file") + 1] for cmd in captured_cmds]
    assert summaries[0] != summaries[1]


def test_worker_demucs_dir_forwarded(tmp_path: Path) -> None:
    """Workers must receive --demucs-dir so they skip the interactive RAM disk check."""
    base = _make_base_dir(tmp_path, n_files=2)
    runner = CliRunner()

    captured_cmds: list[list[str]] = []

    def fake_popen(cmd, stdout, stderr):
        captured_cmds.append(list(cmd))
        for i, arg in enumerate(cmd):
            if arg == "--summary-file":
                _write_worker_summary(Path(cmd[i + 1]))
                break
        return _successful_proc()

    with patch("src.cli.detect_gpu_order", return_value=("cuda:0", "cuda:1")):
        with patch("src.cli.subprocess.Popen", side_effect=fake_popen):
            with patch("sys.argv", ["audio-refinery"]):
                result = runner.invoke(
                    cli,
                    ["pipeline-parallel", "--base-dir", str(base)],
                    input="y\n",
                )

    assert result.exit_code == 0, result.output
    for cmd in captured_cmds:
        assert "--demucs-dir" in cmd, f"--demucs-dir missing from {cmd}"


# ---------------------------------------------------------------------------
# pipeline-parallel — three workers
# ---------------------------------------------------------------------------


def test_three_worker_manifests(tmp_path: Path) -> None:
    """With three --device flags, three manifest files must be created with disjoint IDs."""
    base = _make_base_dir(tmp_path, n_files=6)
    runner = CliRunner()

    captured_cmds: list[list[str]] = []

    def fake_popen(cmd, stdout, stderr):
        captured_cmds.append(list(cmd))
        for i, arg in enumerate(cmd):
            if arg == "--summary-file":
                _write_worker_summary(Path(cmd[i + 1]), processed=2)
                break
        return _successful_proc()

    with patch("src.cli.subprocess.Popen", side_effect=fake_popen), patch("sys.argv", ["audio-refinery"]):
        result = runner.invoke(
            cli,
            [
                "pipeline-parallel",
                "--base-dir",
                str(base),
                "--device",
                "cuda:0",
                "--device",
                "cuda:1",
                "--device",
                "cuda:2",
            ],
            input="y\n",
        )

    assert result.exit_code == 0, result.output
    assert len(captured_cmds) == 3

    manifest_0 = base / "manifests" / "manifest_0.txt"
    manifest_1 = base / "manifests" / "manifest_1.txt"
    manifest_2 = base / "manifests" / "manifest_2.txt"
    assert manifest_0.exists()
    assert manifest_1.exists()
    assert manifest_2.exists()

    ids_0 = [line for line in manifest_0.read_text().splitlines() if line]
    ids_1 = [line for line in manifest_1.read_text().splitlines() if line]
    ids_2 = [line for line in manifest_2.read_text().splitlines() if line]

    all_ids = sorted(ids_0 + ids_1 + ids_2)
    assert all_ids == ["file00", "file01", "file02", "file03", "file04", "file05"]
    assert set(ids_0).isdisjoint(set(ids_1))
    assert set(ids_0).isdisjoint(set(ids_2))
    assert set(ids_1).isdisjoint(set(ids_2))


# ---------------------------------------------------------------------------
# pipeline-parallel — power limit
# ---------------------------------------------------------------------------


def test_power_limit_calls_nvidia_smi(tmp_path: Path) -> None:
    """With --power-limit set, nvidia-smi must be called once per GPU before workers launch."""
    base = _make_base_dir(tmp_path, n_files=2)
    runner = CliRunner()

    nvidia_calls: list[list[str]] = []
    popen_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        nvidia_calls.append(list(cmd))
        return MagicMock(returncode=0, stderr="")

    def fake_popen(cmd, stdout, stderr):
        popen_calls.append(list(cmd))
        for i, arg in enumerate(cmd):
            if arg == "--summary-file":
                _write_worker_summary(Path(cmd[i + 1]))
                break
        return _successful_proc()

    with patch("src.cli.detect_gpu_order", return_value=("cuda:0", "cuda:1")):
        with patch("src.cli.subprocess.run", fake_run):
            with patch("src.cli.subprocess.Popen", side_effect=fake_popen):
                with patch("sys.argv", ["audio-refinery"]):
                    result = runner.invoke(
                        cli,
                        ["pipeline-parallel", "--base-dir", str(base), "--power-limit", "350"],
                        input="y\n",
                    )

    assert result.exit_code == 0, result.output
    # Filter to power-limit calls only (GPU-check calls also go through subprocess.run).
    pl_calls = [c for c in nvidia_calls if "-pl" in c]
    assert len(pl_calls) == 2, f"Expected 2 power-limit calls, got: {pl_calls}"
    for call_args in pl_calls:
        assert "nvidia-smi" in call_args
        assert "350" in call_args
    # nvidia-smi calls should precede Popen calls.
    assert len(popen_calls) == 2


# ---------------------------------------------------------------------------
# pipeline-parallel — worker failure
# ---------------------------------------------------------------------------


def test_worker_failure_exits_nonzero(tmp_path: Path) -> None:
    """If any worker exits non-zero, pipeline-parallel must exit with a non-zero code."""
    base = _make_base_dir(tmp_path, n_files=2)
    runner = CliRunner()

    call_count = 0

    def fake_popen(cmd, stdout, stderr):
        nonlocal call_count
        call_count += 1
        for i, arg in enumerate(cmd):
            if arg == "--summary-file":
                _write_worker_summary(Path(cmd[i + 1]))
                break
        # First call (W0) succeeds; second call (W1) fails.
        return _successful_proc() if call_count == 1 else _failed_proc(rc=1)

    with patch("src.cli.detect_gpu_order", return_value=("cuda:0", "cuda:1")):
        with patch("src.cli.subprocess.Popen", side_effect=fake_popen):
            with patch("sys.argv", ["audio-refinery"]):
                result = runner.invoke(
                    cli,
                    ["pipeline-parallel", "--base-dir", str(base)],
                    input="y\n",
                )

    assert result.exit_code == 1
