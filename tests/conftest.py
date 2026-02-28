"""Shared test fixtures for refinery tests."""

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.audio import AudioFileInfo


@pytest.fixture(autouse=True)
def _gpu_free():
    """Prevent real nvidia-smi GPU-check calls during tests.

    All tests see every GPU as free by default.  Tests that need to simulate
    occupied GPUs patch ``src.gpu_utils.subprocess.run`` themselves —
    the inner patch takes precedence over this one.
    """
    with patch(
        "src.gpu_utils.subprocess.run",
        return_value=MagicMock(returncode=0, stdout="", stderr=""),
    ):
        yield


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for Demucs results."""
    out = tmp_path / "demucs_output"
    out.mkdir()
    return out


@pytest.fixture
def fake_wav(tmp_path: Path) -> Path:
    """Create a minimal valid WAV file (1 second of silence, 16-bit mono 44100 Hz)."""
    wav_path = tmp_path / "test_audio.wav"
    n_frames = 44100
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return wav_path


@pytest.fixture
def mock_audio_info(fake_wav: Path) -> AudioFileInfo:
    """An AudioFileInfo matching the fake_wav fixture."""
    return AudioFileInfo(
        path=fake_wav,
        sample_rate=44100,
        channels=1,
        duration_seconds=1.0,
        frames=44100,
        format_str="WAV",
        subtype="PCM_16",
    )
