"""Shared test fixtures for refinery tests."""

import os
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


@pytest.fixture(autouse=True)
def _no_slack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress Slack notifications during tests.

    ``src.notifier._send`` calls ``load_dotenv()`` on every invocation, which
    would pick up a developer's real ``SLACK_WEBHOOK_URL`` from the project
    ``.env`` and POST live messages whenever pipeline code paths hit a
    notifier call. Tests must never trigger real external side effects.
    """
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.setattr("src.notifier._load_dotenv", None)


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


def _resolve_integration_audio_dir() -> Path:
    """Return the directory holding WAV files for integration tests.

    Resolution order:
      1. ``REFINERY_TEST_AUDIO_DIR`` env var (preferred — lets each host
         point at whatever local fixture store it has).
      2. ``tests/_audio_fixtures/`` in the repo (developer convenience;
         gitignored so contributors can drop files there without committing).
    """
    env = os.getenv("REFINERY_TEST_AUDIO_DIR")
    if env:
        return Path(env)
    return Path(__file__).parent / "_audio_fixtures"


@pytest.fixture
def integration_audio_files() -> list[Path]:
    """Return all WAV files available for integration tests, or skip.

    Used by ``@pytest.mark.integration`` tests that need real audio. The
    fixture skips the test cleanly when no fixtures are present, so the
    suite runs everywhere — only running the integration paths on hosts
    that actually have audio files.
    """
    audio_dir = _resolve_integration_audio_dir()
    if not audio_dir.is_dir():
        pytest.skip(
            f"Integration audio directory not found: {audio_dir}. "
            "Set REFINERY_TEST_AUDIO_DIR=<path-to-wavs> or drop WAV files in tests/_audio_fixtures/."
        )
    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        pytest.skip(f"No WAV files found in {audio_dir}; integration tests require at least one .wav")
    return wavs


@pytest.fixture
def integration_audio(integration_audio_files: list[Path]) -> Path:
    """Single-WAV convenience fixture for tests that only need one file."""
    return integration_audio_files[0]


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
