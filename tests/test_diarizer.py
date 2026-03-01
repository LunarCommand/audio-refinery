"""Unit tests for src.diarizer — all Pyannote calls are mocked."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.diarizer import DEFAULT_MODEL, DiarizationError, diarize, load_pipeline
from src.models.diarization import DiarizationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_annotation(turns: list[tuple[str, float, float]]):
    """Build a mock pyannote Annotation that iterates over (speaker, start, end) tuples."""
    mock_annotation = MagicMock()
    track_list = []
    for speaker, start, end in turns:
        seg = MagicMock()
        seg.start = start
        seg.end = end
        track_list.append((seg, None, speaker))
    mock_annotation.itertracks.return_value = iter(track_list)
    return mock_annotation


# ---------------------------------------------------------------------------
# load_pipeline
# ---------------------------------------------------------------------------


class TestLoadPipeline:
    def test_loads_and_moves_to_device(self):
        mock_pipeline = MagicMock()
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

        # `load_pipeline` does a local `from pyannote.audio import Pipeline`.
        # Patching the live attribute triggers the real module import, which pulls in
        # matplotlib and crashes in headless CI. Pre-populate sys.modules instead so
        # the local import inside load_pipeline picks up the mock without any I/O.
        mock_pa = MagicMock()
        mock_pa.Pipeline = mock_pipeline_cls
        with patch.dict(sys.modules, {"pyannote.audio": mock_pa}):
            result = load_pipeline(DEFAULT_MODEL, "cuda", "hf_fake")

        mock_pipeline_cls.from_pretrained.assert_called_once_with(DEFAULT_MODEL)
        mock_pipeline.to.assert_called_once()
        assert result == mock_pipeline

    def test_raises_diarization_error_on_failure(self):
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.side_effect = RuntimeError("model not found")

        mock_pa = MagicMock()
        mock_pa.Pipeline = mock_pipeline_cls
        with patch.dict(sys.modules, {"pyannote.audio": mock_pa}):
            with pytest.raises(DiarizationError, match="Failed to load Pyannote pipeline"):
                load_pipeline(DEFAULT_MODEL, "cuda", "hf_fake")


# ---------------------------------------------------------------------------
# diarize()
# ---------------------------------------------------------------------------


class TestDiarize:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            diarize(Path("/nonexistent/audio.wav"), hf_token="hf_fake")

    def test_missing_hf_token_raises(self, fake_wav, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        # Prevent dotenv from loading a real .env during test
        with patch("src.diarizer.load_dotenv"):
            with pytest.raises(DiarizationError, match="HuggingFace token not found"):
                diarize(fake_wav)  # no hf_token, no env var

    def test_hf_token_from_env(self, fake_wav, mock_audio_info, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        annotation = _make_fake_annotation([("SPEAKER_00", 0.0, 1.5)])
        mock_pipeline = MagicMock(return_value=annotation)

        with (
            patch("src.diarizer.load_dotenv"),
            patch("src.diarizer.load_pipeline", return_value=mock_pipeline),
            patch("src.diarizer.probe_audio_file", return_value=mock_audio_info),
        ):
            result = diarize(fake_wav)  # token comes from env

        assert result.num_speakers == 1
        assert result.segments[0].speaker_label == "SPEAKER_00"

    def test_hf_token_arg_overrides_env(self, fake_wav, mock_audio_info, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        annotation = _make_fake_annotation([("SPEAKER_00", 0.0, 1.0)])
        mock_pipeline = MagicMock(return_value=annotation)

        with (
            patch("src.diarizer.load_dotenv"),
            patch("src.diarizer.load_pipeline") as mock_load,
            patch("src.diarizer.probe_audio_file", return_value=mock_audio_info),
        ):
            mock_load.return_value = mock_pipeline
            diarize(fake_wav, hf_token="hf_explicit")

        mock_load.assert_called_once_with(DEFAULT_MODEL, "cuda", "hf_explicit")

    def test_successful_diarization(self, fake_wav, mock_audio_info):
        turns = [
            ("SPEAKER_00", 0.0, 2.0),
            ("SPEAKER_01", 2.5, 4.5),
            ("SPEAKER_00", 5.0, 7.0),
        ]
        annotation = _make_fake_annotation(turns)
        mock_pipeline = MagicMock(return_value=annotation)

        with (
            patch("src.diarizer._resolve_hf_token", return_value="hf_fake"),
            patch("src.diarizer.load_pipeline", return_value=mock_pipeline),
            patch("src.diarizer.probe_audio_file", return_value=mock_audio_info),
        ):
            result = diarize(fake_wav, hf_token="hf_fake")

        assert isinstance(result, DiarizationResult)
        assert result.num_speakers == 2
        assert len(result.segments) == 3
        assert result.segments[0].speaker_label == "SPEAKER_00"
        assert result.segments[0].start_seconds == 0.0
        assert result.segments[0].end_seconds == 2.0
        assert result.segments[0].duration_seconds == 2.0
        assert result.processing_time_seconds >= 0
        assert result.model_name == DEFAULT_MODEL

    def test_min_max_speakers_forwarded(self, fake_wav, mock_audio_info):
        annotation = _make_fake_annotation([("SPEAKER_00", 0.0, 1.0)])
        mock_pipeline = MagicMock(return_value=annotation)

        with (
            patch("src.diarizer._resolve_hf_token", return_value="hf_fake"),
            patch("src.diarizer.load_pipeline", return_value=mock_pipeline),
            patch("src.diarizer.probe_audio_file", return_value=mock_audio_info),
        ):
            result = diarize(fake_wav, hf_token="hf_fake", min_speakers=2, max_speakers=4)

        mock_pipeline.assert_called_once_with(str(fake_wav.resolve()), min_speakers=2, max_speakers=4)
        assert result.min_speakers == 2
        assert result.max_speakers == 4

    def test_no_speaker_hints_omits_kwargs(self, fake_wav, mock_audio_info):
        annotation = _make_fake_annotation([("SPEAKER_00", 0.0, 1.0)])
        mock_pipeline = MagicMock(return_value=annotation)

        with (
            patch("src.diarizer._resolve_hf_token", return_value="hf_fake"),
            patch("src.diarizer.load_pipeline", return_value=mock_pipeline),
            patch("src.diarizer.probe_audio_file", return_value=mock_audio_info),
        ):
            diarize(fake_wav, hf_token="hf_fake")

        # Called with only the file path — no min/max kwargs
        mock_pipeline.assert_called_once_with(str(fake_wav.resolve()))

    def test_pipeline_runtime_error_raises_diarization_error(self, fake_wav, mock_audio_info):
        mock_pipeline = MagicMock(side_effect=RuntimeError("CUDA OOM"))

        with (
            patch("src.diarizer._resolve_hf_token", return_value="hf_fake"),
            patch("src.diarizer.load_pipeline", return_value=mock_pipeline),
            patch("src.diarizer.probe_audio_file", return_value=mock_audio_info),
        ):
            with pytest.raises(DiarizationError, match="Pyannote pipeline failed"):
                diarize(fake_wav, hf_token="hf_fake")
