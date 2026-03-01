"""Unit tests for src.transcriber — all whisperx calls are mocked."""

import sys
from datetime import UTC
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.transcription import TranscriptionResult
from src.transcriber import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_MODEL,
    TranscriptionError,
    _build_segments,
    _parse_whisperx_device,
    transcribe,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_whisperx(
    segments=None,
    language="en",
    language_probability=None,
    aligned_segments=None,
):
    """Build a mock whisperx module with sensible defaults."""
    if segments is None:
        segments = [
            {
                "text": " Look out Spider-Man!",
                "start": 1.0,
                "end": 3.5,
                "words": [
                    {"word": " Look", "start": 1.0, "end": 1.3, "score": 0.95},
                    {"word": " out", "start": 1.4, "end": 1.6, "score": 0.92},
                    {"word": " Spider-Man", "start": 1.7, "end": 2.5, "score": 0.88},
                ],
            }
        ]

    raw_result = {"segments": segments, "language": language}
    if language_probability is not None:
        raw_result["language_probability"] = language_probability

    if aligned_segments is None:
        aligned_segments = segments

    aligned_result = {"segments": aligned_segments, "language": language}

    mock_wx = MagicMock()
    mock_model = MagicMock()
    mock_model.transcribe.return_value = raw_result
    mock_wx.load_model.return_value = mock_model
    mock_wx.load_audio.return_value = MagicMock()
    mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_wx.align.return_value = aligned_result
    mock_wx.assign_word_speakers.return_value = aligned_result

    return mock_wx, raw_result, aligned_result


# ---------------------------------------------------------------------------
# _parse_whisperx_device
# ---------------------------------------------------------------------------


class TestParseWhisperxDevice:
    def test_cuda_n_split(self):
        assert _parse_whisperx_device("cuda:0") == ("cuda", 0)
        assert _parse_whisperx_device("cuda:1") == ("cuda", 1)
        assert _parse_whisperx_device("cuda:3") == ("cuda", 3)

    def test_plain_cuda(self):
        assert _parse_whisperx_device("cuda") == ("cuda", 0)

    def test_cpu(self):
        assert _parse_whisperx_device("cpu") == ("cpu", 0)


# ---------------------------------------------------------------------------
# _build_segments
# ---------------------------------------------------------------------------


class TestBuildSegments:
    def test_empty(self):
        assert _build_segments([]) == []

    def test_basic_segment(self):
        raw = [{"text": "Hello.", "start": 0.0, "end": 1.0, "words": []}]
        segs = _build_segments(raw)
        assert len(segs) == 1
        assert segs[0].text == "Hello."
        assert segs[0].start == 0.0
        assert segs[0].end == 1.0
        assert segs[0].words == []
        assert segs[0].speaker is None

    def test_words_built(self):
        raw = [
            {
                "text": "Hi there.",
                "start": 0.0,
                "end": 1.5,
                "words": [
                    {"word": "Hi", "start": 0.0, "end": 0.4, "score": 0.9},
                    {"word": "there", "start": 0.5, "end": 0.9, "score": 0.85},
                ],
            }
        ]
        segs = _build_segments(raw)
        assert len(segs[0].words) == 2
        assert segs[0].words[0].word == "Hi"
        assert segs[0].words[0].score == 0.9
        assert segs[0].words[1].word == "there"

    def test_word_missing_timestamps_defaults_to_zero(self):
        raw = [
            {
                "text": "Hi.",
                "start": 0.0,
                "end": 1.0,
                "words": [{"word": "Hi", "score": 0.7}],
            }
        ]
        segs = _build_segments(raw)
        assert segs[0].words[0].start == 0.0
        assert segs[0].words[0].end == 0.0

    def test_speaker_propagated(self):
        raw = [
            {
                "text": "I am Spider-Man.",
                "start": 0.0,
                "end": 2.0,
                "speaker": "SPEAKER_00",
                "words": [{"word": "I", "start": 0.0, "end": 0.2, "speaker": "SPEAKER_00"}],
            }
        ]
        segs = _build_segments(raw)
        assert segs[0].speaker == "SPEAKER_00"
        assert segs[0].words[0].speaker == "SPEAKER_00"

    def test_missing_optional_fields_handled(self):
        raw: list[dict] = [{}]
        segs = _build_segments(raw)
        assert segs[0].text == ""
        assert segs[0].start == 0.0
        assert segs[0].end == 0.0
        assert segs[0].words == []
        assert segs[0].speaker is None


# ---------------------------------------------------------------------------
# transcribe() — error paths
# ---------------------------------------------------------------------------


class TestTranscribeErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            transcribe(Path("/nonexistent/vocals.wav"))

    def test_import_error_raises_transcription_error(self, fake_wav):
        with patch.dict(sys.modules, {"whisperx": None}):
            with pytest.raises(TranscriptionError, match="whisperx is not installed"):
                transcribe(fake_wav)

    def test_diarization_file_not_found(self, fake_wav, mock_audio_info):
        mock_wx, _, _ = _make_mock_whisperx()
        missing = Path("/nonexistent/diarization.json")
        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                with pytest.raises(TranscriptionError, match="Diarization file not found"):
                    transcribe(fake_wav, diarization_file=missing)

    def test_invalid_diarization_json_raises(self, fake_wav, mock_audio_info, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("not valid json {{{")
        mock_wx, _, _ = _make_mock_whisperx()
        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                with pytest.raises(TranscriptionError, match="Failed to load diarization file"):
                    transcribe(fake_wav, diarization_file=bad_json)

    def test_transcription_runtime_error_raises(self, fake_wav, mock_audio_info):
        mock_wx = MagicMock()
        mock_wx.load_model.side_effect = RuntimeError("CUDA OOM")
        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                with pytest.raises(TranscriptionError, match="WhisperX transcription failed"):
                    transcribe(fake_wav)

    def test_speaker_assignment_failure_raises(self, fake_wav, mock_audio_info, tmp_path):
        mock_wx, _, _ = _make_mock_whisperx()
        mock_wx.assign_word_speakers.side_effect = RuntimeError("speaker merge failed")

        # Write a minimal valid DiarizationResult JSON
        diar_json = _make_diarization_json(fake_wav, tmp_path)

        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                with pytest.raises(TranscriptionError, match="Speaker assignment failed"):
                    transcribe(fake_wav, diarization_file=diar_json)


# ---------------------------------------------------------------------------
# transcribe() — happy path
# ---------------------------------------------------------------------------


class TestTranscribeSuccess:
    def test_returns_transcription_result(self, fake_wav, mock_audio_info):
        mock_wx, _, _ = _make_mock_whisperx(language_probability=0.99)
        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                result = transcribe(fake_wav)

        assert isinstance(result, TranscriptionResult)
        assert result.language == "en"
        assert result.language_probability == 0.99
        assert len(result.segments) == 1
        assert result.segments[0].text == " Look out Spider-Man!"
        assert len(result.segments[0].words) == 3
        assert result.diarization_applied is False
        assert result.diarization_file is None
        assert result.model_name == DEFAULT_MODEL
        assert result.compute_type == DEFAULT_COMPUTE_TYPE
        assert result.batch_size == DEFAULT_BATCH_SIZE
        assert result.processing_time_seconds >= 0

    def test_cuda_device_index_split_for_load_model(self, fake_wav, mock_audio_info):
        """cuda:0 is split into device='cuda', device_index=0 for ctranslate2."""
        mock_wx, _, _ = _make_mock_whisperx()
        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                transcribe(fake_wav, device="cuda:0")

        args, kwargs = mock_wx.load_model.call_args
        assert args[1] == "cuda"
        assert kwargs.get("device_index") == 0

    def test_language_en_default_passed_through(self, fake_wav, mock_audio_info):
        mock_wx, _, _ = _make_mock_whisperx()
        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                transcribe(fake_wav, language="en")

        _, kwargs = mock_wx.load_model.call_args
        # load_model is called with positional model + device, compute_type as kwarg
        # transcribe() on the model is called with language=wx_language
        model_call_args, model_call_kwargs = mock_wx.load_model.return_value.transcribe.call_args
        assert model_call_kwargs.get("language") == "en"

    def test_language_auto_passes_none(self, fake_wav, mock_audio_info):
        mock_wx, _, _ = _make_mock_whisperx()
        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                transcribe(fake_wav, language="auto")

        _, kwargs = mock_wx.load_model.return_value.transcribe.call_args
        assert kwargs.get("language") is None

    def test_compute_type_forwarded(self, fake_wav, mock_audio_info):
        mock_wx, _, _ = _make_mock_whisperx()
        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                result = transcribe(fake_wav, compute_type="int8")

        _, kwargs = mock_wx.load_model.call_args
        assert kwargs.get("compute_type") == "int8"
        assert result.compute_type == "int8"

    def test_batch_size_forwarded(self, fake_wav, mock_audio_info):
        mock_wx, _, _ = _make_mock_whisperx()
        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                result = transcribe(fake_wav, batch_size=8)

        _, kwargs = mock_wx.load_model.return_value.transcribe.call_args
        assert kwargs.get("batch_size") == 8
        assert result.batch_size == 8

    def test_alignment_called_with_detected_language(self, fake_wav, mock_audio_info):
        mock_wx, _, _ = _make_mock_whisperx(language="fr")
        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                result = transcribe(fake_wav, language="fr")

        _, kwargs = mock_wx.load_align_model.call_args
        assert kwargs.get("language_code") == "fr"
        assert result.language == "fr"

    def test_alignment_failure_sets_fallback_flag(self, fake_wav, mock_audio_info):
        mock_wx, raw_result, _ = _make_mock_whisperx()
        mock_wx.align.side_effect = RuntimeError("no alignment model for language")

        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                result = transcribe(fake_wav)

        assert result.alignment_fallback is True
        # Falls back to raw_result segments
        assert len(result.segments) == len(raw_result["segments"])


# ---------------------------------------------------------------------------
# transcribe() — diarization merge
# ---------------------------------------------------------------------------


class TestDiarizationMerge:
    def test_diarization_merge_sets_flag(self, fake_wav, mock_audio_info, tmp_path):
        mock_wx, _, _ = _make_mock_whisperx()
        diar_json = _make_diarization_json(fake_wav, tmp_path)

        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                result = transcribe(fake_wav, diarization_file=diar_json)

        assert result.diarization_applied is True
        assert result.diarization_file == diar_json
        mock_wx.assign_word_speakers.assert_called_once()

    def test_no_diarization_file_skips_assign(self, fake_wav, mock_audio_info):
        mock_wx, _, _ = _make_mock_whisperx()
        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                transcribe(fake_wav)

        mock_wx.assign_word_speakers.assert_not_called()

    def test_diarization_dataframe_built_correctly(self, fake_wav, mock_audio_info, tmp_path):
        """Verify assign_word_speakers receives a DataFrame with correct columns."""
        import pandas as pd

        mock_wx, _, _ = _make_mock_whisperx()
        diar_json = _make_diarization_json(fake_wav, tmp_path)
        captured_df = {}

        def capture(df, aligned):
            captured_df["df"] = df
            return aligned

        mock_wx.assign_word_speakers.side_effect = capture
        mock_wx.assign_word_speakers.return_value = None  # will use side_effect

        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch("src.transcriber.probe_audio_file", return_value=mock_audio_info):
                # Reset return_value so side_effect controls it
                mock_wx.assign_word_speakers.return_value = {"segments": []}
                transcribe(fake_wav, diarization_file=diar_json)

        df = captured_df["df"]
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"start", "end", "speaker"}
        assert len(df) == 2  # two segments in the fixture
        assert df.iloc[0]["speaker"] == "SPEAKER_00"
        assert df.iloc[1]["speaker"] == "SPEAKER_01"


# ---------------------------------------------------------------------------
# Helpers for fixtures
# ---------------------------------------------------------------------------


def _make_diarization_json(fake_wav: Path, tmp_path: Path) -> Path:
    """Write a minimal valid DiarizationResult JSON and return its path."""
    from datetime import datetime

    from src.models.audio import AudioFileInfo
    from src.models.diarization import DiarizationResult, SpeakerSegment

    now = datetime.now(UTC)
    info = AudioFileInfo(
        path=fake_wav,
        sample_rate=44100,
        channels=1,
        duration_seconds=7.0,
        frames=308700,
        format_str="WAV",
        subtype="PCM_16",
    )
    diar = DiarizationResult(
        input_file=fake_wav,
        input_info=info,
        segments=[
            SpeakerSegment(speaker_label="SPEAKER_00", start_seconds=0.0, end_seconds=3.5),
            SpeakerSegment(speaker_label="SPEAKER_01", start_seconds=4.0, end_seconds=7.0),
        ],
        num_speakers=2,
        processing_time_seconds=1.5,
        started_at=now,
        completed_at=now,
    )
    out = tmp_path / "diarization.json"
    out.write_text(diar.model_dump_json())
    return out
