"""Unit tests for src.sentiment_analyzer — all transformers calls are mocked."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.audio import AudioFileInfo
from src.models.transcription import TranscriptionResult
from src.sentiment_analyzer import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    SentimentError,
    _parse_device,
    analyze_sentiment,
    load_sentiment_pipeline,
    merge_sentiment_into_transcription,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio_info(path: Path) -> AudioFileInfo:
    return AudioFileInfo(
        path=path,
        sample_rate=44100,
        channels=1,
        duration_seconds=10.0,
        frames=441000,
        format_str="WAV",
        subtype="PCM_16",
    )


def _make_transcription(
    tmp_path: Path,
    segments: list[dict] | None = None,
    filename: str = "transcription_001.json",
) -> Path:
    """Write a minimal TranscriptionResult JSON to a temp file and return the path."""
    now = datetime.now(timezone.utc).isoformat()
    audio_path = tmp_path / "audio_001.wav"
    audio_path.write_bytes(b"\x00" * 64)
    if segments is None:
        segments = [
            {"text": " Hello world", "start": 0.0, "end": 2.0, "words": []},
            {"text": " Goodbye", "start": 3.0, "end": 4.5, "words": [], "speaker": "SPEAKER_00"},
        ]
    tx_data = {
        "input_file": str(audio_path),
        "input_info": {
            "path": str(audio_path),
            "sample_rate": 44100,
            "channels": 1,
            "duration_seconds": 10.0,
            "frames": 441000,
            "format_str": "WAV",
            "subtype": "PCM_16",
        },
        "language": "en",
        "segments": segments,
        "device": "cuda",
        "compute_type": "float16",
        "batch_size": 16,
        "processing_time_seconds": 5.0,
        "started_at": now,
        "completed_at": now,
    }
    p = tmp_path / filename
    p.write_text(json.dumps(tx_data))
    return p


def _mock_pipeline_output():
    """Return a callable that mimics a HuggingFace text-classification pipeline."""

    def _pipeline(text, top_k=None):
        return [
            {"label": "positive", "score": 0.8},
            {"label": "neutral", "score": 0.15},
            {"label": "negative", "score": 0.05},
        ]

    return _pipeline


# ---------------------------------------------------------------------------
# _parse_device
# ---------------------------------------------------------------------------


class TestParseDevice:
    def test_cpu(self):
        assert _parse_device("cpu") == "cpu"

    def test_cuda(self):
        assert _parse_device("cuda") == 0

    def test_cuda_n(self):
        assert _parse_device("cuda:0") == 0
        assert _parse_device("cuda:1") == 1
        assert _parse_device("cuda:3") == 3


# ---------------------------------------------------------------------------
# load_sentiment_pipeline
# ---------------------------------------------------------------------------


class TestLoadSentimentPipeline:
    def test_calls_hf_pipeline(self):
        mock_pipe_func = MagicMock(return_value="mocked_pipeline_object")
        mock_transformers = MagicMock()
        mock_transformers.pipeline = mock_pipe_func
        # Patch the transformers module in sys.modules so the function imports our mock
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            result = load_sentiment_pipeline("mymodel", "cpu")
        # Verify the call used "text-classification" and the model name
        mock_pipe_func.assert_called_once_with("text-classification", model="mymodel", device="cpu")
        assert result == "mocked_pipeline_object"

    def test_import_error_raises_sentiment_error(self):
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(SentimentError, match="transformers not available"):
                load_sentiment_pipeline(DEFAULT_MODEL, "cpu")


# ---------------------------------------------------------------------------
# analyze_sentiment
# ---------------------------------------------------------------------------


class TestAnalyzeSentiment:
    def test_happy_path(self, tmp_path: Path):
        tx_file = _make_transcription(tmp_path)
        pipeline = _mock_pipeline_output()

        result = analyze_sentiment(tx_file, _sentiment_pipeline=pipeline)

        assert len(result.segments) == 2
        assert result.device == DEFAULT_DEVICE
        assert result.model_name == DEFAULT_MODEL
        assert result.processing_time_seconds >= 0
        # Scores should be sorted descending
        for seg in result.segments:
            scores = [s.score for s in seg.scores]
            assert scores == sorted(scores, reverse=True)
            assert seg.primary_sentiment == seg.scores[0].label

    def test_text_and_speaker_carried_through(self, tmp_path: Path):
        tx_file = _make_transcription(tmp_path)
        pipeline = _mock_pipeline_output()

        result = analyze_sentiment(tx_file, _sentiment_pipeline=pipeline)

        seg_with_speaker = next((s for s in result.segments if s.speaker is not None), None)
        assert seg_with_speaker is not None
        assert seg_with_speaker.speaker == "SPEAKER_00"
        assert seg_with_speaker.text == " Goodbye"

    def test_file_not_found(self, tmp_path: Path):
        missing = tmp_path / "no_such_file.json"
        with pytest.raises(FileNotFoundError):
            analyze_sentiment(missing, _sentiment_pipeline=_mock_pipeline_output())

    def test_invalid_json_raises_sentiment_error(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json at all {{{")
        with pytest.raises(SentimentError, match="Failed to parse"):
            analyze_sentiment(bad, _sentiment_pipeline=_mock_pipeline_output())

    def test_no_segments_raises_sentiment_error(self, tmp_path: Path):
        tx_file = _make_transcription(tmp_path, segments=[])
        with pytest.raises(SentimentError, match="No usable text"):
            analyze_sentiment(tx_file, _sentiment_pipeline=_mock_pipeline_output())

    def test_all_empty_text_raises_sentiment_error(self, tmp_path: Path):
        tx_file = _make_transcription(
            tmp_path,
            segments=[
                {"text": "", "start": 0.0, "end": 1.0, "words": []},
                {"text": "   ", "start": 1.0, "end": 2.0, "words": []},
            ],
        )
        with pytest.raises(SentimentError, match="No usable text"):
            analyze_sentiment(tx_file, _sentiment_pipeline=_mock_pipeline_output())

    def test_single_segment_fails_others_succeed(self, tmp_path: Path):
        tx_file = _make_transcription(
            tmp_path,
            segments=[
                {"text": " Good segment", "start": 0.0, "end": 1.0, "words": []},
                {"text": " Bad segment", "start": 2.0, "end": 3.0, "words": []},
                {"text": " Another good", "start": 4.0, "end": 5.0, "words": []},
            ],
        )
        call_count = 0

        def _failing_pipeline(text, top_k=None):
            nonlocal call_count
            call_count += 1
            if "Bad" in text:
                raise RuntimeError("model error")
            return [{"label": "positive", "score": 1.0}]

        result = analyze_sentiment(tx_file, _sentiment_pipeline=_failing_pipeline)

        assert len(result.segments) == 2
        assert all(s.primary_sentiment == "positive" for s in result.segments)

    def test_all_segments_fail_raises_sentiment_error(self, tmp_path: Path):
        tx_file = _make_transcription(tmp_path)

        def _always_fail(text, top_k=None):
            raise RuntimeError("crash")

        with pytest.raises(SentimentError, match="All segments failed"):
            analyze_sentiment(tx_file, _sentiment_pipeline=_always_fail)

    def test_preloaded_pipeline_used_directly(self, tmp_path: Path):
        tx_file = _make_transcription(tmp_path)
        pipeline = MagicMock(return_value=[{"label": "positive", "score": 1.0}])

        with patch("src.sentiment_analyzer.load_sentiment_pipeline") as mock_load:
            analyze_sentiment(tx_file, _sentiment_pipeline=pipeline)

        mock_load.assert_not_called()
        assert pipeline.call_count == 2  # two non-empty segments

    def test_no_preloaded_pipeline_loads_model(self, tmp_path: Path):
        tx_file = _make_transcription(tmp_path)
        mock_pipe = MagicMock(return_value=[{"label": "positive", "score": 1.0}])

        with patch("src.sentiment_analyzer.load_sentiment_pipeline", return_value=mock_pipe) as mock_load:
            analyze_sentiment(tx_file, device="cpu", model="mymodel")

        mock_load.assert_called_once_with("mymodel", "cpu")

    def test_empty_segments_skipped_silently(self, tmp_path: Path):
        tx_file = _make_transcription(
            tmp_path,
            segments=[
                {"text": "", "start": 0.0, "end": 1.0, "words": []},
                {"text": " Valid", "start": 2.0, "end": 3.0, "words": []},
                {"text": "   ", "start": 4.0, "end": 5.0, "words": []},
            ],
        )
        pipeline = MagicMock(return_value=[{"label": "neutral", "score": 1.0}])
        result = analyze_sentiment(tx_file, _sentiment_pipeline=pipeline)

        assert len(result.segments) == 1
        pipeline.assert_called_once()


# ---------------------------------------------------------------------------
# merge_sentiment_into_transcription
# ---------------------------------------------------------------------------


class TestMergeSentimentIntoTranscription:
    def _make_sentiment_result(self, tx_file: Path):
        from src.sentiment_analyzer import analyze_sentiment

        pipeline = _mock_pipeline_output()
        return analyze_sentiment(tx_file, _sentiment_pipeline=pipeline)

    def test_happy_path(self, tmp_path: Path):
        tx_file = _make_transcription(tmp_path)
        sent = self._make_sentiment_result(tx_file)

        merge_sentiment_into_transcription(tx_file, sent)

        reloaded = TranscriptionResult.model_validate_json(tx_file.read_text())
        assert reloaded.sentiment_applied is True
        non_empty = [s for s in reloaded.segments if s.text and s.text.strip()]
        assert all(s.sentiment is not None for s in non_empty)

    def test_unmatched_segments_remain_none(self, tmp_path: Path):
        """Segments not present in sentiment_result (e.g. empty text) keep sentiment=None."""
        tx_file = _make_transcription(
            tmp_path,
            segments=[
                {"text": " Hello", "start": 0.0, "end": 1.0, "words": []},
                {"text": "", "start": 2.0, "end": 3.0, "words": []},
            ],
        )
        # Only the first segment is analyzed (second has empty text)
        pipeline = _mock_pipeline_output()
        from src.sentiment_analyzer import analyze_sentiment

        sent = analyze_sentiment(tx_file, _sentiment_pipeline=pipeline)

        merge_sentiment_into_transcription(tx_file, sent)

        reloaded = TranscriptionResult.model_validate_json(tx_file.read_text())
        # First segment matched
        assert reloaded.segments[0].sentiment is not None
        # Second segment (empty text, not in sentiment_result) still None
        assert reloaded.segments[1].sentiment is None

    def test_existing_transcription_fields_preserved(self, tmp_path: Path):
        tx_file = _make_transcription(tmp_path)
        sent = self._make_sentiment_result(tx_file)

        merge_sentiment_into_transcription(tx_file, sent)

        reloaded = TranscriptionResult.model_validate_json(tx_file.read_text())
        assert reloaded.language == "en"
        assert reloaded.device == "cuda"
        assert reloaded.compute_type == "float16"

    def test_partial_match(self, tmp_path: Path):
        """Only the two analyzed segments get sentiment; no error on partial coverage."""
        tx_file = _make_transcription(
            tmp_path,
            segments=[
                {"text": " Seg A", "start": 0.0, "end": 1.0, "words": []},
                {"text": " Seg B", "start": 2.0, "end": 3.0, "words": []},
                {"text": " Seg C", "start": 4.0, "end": 5.0, "words": []},
            ],
        )
        call_count = 0

        def _partial_pipeline(text, top_k=None):
            nonlocal call_count
            call_count += 1
            if "B" in text:
                raise RuntimeError("skip B")
            return [{"label": "positive", "score": 1.0}]

        from src.sentiment_analyzer import analyze_sentiment

        sent = analyze_sentiment(tx_file, _sentiment_pipeline=_partial_pipeline)
        # Only A and C in sentiment result
        assert len(sent.segments) == 2

        merge_sentiment_into_transcription(tx_file, sent)
        reloaded = TranscriptionResult.model_validate_json(tx_file.read_text())

        assert reloaded.segments[0].sentiment is not None  # A matched
        assert reloaded.segments[1].sentiment is None  # B not in sentiment_result
        assert reloaded.segments[2].sentiment is not None  # C matched
