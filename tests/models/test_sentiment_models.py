"""Unit tests for shared sentiment and updated transcription models."""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.models.sentiment import SegmentSentiment, SentimentResult, SentimentScore
from src.models.transcription import TranscriptionResult, TranscriptSegment

# ---------------------------------------------------------------------------
# SentimentScore
# ---------------------------------------------------------------------------


class TestSentimentScore:
    def test_construction(self):
        s = SentimentScore(label="positive", score=0.9)
        assert s.label == "positive"
        assert s.score == 0.9

    def test_json_roundtrip(self):
        s = SentimentScore(label="neutral", score=0.5)
        data = json.loads(s.model_dump_json())
        assert data["label"] == "neutral"
        assert data["score"] == 0.5
        s2 = SentimentScore.model_validate(data)
        assert s2 == s


# ---------------------------------------------------------------------------
# SegmentSentiment
# ---------------------------------------------------------------------------


class TestSegmentSentiment:
    def _scores(self):
        return [
            SentimentScore(label="positive", score=0.8),
            SentimentScore(label="neutral", score=0.15),
            SentimentScore(label="negative", score=0.05),
        ]

    def test_construction_with_all_fields(self):
        seg = SegmentSentiment(
            start=1.0,
            end=3.5,
            text=" Hello world",
            speaker="SPEAKER_00",
            scores=self._scores(),
            primary_sentiment="positive",
        )
        assert seg.start == 1.0
        assert seg.end == 3.5
        assert seg.text == " Hello world"
        assert seg.speaker == "SPEAKER_00"
        assert seg.primary_sentiment == "positive"
        assert len(seg.scores) == 3

    def test_optional_fields_default_to_none(self):
        seg = SegmentSentiment(
            start=0.0,
            end=1.0,
            scores=self._scores(),
            primary_sentiment="neutral",
        )
        assert seg.text is None
        assert seg.speaker is None

    def test_json_roundtrip(self):
        seg = SegmentSentiment(
            start=2.0,
            end=4.0,
            text="Test text",
            scores=self._scores(),
            primary_sentiment="positive",
        )
        data = json.loads(seg.model_dump_json())
        seg2 = SegmentSentiment.model_validate(data)
        assert seg2.start == seg.start
        assert seg2.primary_sentiment == seg.primary_sentiment
        assert len(seg2.scores) == 3


# ---------------------------------------------------------------------------
# SentimentResult
# ---------------------------------------------------------------------------


class TestSentimentResult:
    def _make(self):
        now = datetime.now(timezone.utc)
        return SentimentResult(
            transcription_file=Path("/audio/test/transcription/transcription_001.json"),
            segments=[
                SegmentSentiment(
                    start=0.0,
                    end=2.0,
                    text="Hello",
                    scores=[SentimentScore(label="positive", score=1.0)],
                    primary_sentiment="positive",
                )
            ],
            device="cpu",
            processing_time_seconds=0.5,
            started_at=now,
            completed_at=now,
        )

    def test_construction(self):
        r = self._make()
        assert r.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"
        assert r.device == "cpu"
        assert len(r.segments) == 1

    def test_json_roundtrip(self):
        r = self._make()
        data = json.loads(r.model_dump_json())
        r2 = SentimentResult.model_validate(data)
        assert r2.device == r.device
        assert r2.processing_time_seconds == r.processing_time_seconds
        assert len(r2.segments) == 1


# ---------------------------------------------------------------------------
# TranscriptSegment — backwards compat and sentiment field
# ---------------------------------------------------------------------------


class TestTranscriptSegmentSentimentField:
    def test_sentiment_defaults_to_none(self):
        seg = TranscriptSegment(text="Hello", start=0.0, end=1.0)
        assert seg.sentiment is None

    def test_sentiment_field_roundtrip(self):
        seg_sentiment = SegmentSentiment(
            start=0.0,
            end=1.0,
            text="Hello",
            scores=[SentimentScore(label="positive", score=0.9)],
            primary_sentiment="positive",
        )
        seg = TranscriptSegment(text="Hello", start=0.0, end=1.0, sentiment=seg_sentiment)
        data = json.loads(seg.model_dump_json())
        seg2 = TranscriptSegment.model_validate(data)
        assert seg2.sentiment is not None
        assert seg2.sentiment.primary_sentiment == "positive"

    def test_backwards_compat_json_without_sentiment(self):
        """Old transcription JSON (without sentiment field) should load cleanly."""
        raw = json.dumps({"text": "Hello", "start": 0.0, "end": 1.0})
        seg = TranscriptSegment.model_validate_json(raw)
        assert seg.sentiment is None


# ---------------------------------------------------------------------------
# TranscriptionResult — sentiment_applied flag
# ---------------------------------------------------------------------------


class TestTranscriptionResultSentimentApplied:
    def _minimal_tx_dict(self):
        now = datetime.now(timezone.utc).isoformat()
        return {
            "input_file": "/audio/test.wav",
            "input_info": {
                "path": "/audio/test.wav",
                "sample_rate": 44100,
                "channels": 1,
                "duration_seconds": 5.0,
                "frames": 220500,
                "format_str": "WAV",
                "subtype": "PCM_16",
            },
            "language": "en",
            "segments": [],
            "device": "cuda",
            "compute_type": "float16",
            "batch_size": 16,
            "processing_time_seconds": 1.0,
            "started_at": now,
            "completed_at": now,
        }

    def test_sentiment_applied_defaults_to_false(self):
        tx = TranscriptionResult.model_validate(self._minimal_tx_dict())
        assert tx.sentiment_applied is False

    def test_sentiment_applied_roundtrip(self):
        d = self._minimal_tx_dict()
        d["sentiment_applied"] = True
        tx = TranscriptionResult.model_validate(d)
        assert tx.sentiment_applied is True
        data = json.loads(tx.model_dump_json())
        assert data["sentiment_applied"] is True

    def test_backwards_compat_json_without_sentiment_applied(self):
        """Old transcription JSON lacking sentiment_applied should load with False."""
        tx = TranscriptionResult.model_validate(self._minimal_tx_dict())
        assert tx.sentiment_applied is False
