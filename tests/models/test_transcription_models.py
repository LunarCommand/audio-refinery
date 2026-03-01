"""Unit tests for transcription Pydantic models."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.models.audio import AudioFileInfo
from src.models.transcription import TranscriptionResult, TranscriptSegment, WordSegment


@pytest.fixture
def sample_audio_info():
    return AudioFileInfo(
        path=Path("/tmp/vocals.wav"),
        sample_rate=44100,
        channels=1,
        duration_seconds=10.0,
        frames=441000,
        format_str="WAV",
        subtype="PCM_16",
    )


class TestWordSegment:
    def test_construction(self):
        w = WordSegment(word="hello", start=1.0, end=1.5)
        assert w.word == "hello"
        assert w.start == 1.0
        assert w.end == 1.5
        assert w.score is None
        assert w.speaker is None

    def test_optional_fields(self):
        w = WordSegment(word="world", start=2.0, end=2.4, score=0.98, speaker="SPEAKER_00")
        assert w.score == 0.98
        assert w.speaker == "SPEAKER_00"

    def test_json_roundtrip(self):
        w = WordSegment(word="test", start=0.5, end=0.9, score=0.75, speaker="SPEAKER_01")
        data = json.loads(w.model_dump_json())
        restored = WordSegment(**data)
        assert restored.word == w.word
        assert restored.start == w.start
        assert restored.end == w.end
        assert restored.score == w.score
        assert restored.speaker == w.speaker

    def test_json_roundtrip_none_fields(self):
        w = WordSegment(word="hi", start=0.0, end=0.3)
        data = json.loads(w.model_dump_json())
        restored = WordSegment(**data)
        assert restored.score is None
        assert restored.speaker is None


class TestTranscriptSegment:
    def test_construction_minimal(self):
        seg = TranscriptSegment(text="Hello world.", start=0.0, end=1.2)
        assert seg.text == "Hello world."
        assert seg.start == 0.0
        assert seg.end == 1.2
        assert seg.words == []
        assert seg.speaker is None

    def test_construction_with_words(self):
        words = [
            WordSegment(word="Hello", start=0.0, end=0.5),
            WordSegment(word="world", start=0.6, end=1.0),
        ]
        seg = TranscriptSegment(text="Hello world.", start=0.0, end=1.2, words=words, speaker="SPEAKER_00")
        assert len(seg.words) == 2
        assert seg.speaker == "SPEAKER_00"

    def test_json_roundtrip(self):
        words = [WordSegment(word="Quick", start=1.0, end=1.4, score=0.9)]
        seg = TranscriptSegment(text="Quick.", start=1.0, end=1.6, words=words)
        data = json.loads(seg.model_dump_json())
        restored = TranscriptSegment(**data)
        assert restored.text == seg.text
        assert len(restored.words) == 1
        assert restored.words[0].word == "Quick"
        assert restored.words[0].score == 0.9


class TestTranscriptionResult:
    def test_construction(self, sample_audio_info):
        now = datetime.now(UTC)
        segments = [
            TranscriptSegment(
                text="Spider-Man swings into action.",
                start=4.0,
                end=6.5,
                words=[WordSegment(word="Spider-Man", start=4.0, end=4.8)],
            )
        ]
        result = TranscriptionResult(
            input_file=Path("/tmp/vocals.wav"),
            input_info=sample_audio_info,
            language="en",
            segments=segments,
            model_name="large-v3",
            device="cuda",
            compute_type="float16",
            batch_size=16,
            processing_time_seconds=12.5,
            started_at=now,
            completed_at=now,
        )
        assert result.language == "en"
        assert result.model_name == "large-v3"
        assert result.device == "cuda"
        assert result.compute_type == "float16"
        assert result.batch_size == 16
        assert len(result.segments) == 1
        assert result.diarization_applied is False
        assert result.diarization_file is None
        assert result.language_probability is None

    def test_optional_fields(self, sample_audio_info):
        now = datetime.now(UTC)
        result = TranscriptionResult(
            input_file=Path("/tmp/vocals.wav"),
            input_info=sample_audio_info,
            language="en",
            language_probability=0.99,
            segments=[],
            device="cuda",
            compute_type="float16",
            batch_size=16,
            processing_time_seconds=5.0,
            started_at=now,
            completed_at=now,
            diarization_applied=True,
            diarization_file=Path("/tmp/diarization.json"),
        )
        assert result.language_probability == 0.99
        assert result.diarization_applied is True
        assert result.diarization_file == Path("/tmp/diarization.json")

    def test_json_roundtrip(self, sample_audio_info):
        now = datetime.now(UTC)
        words = [WordSegment(word="Quickly", start=0.0, end=0.5, score=0.95, speaker="SPEAKER_00")]
        segments = [TranscriptSegment(text="Quickly!", start=0.0, end=0.8, words=words, speaker="SPEAKER_00")]
        result = TranscriptionResult(
            input_file=Path("/tmp/vocals.wav"),
            input_info=sample_audio_info,
            language="en",
            language_probability=0.98,
            segments=segments,
            device="cuda",
            compute_type="float16",
            batch_size=16,
            processing_time_seconds=8.2,
            started_at=now,
            completed_at=now,
            diarization_applied=True,
            diarization_file=Path("/tmp/diarization.json"),
        )
        data = json.loads(result.model_dump_json())
        restored = TranscriptionResult(**data)
        assert restored.language == "en"
        assert restored.language_probability == 0.98
        assert len(restored.segments) == 1
        assert restored.segments[0].text == "Quickly!"
        assert restored.segments[0].words[0].word == "Quickly"
        assert restored.segments[0].words[0].speaker == "SPEAKER_00"
        assert restored.diarization_applied is True
        assert restored.processing_time_seconds == 8.2
