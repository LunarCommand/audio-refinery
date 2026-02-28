"""Unit tests for diarization Pydantic models."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models.audio import AudioFileInfo
from src.models.diarization import DiarizationResult, SpeakerSegment


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


class TestSpeakerSegment:
    def test_construction(self):
        seg = SpeakerSegment(speaker_label="SPEAKER_00", start_seconds=1.0, end_seconds=3.5)
        assert seg.speaker_label == "SPEAKER_00"
        assert seg.start_seconds == 1.0
        assert seg.end_seconds == 3.5

    def test_duration_computed(self):
        seg = SpeakerSegment(speaker_label="SPEAKER_01", start_seconds=2.0, end_seconds=5.0)
        assert seg.duration_seconds == 3.0

    def test_duration_precision(self):
        seg = SpeakerSegment(speaker_label="SPEAKER_00", start_seconds=0.0, end_seconds=1.2345678)
        assert seg.duration_seconds == round(1.2345678, 6)

    def test_json_roundtrip(self):
        seg = SpeakerSegment(speaker_label="SPEAKER_02", start_seconds=0.5, end_seconds=4.25)
        data = json.loads(seg.model_dump_json())
        restored = SpeakerSegment(**data)
        assert restored.speaker_label == seg.speaker_label
        assert restored.start_seconds == seg.start_seconds
        assert restored.end_seconds == seg.end_seconds
        assert restored.duration_seconds == seg.duration_seconds


class TestDiarizationResult:
    def test_construction(self, sample_audio_info):
        now = datetime.now(timezone.utc)
        segments = [
            SpeakerSegment(speaker_label="SPEAKER_00", start_seconds=0.0, end_seconds=2.0),
            SpeakerSegment(speaker_label="SPEAKER_01", start_seconds=2.5, end_seconds=5.0),
        ]
        result = DiarizationResult(
            input_file=Path("/tmp/vocals.wav"),
            input_info=sample_audio_info,
            segments=segments,
            num_speakers=2,
            processing_time_seconds=3.14,
            started_at=now,
            completed_at=now,
        )
        assert result.num_speakers == 2
        assert len(result.segments) == 2
        assert result.model_name == "pyannote/speaker-diarization-3.1"
        assert result.device == "cuda"
        assert result.min_speakers is None
        assert result.max_speakers is None

    def test_speaker_hint_fields(self, sample_audio_info):
        now = datetime.now(timezone.utc)
        result = DiarizationResult(
            input_file=Path("/tmp/vocals.wav"),
            input_info=sample_audio_info,
            segments=[],
            num_speakers=0,
            processing_time_seconds=1.0,
            started_at=now,
            completed_at=now,
            min_speakers=2,
            max_speakers=5,
        )
        assert result.min_speakers == 2
        assert result.max_speakers == 5

    def test_json_roundtrip(self, sample_audio_info):
        now = datetime.now(timezone.utc)
        segments = [
            SpeakerSegment(speaker_label="SPEAKER_00", start_seconds=1.0, end_seconds=3.0),
        ]
        result = DiarizationResult(
            input_file=Path("/tmp/vocals.wav"),
            input_info=sample_audio_info,
            segments=segments,
            num_speakers=1,
            processing_time_seconds=2.5,
            started_at=now,
            completed_at=now,
        )
        data = json.loads(result.model_dump_json())
        restored = DiarizationResult(**data)
        assert restored.num_speakers == result.num_speakers
        assert len(restored.segments) == 1
        assert restored.segments[0].speaker_label == "SPEAKER_00"
        assert restored.processing_time_seconds == result.processing_time_seconds
