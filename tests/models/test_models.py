"""Tests for shared Pydantic models."""

import json
from datetime import UTC, datetime
from pathlib import Path

from src.models.audio import AudioFileInfo, SeparationResult


def _make_audio_info(**overrides) -> AudioFileInfo:
    defaults = {
        "path": Path("/audio/test.wav"),
        "sample_rate": 44100,
        "channels": 2,
        "duration_seconds": 180.5,
        "frames": 7960050,
        "format_str": "WAV",
        "subtype": "PCM_16",
    }
    defaults.update(overrides)
    return AudioFileInfo(**defaults)  # type: ignore[arg-type]


class TestAudioFileInfo:
    def test_construction(self):
        info = _make_audio_info()
        assert info.sample_rate == 44100
        assert info.channels == 2
        assert info.duration_seconds == 180.5
        assert info.frames == 7960050
        assert info.format_str == "WAV"
        assert info.subtype == "PCM_16"
        assert info.path == Path("/audio/test.wav")

    def test_json_roundtrip(self):
        info = _make_audio_info()
        json_str = info.model_dump_json()
        restored = AudioFileInfo.model_validate_json(json_str)
        assert restored == info

    def test_path_coercion(self):
        info = _make_audio_info(path="/some/string/path.wav")
        assert isinstance(info.path, Path)
        assert info.path == Path("/some/string/path.wav")


class TestSeparationResult:
    def _make_result(self, **overrides) -> SeparationResult:
        now = datetime.now(UTC)
        defaults = {
            "input_file": Path("/audio/side_a.wav"),
            "input_info": _make_audio_info(),
            "vocals_path": Path("/output/htdemucs/side_a/vocals.wav"),
            "no_vocals_path": Path("/output/htdemucs/side_a/no_vocals.wav"),
            "output_dir": Path("/output"),
            "model_name": "htdemucs",
            "device": "cuda",
            "processing_time_seconds": 42.3,
            "started_at": now,
            "completed_at": now,
        }
        defaults.update(overrides)
        return SeparationResult(**defaults)  # type: ignore[arg-type]

    def test_construction(self):
        result = self._make_result()
        assert result.model_name == "htdemucs"
        assert result.device == "cuda"
        assert result.two_stems == "vocals"
        assert result.segment is None

    def test_defaults(self):
        result = self._make_result()
        assert result.model_name == "htdemucs"
        assert result.device == "cuda"
        assert result.two_stems == "vocals"
        assert result.segment is None

    def test_optional_segment(self):
        result = self._make_result(segment=40)
        assert result.segment == 40

    def test_json_roundtrip(self):
        result = self._make_result(segment=40)
        json_str = result.model_dump_json()
        restored = SeparationResult.model_validate_json(json_str)
        assert restored == result

    def test_json_contains_nested_info(self):
        result = self._make_result()
        data = json.loads(result.model_dump_json())
        assert "input_info" in data
        assert data["input_info"]["sample_rate"] == 44100
