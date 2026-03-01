"""Diarization data models for audio-refinery."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, computed_field

from src.models.audio import AudioFileInfo


class SpeakerSegment(BaseModel):
    """A single speaker turn detected by diarization."""

    speaker_label: str = Field(description="Speaker cluster label, e.g. 'SPEAKER_00'")
    start_seconds: float
    end_seconds: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_seconds(self) -> float:
        return round(self.end_seconds - self.start_seconds, 6)


class DiarizationResult(BaseModel):
    """Full provenance record of a Pyannote diarization run."""

    # Input
    input_file: Path
    input_info: AudioFileInfo

    # Output
    segments: list[SpeakerSegment]
    num_speakers: int = Field(description="Number of unique speaker clusters detected")

    # Processing metadata
    model_name: str = "pyannote/speaker-diarization-3.1"
    device: str = "cuda"
    processing_time_seconds: float
    started_at: datetime
    completed_at: datetime
    min_speakers: int | None = None
    max_speakers: int | None = None
