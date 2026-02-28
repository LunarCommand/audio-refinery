"""Audio data models for audio-refinery."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class AudioFileInfo(BaseModel):
    """Metadata about an audio file, typically from soundfile.info()."""

    path: Path
    sample_rate: int
    channels: int
    duration_seconds: float
    frames: int
    format_str: str = Field(description="Audio format, e.g. 'WAV'")
    subtype: str = Field(description="Audio subtype, e.g. 'PCM_16'")


class SeparationResult(BaseModel):
    """Full provenance record of a Demucs vocal separation run."""

    # Input
    input_file: Path
    input_info: AudioFileInfo

    # Output
    vocals_path: Path
    no_vocals_path: Path
    output_dir: Path

    # Processing metadata
    model_name: str = "htdemucs"
    device: str = "cuda"
    processing_time_seconds: float
    started_at: datetime
    completed_at: datetime
    segment: int | None = None
    two_stems: str = "vocals"
