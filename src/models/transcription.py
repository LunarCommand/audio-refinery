"""Transcription data models for audio-refinery."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from src.models.audio import AudioFileInfo
from src.models.sentiment import SegmentSentiment


class WordSegment(BaseModel):
    """A single word with timing and optional speaker assignment."""

    word: str = Field(description="Transcribed word (may include surrounding whitespace)")
    start: float
    end: float
    score: float | None = None
    speaker: str | None = None


class TranscriptSegment(BaseModel):
    """A sentence or clause boundary segment with word-level breakdown."""

    text: str
    start: float
    end: float
    words: list[WordSegment] = Field(default_factory=list)
    speaker: str | None = None
    sentiment: SegmentSentiment | None = None


class TranscriptionResult(BaseModel):
    """Full provenance record of a WhisperX transcription run."""

    # Input
    input_file: Path
    input_info: AudioFileInfo

    # Output
    language: str = Field(description="Language code, e.g. 'en'")
    language_probability: float | None = None
    segments: list[TranscriptSegment]

    # Processing metadata
    model_name: str = "large-v3"
    device: str
    compute_type: str
    batch_size: int
    processing_time_seconds: float
    started_at: datetime
    completed_at: datetime
    diarization_applied: bool = False
    diarization_file: Path | None = None
    alignment_fallback: bool = False  # True if Wav2Vec2 alignment failed and raw timestamps were used
    sentiment_applied: bool = False
