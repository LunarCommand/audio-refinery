"""Sentiment analysis data models for audio-refinery."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class SentimentScore(BaseModel):
    """A single sentiment class score from the classifier."""

    label: str
    score: float


class SegmentSentiment(BaseModel):
    """Sentiment analysis result for a single transcript segment."""

    start: float
    end: float
    text: str | None = None
    speaker: str | None = None
    scores: list[SentimentScore]
    primary_sentiment: str


class SentimentResult(BaseModel):
    """Full provenance record of a sentiment analysis run."""

    transcription_file: Path
    segments: list[SegmentSentiment]
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    device: str
    processing_time_seconds: float
    started_at: datetime
    completed_at: datetime
