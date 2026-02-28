"""Text sentiment analysis for the audio-refinery pipeline.

Runs a HuggingFace text-classification pipeline (default:
cardiffnlp/twitter-roberta-base-sentiment-latest) on each segment in a
TranscriptionResult JSON to produce per-segment sentiment scores.

This is a text-only step — no audio file or vocals.wav is required.
"""

from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

from src.models.sentiment import SegmentSentiment, SentimentResult, SentimentScore
from src.models.transcription import TranscriptionResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_DEVICE = "cpu"


class SentimentError(Exception):
    """Raised when sentiment analysis fails."""


def _parse_device(device: str) -> int | str:
    """Convert device string to the form expected by HuggingFace pipeline.

    'cuda:N' → N (int), 'cuda' → 0, 'cpu' → 'cpu'.
    """
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    return device


def load_sentiment_pipeline(model: str, device: str):
    """Load a HuggingFace text-classification pipeline.

    Raises:
        SentimentError: If transformers is not installed.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise SentimentError(f"transformers not available — install it with: pip install transformers: {exc}") from exc

    # Silence noisy model checkpoint warnings from transformers that the user cannot act on.
    logging.getLogger("transformers").setLevel(logging.ERROR)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return hf_pipeline("text-classification", model=model, device=_parse_device(device))


def analyze_sentiment(
    transcription_file: Path,
    device: str = DEFAULT_DEVICE,
    model: str = DEFAULT_MODEL,
    _sentiment_pipeline=None,
) -> SentimentResult:
    """Run sentiment classification on all non-empty segments in a transcription.

    Args:
        transcription_file: Path to a TranscriptionResult JSON produced by
            ``audio-refinery transcribe``. No audio file is required.
        device: Compute device ('cpu', 'cuda', or 'cuda:N'). Defaults to 'cpu'.
        model: HuggingFace text-classification model name.
        _sentiment_pipeline: Pre-loaded HuggingFace pipeline. When provided,
            ``load_sentiment_pipeline`` is not called — used by the batch pipeline
            runner to reuse a single loaded model across all files.

    Returns:
        SentimentResult with per-segment scores and provenance metadata.

    Raises:
        FileNotFoundError: If transcription_file does not exist.
        SentimentError: If JSON parsing fails, no usable text exists, or all
            segments fail classification.
    """
    if not transcription_file.exists():
        raise FileNotFoundError(f"Transcription file not found: {transcription_file}")

    try:
        transcription = TranscriptionResult.model_validate_json(transcription_file.read_text())
    except Exception as exc:
        raise SentimentError(f"Failed to parse transcription JSON: {exc}") from exc

    pipeline = _sentiment_pipeline
    if pipeline is None:
        pipeline = load_sentiment_pipeline(model, device)

    started_at = datetime.now(timezone.utc)
    t0 = time.monotonic()

    segment_results: list[SegmentSentiment] = []

    for seg in transcription.segments:
        if not seg.text or not seg.text.strip():
            continue
        try:
            raw_scores = pipeline(seg.text, top_k=None)
            sorted_scores = sorted(raw_scores, key=lambda x: x["score"], reverse=True)
            scores = [SentimentScore(label=s["label"], score=s["score"]) for s in sorted_scores]
            segment_results.append(
                SegmentSentiment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker=seg.speaker,
                    scores=scores,
                    primary_sentiment=scores[0].label,
                )
            )
        except Exception as exc:
            logger.warning("Sentiment analysis failed for segment at %.2fs: %s", seg.start, exc)

    processing_time = round(time.monotonic() - t0, 3)
    completed_at = datetime.now(timezone.utc)

    if not segment_results:
        if not any(seg.text and seg.text.strip() for seg in transcription.segments):
            raise SentimentError("No usable text found in transcription segments")
        raise SentimentError("All segments failed sentiment analysis")

    return SentimentResult(
        transcription_file=transcription_file,
        segments=segment_results,
        model_name=model,
        device=device,
        processing_time_seconds=processing_time,
        started_at=started_at,
        completed_at=completed_at,
    )


def merge_sentiment_into_transcription(
    transcription_file: Path,
    sentiment_result: SentimentResult,
) -> None:
    """Update the transcription JSON in place with sentiment data for each segment.

    Reads the TranscriptionResult from ``transcription_file``, populates
    ``sentiment`` on each ``TranscriptSegment`` that appears in
    ``sentiment_result`` (matched by ``start`` timestamp), sets
    ``sentiment_applied = True``, and writes the updated result back.

    Args:
        transcription_file: Path to the TranscriptionResult JSON to update.
        sentiment_result: The SentimentResult produced by ``analyze_sentiment``.
    """
    transcription = TranscriptionResult.model_validate_json(transcription_file.read_text())

    lookup = {seg.start: seg for seg in sentiment_result.segments}

    for seg in transcription.segments:
        seg.sentiment = lookup.get(seg.start)

    transcription.sentiment_applied = True
    transcription_file.write_text(transcription.model_dump_json(indent=2))
