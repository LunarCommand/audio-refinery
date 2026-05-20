"""Tests for `src.service.transcript`.

Covers:
- CombinedTranscript assembly with and without sentiment
- BatchSummary assembly with all-success, all-failure, and mixed batches
- schema_version fields present on both documents
- JSON round-trip preserves every field
- model_versions block reflects the stages that ran
- BatchTotals derived correctly from job entries
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.models.audio import AudioFileInfo
from src.models.diarization import DiarizationResult, SpeakerSegment
from src.models.sentiment import (
    SegmentSentiment,
    SentimentResult,
    SentimentScore,
)
from src.models.transcription import (
    TranscriptionResult,
    TranscriptSegment,
    WordSegment,
)
from src.service.transcript import (
    BATCH_SUMMARY_SCHEMA_VERSION,
    TRANSCRIPT_SCHEMA_VERSION,
    BatchSummary,
    CombinedTranscript,
    JobSummaryEntry,
    build_combined,
    build_summary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_now() -> datetime:
    return datetime(2026, 5, 19, 22, 0, 0)


@pytest.fixture
def audio_info(tmp_path: Path) -> AudioFileInfo:
    return AudioFileInfo(
        path=tmp_path / "audio.wav",
        sample_rate=16000,
        channels=1,
        duration_seconds=180.0,
        frames=2_880_000,
        format_str="WAV",
        subtype="PCM_16",
    )


@pytest.fixture
def diarization_result(audio_info: AudioFileInfo, fixed_now: datetime) -> DiarizationResult:
    return DiarizationResult(
        input_file=audio_info.path,
        input_info=audio_info,
        segments=[
            SpeakerSegment(speaker_label="SPEAKER_00", start_seconds=0.0, end_seconds=90.0),
            SpeakerSegment(speaker_label="SPEAKER_01", start_seconds=90.0, end_seconds=180.0),
        ],
        num_speakers=2,
        device="cuda:0",
        processing_time_seconds=3.2,
        started_at=fixed_now,
        completed_at=fixed_now,
    )


@pytest.fixture
def transcription_result(audio_info: AudioFileInfo, fixed_now: datetime) -> TranscriptionResult:
    return TranscriptionResult(
        input_file=audio_info.path,
        input_info=audio_info,
        language="en",
        language_probability=0.98,
        segments=[
            TranscriptSegment(
                text="Hello world.",
                start=0.0,
                end=1.5,
                words=[
                    WordSegment(word="Hello", start=0.0, end=0.6, score=0.95),
                    WordSegment(word="world", start=0.7, end=1.5, score=0.92),
                ],
                speaker="SPEAKER_00",
            ),
        ],
        model_name="large-v3",
        device="cuda:0",
        compute_type="float16",
        batch_size=16,
        processing_time_seconds=4.5,
        started_at=fixed_now,
        completed_at=fixed_now,
    )


@pytest.fixture
def sentiment_result(audio_info: AudioFileInfo, fixed_now: datetime) -> SentimentResult:
    return SentimentResult(
        transcription_file=audio_info.path.with_suffix(".transcription.json"),
        segments=[
            SegmentSentiment(
                start=0.0,
                end=1.5,
                text="Hello world.",
                speaker="SPEAKER_00",
                scores=[
                    SentimentScore(label="positive", score=0.7),
                    SentimentScore(label="neutral", score=0.2),
                    SentimentScore(label="negative", score=0.1),
                ],
                primary_sentiment="positive",
            )
        ],
        device="cpu",
        processing_time_seconds=0.5,
        started_at=fixed_now,
        completed_at=fixed_now,
    )


# ---------------------------------------------------------------------------
# CombinedTranscript
# ---------------------------------------------------------------------------


def test_build_combined_includes_all_three_stages_when_sentiment_enabled(
    diarization_result, transcription_result, sentiment_result, fixed_now
):
    combined = build_combined(
        diarization_result,
        transcription_result,
        sentiment_result,
        processed_at=fixed_now,
    )

    assert combined.schema_version == TRANSCRIPT_SCHEMA_VERSION
    assert combined.schema_version == "1.0.0"
    assert combined.audio == diarization_result.input_info
    assert combined.diarization is diarization_result
    assert combined.transcription is transcription_result
    assert combined.sentiment is sentiment_result
    assert combined.processed_at == fixed_now
    assert combined.model_versions == {
        "diarization": "pyannote/speaker-diarization-3.1",
        "transcription": "large-v3",
        "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    }


def test_build_combined_omits_sentiment_when_disabled(diarization_result, transcription_result, fixed_now):
    combined = build_combined(diarization_result, transcription_result, processed_at=fixed_now)

    assert combined.sentiment is None
    assert "sentiment" not in combined.model_versions
    assert combined.model_versions["diarization"] == "pyannote/speaker-diarization-3.1"
    assert combined.model_versions["transcription"] == "large-v3"


def test_build_combined_audio_refinery_version_populated(diarization_result, transcription_result, fixed_now):
    combined = build_combined(diarization_result, transcription_result, processed_at=fixed_now)
    # Either the real installed version or our fallback string — never empty.
    assert combined.audio_refinery_version
    assert isinstance(combined.audio_refinery_version, str)


def test_combined_transcript_round_trips_through_json(
    diarization_result, transcription_result, sentiment_result, fixed_now
):
    combined = build_combined(
        diarization_result,
        transcription_result,
        sentiment_result,
        processed_at=fixed_now,
    )
    raw = combined.model_dump_json()
    reloaded = CombinedTranscript.model_validate_json(raw)

    assert reloaded.schema_version == "1.0.0"
    assert reloaded.audio.sample_rate == 16000
    assert reloaded.diarization.num_speakers == 2
    assert reloaded.transcription.language == "en"
    assert reloaded.sentiment is not None
    assert reloaded.sentiment.segments[0].primary_sentiment == "positive"
    assert json.loads(raw)["schema_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# JobSummaryEntry + BatchSummary
# ---------------------------------------------------------------------------


def _success_entry(job_id: str, started: datetime, completed: datetime) -> JobSummaryEntry:
    return JobSummaryEntry(
        job_id=job_id,
        input_uri="file:///inbox/a.wav",
        output_uri="file:///outbox/a.json",
        status="completed",
        started_at=started,
        completed_at=completed,
        duration_seconds=(completed - started).total_seconds(),
    )


def _failure_entry(
    job_id: str,
    started: datetime,
    failed: datetime,
    *,
    stage: str = "transcribe",
    error: str = "ValueError: bad audio",
    retryable: bool = False,
) -> JobSummaryEntry:
    return JobSummaryEntry(
        job_id=job_id,
        input_uri="file:///inbox/b.wav",
        output_uri="file:///outbox/b.json",
        status="failed",
        started_at=started,
        failed_at=failed,
        stage=stage,  # type: ignore[arg-type]
        error=error,
        retryable=retryable,
    )


def test_build_summary_all_success_totals(fixed_now: datetime):
    later = datetime(2026, 5, 19, 22, 5, 0)
    entries = [_success_entry("rfj_a", fixed_now, later), _success_entry("rfj_b", fixed_now, later)]

    summary = build_summary(
        batch_id="btc_xyz",
        submitted_at=fixed_now,
        completed_at=later,
        jobs=entries,
    )

    assert summary.schema_version == BATCH_SUMMARY_SCHEMA_VERSION == "1.0.0"
    assert summary.batch_id == "btc_xyz"
    assert summary.totals.submitted == 2
    assert summary.totals.completed == 2
    assert summary.totals.failed == 0
    assert [j.job_id for j in summary.jobs] == ["rfj_a", "rfj_b"]


def test_build_summary_all_failure_still_produces_summary(fixed_now: datetime):
    later = datetime(2026, 5, 19, 22, 5, 0)
    entries = [
        _failure_entry("rfj_a", fixed_now, later, stage="download", retryable=True),
        _failure_entry("rfj_b", fixed_now, later, stage="transcribe", retryable=False),
    ]

    summary = build_summary("btc_xyz", fixed_now, later, entries)

    assert summary.totals.submitted == 2
    assert summary.totals.completed == 0
    assert summary.totals.failed == 2
    assert summary.jobs[0].stage == "download"
    assert summary.jobs[0].retryable is True
    assert summary.jobs[1].retryable is False


def test_build_summary_mixed_outcomes(fixed_now: datetime):
    later = datetime(2026, 5, 19, 22, 5, 0)
    entries = [
        _success_entry("rfj_a", fixed_now, later),
        _failure_entry("rfj_b", fixed_now, later),
        _success_entry("rfj_c", fixed_now, later),
    ]

    summary = build_summary("btc_xyz", fixed_now, later, entries)

    assert summary.totals.submitted == 3
    assert summary.totals.completed == 2
    assert summary.totals.failed == 1
    # Submission order preserved
    assert [j.job_id for j in summary.jobs] == ["rfj_a", "rfj_b", "rfj_c"]
    assert summary.jobs[0].status == "completed"
    assert summary.jobs[1].status == "failed"
    assert summary.jobs[2].status == "completed"


def test_batch_summary_round_trips_through_json(fixed_now: datetime):
    later = datetime(2026, 5, 19, 22, 5, 0)
    summary = build_summary(
        "btc_xyz",
        fixed_now,
        later,
        [_success_entry("rfj_a", fixed_now, later), _failure_entry("rfj_b", fixed_now, later)],
    )

    raw = summary.model_dump_json()
    reloaded = BatchSummary.model_validate_json(raw)

    assert reloaded.schema_version == "1.0.0"
    assert reloaded.batch_id == "btc_xyz"
    assert len(reloaded.jobs) == 2
    assert reloaded.jobs[1].status == "failed"
    assert reloaded.jobs[1].stage == "transcribe"
    assert json.loads(raw)["totals"] == {"submitted": 2, "completed": 1, "failed": 1}


def test_failure_entry_carries_required_failure_fields(fixed_now: datetime):
    later = datetime(2026, 5, 19, 22, 0, 30)
    entry = _failure_entry("rfj_x", fixed_now, later, stage="upload", retryable=True)

    assert entry.failed_at == later
    assert entry.stage == "upload"
    assert entry.retryable is True
    assert entry.completed_at is None
    assert entry.duration_seconds is None


def test_success_entry_carries_required_success_fields(fixed_now: datetime):
    later = datetime(2026, 5, 19, 22, 1, 30)
    entry = _success_entry("rfj_x", fixed_now, later)

    assert entry.completed_at == later
    assert entry.duration_seconds == pytest.approx(90.0)
    assert entry.failed_at is None
    assert entry.stage is None
    assert entry.error is None
    assert entry.retryable is None


def test_empty_jobs_list_produces_empty_summary(fixed_now: datetime):
    """Not a real production case (the API rejects empty batches at the boundary),
    but the factory shouldn't blow up if given one."""
    summary = build_summary("btc_empty", fixed_now, fixed_now, [])
    assert summary.totals.submitted == 0
    assert summary.totals.completed == 0
    assert summary.totals.failed == 0
    assert summary.jobs == []
