"""Integration tests — real Demucs, Pyannote, WhisperX, and sentiment on GPU.

These tests require:
- Demucs installed and on PATH
- A GPU (or CPU with patience)
- A test audio file at the path below
- For diarization: HF_TOKEN set in environment or .env
- For transcription: whisperx installed (see README for install steps)
- For sentiment: transformers installed

Run with: make test-refinery-integration
"""

from pathlib import Path

import pytest

from src.diarizer import diarize
from src.sentiment_analyzer import analyze_sentiment, merge_sentiment_into_transcription
from src.separator import separate
from src.transcriber import transcribe

TEST_AUDIO_PATH = Path("/mnt/fast_scratch/test_fixtures/test_audio.wav")


pytestmark = pytest.mark.integration


@pytest.fixture
def integration_audio():
    if not TEST_AUDIO_PATH.exists():
        pytest.skip(f"Test audio not found: {TEST_AUDIO_PATH}")
    return TEST_AUDIO_PATH


def test_real_separation(integration_audio, tmp_path):
    """Run real Demucs separation and verify output files exist."""
    result = separate(
        input_file=integration_audio,
        output_dir=tmp_path / "demucs_output",
        device="cuda",
    )

    assert result.vocals_path.exists()
    assert result.no_vocals_path.exists()
    assert result.vocals_path.stat().st_size > 0
    assert result.no_vocals_path.stat().st_size > 0
    assert result.processing_time_seconds > 0
    assert result.model_name == "htdemucs"


@pytest.fixture
def integration_vocals(integration_audio, tmp_path):
    """Produce a vocals stem via real Demucs for use in diarization tests."""
    result = separate(
        input_file=integration_audio,
        output_dir=tmp_path / "demucs_output",
        device="cuda",
    )
    return result.vocals_path


def test_real_diarization(integration_vocals):
    """Run real Pyannote diarization on the Ghost Track and verify output."""
    result = diarize(
        input_file=integration_vocals,
        device="cuda",
    )

    assert result.num_speakers >= 1
    assert len(result.segments) > 0
    assert all(seg.duration_seconds > 0 for seg in result.segments)
    assert all(seg.end_seconds > seg.start_seconds for seg in result.segments)
    assert result.processing_time_seconds > 0
    assert result.model_name == "pyannote/speaker-diarization-3.1"


def test_real_transcription(integration_vocals):
    """Run real WhisperX transcription on the Ghost Track and verify output."""
    result = transcribe(
        input_file=integration_vocals,
        device="cuda",
        compute_type="float16",
        batch_size=16,
        language="en",
    )

    assert len(result.segments) > 0
    assert result.language == "en"
    assert result.processing_time_seconds > 0
    assert result.model_name == "large-v3"
    # At least some words should have alignment timestamps
    all_words = [w for seg in result.segments for w in seg.words]
    assert any(w.start > 0 for w in all_words)


def test_real_transcription_with_diarization(integration_vocals, tmp_path):
    """Run transcription with speaker assignment from a prior diarization pass."""
    # Step 1: diarize
    diar_result = diarize(input_file=integration_vocals, device="cuda")
    diar_json = tmp_path / "diarization.json"
    diar_json.write_text(diar_result.model_dump_json())

    # Step 2: transcribe with speaker merge
    tx_result = transcribe(
        input_file=integration_vocals,
        device="cuda",
        diarization_file=diar_json,
    )

    assert tx_result.diarization_applied is True
    assert len(tx_result.segments) > 0
    # At least some segments should have speaker labels after merging
    labeled = [s for s in tx_result.segments if s.speaker is not None]
    assert len(labeled) > 0


@pytest.fixture
def integration_transcription(integration_vocals, tmp_path):
    """Produce a transcription JSON via real WhisperX for use in sentiment tests."""
    tx_result = transcribe(
        input_file=integration_vocals,
        device="cuda",
        compute_type="float16",
        batch_size=16,
        language="en",
    )
    tx_json = tmp_path / "transcription_001.json"
    tx_json.write_text(tx_result.model_dump_json(indent=2))
    return tx_json


def test_real_sentiment(integration_transcription, tmp_path):
    """Run real HuggingFace sentiment analysis on a transcription and verify output."""
    result = analyze_sentiment(integration_transcription, device="cpu")

    assert len(result.segments) > 0
    valid_labels = {"positive", "neutral", "negative"}
    assert all(seg.primary_sentiment in valid_labels for seg in result.segments)
    assert result.processing_time_seconds >= 0

    merge_sentiment_into_transcription(integration_transcription, result)

    from src.models.transcription import TranscriptionResult

    reloaded = TranscriptionResult.model_validate_json(integration_transcription.read_text())
    assert reloaded.sentiment_applied is True
    non_empty = [s for s in reloaded.segments if s.text and s.text.strip()]
    assert any(s.sentiment is not None for s in non_empty)
