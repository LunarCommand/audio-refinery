"""CLI tests for the `audio-refinery transcribe` subcommand."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.cli import cli
from src.models.audio import AudioFileInfo
from src.models.transcription import TranscriptionResult, TranscriptSegment, WordSegment
from src.transcriber import TranscriptionError


@pytest.fixture
def sample_transcription_result(fake_wav):
    now = datetime.now(timezone.utc)
    info = AudioFileInfo(
        path=fake_wav,
        sample_rate=44100,
        channels=1,
        duration_seconds=10.0,
        frames=441000,
        format_str="WAV",
        subtype="PCM_16",
    )
    segments = [
        TranscriptSegment(
            text=" Look out Spider-Man!",
            start=1.0,
            end=3.5,
            words=[
                WordSegment(word=" Look", start=1.0, end=1.3, score=0.95),
                WordSegment(word=" out", start=1.4, end=1.6, score=0.92),
                WordSegment(word=" Spider-Man", start=1.7, end=2.5, score=0.88),
            ],
        ),
        TranscriptSegment(
            text=" Quickly! We must get to the laboratory.",
            start=4.0,
            end=7.2,
            words=[
                WordSegment(word=" Quickly", start=4.0, end=4.5, score=0.91),
            ],
        ),
    ]
    return TranscriptionResult(
        input_file=fake_wav,
        input_info=info,
        language="en",
        language_probability=0.99,
        segments=segments,
        device="cuda",
        compute_type="float16",
        batch_size=16,
        processing_time_seconds=8.34,
        started_at=now,
        completed_at=now,
    )


@pytest.fixture
def sample_result_with_speakers(fake_wav):
    now = datetime.now(timezone.utc)
    info = AudioFileInfo(
        path=fake_wav,
        sample_rate=44100,
        channels=1,
        duration_seconds=10.0,
        frames=441000,
        format_str="WAV",
        subtype="PCM_16",
    )
    return TranscriptionResult(
        input_file=fake_wav,
        input_info=info,
        language="en",
        segments=[
            TranscriptSegment(
                text=" I am Spider-Man.",
                start=0.0,
                end=2.0,
                speaker="SPEAKER_00",
                words=[WordSegment(word=" I", start=0.0, end=0.3, speaker="SPEAKER_00")],
            )
        ],
        device="cuda",
        compute_type="float16",
        batch_size=16,
        processing_time_seconds=5.0,
        started_at=now,
        completed_at=now,
        diarization_applied=True,
        diarization_file=Path("/tmp/diarization.json"),
    )


class TestTranscribeCommand:
    def test_happy_path(self, fake_wav, sample_transcription_result):
        runner = CliRunner()
        with patch("src.cli.transcribe", return_value=sample_transcription_result):
            result = runner.invoke(cli, ["transcribe", str(fake_wav)])

        assert result.exit_code == 0, result.output
        assert "Transcription Complete" in result.output
        assert "Transcript Preview" in result.output
        assert "Look out Spider-Man" in result.output

    def test_transcription_error_exits_nonzero(self, fake_wav):
        runner = CliRunner()
        with patch("src.cli.transcribe", side_effect=TranscriptionError("whisperx is not installed")):
            result = runner.invoke(cli, ["transcribe", str(fake_wav)])

        assert result.exit_code == 1
        assert "whisperx is not installed" in result.output

    def test_file_not_found_exits_nonzero(self, tmp_path):
        runner = CliRunner()
        missing = tmp_path / "no_such_file.wav"
        result = runner.invoke(cli, ["transcribe", str(missing)])
        assert result.exit_code != 0

    def test_output_file_written(self, fake_wav, sample_transcription_result, tmp_path):
        out_file = tmp_path / "result.json"
        runner = CliRunner()
        with patch("src.cli.transcribe", return_value=sample_transcription_result):
            result = runner.invoke(cli, ["transcribe", str(fake_wav), "-o", str(out_file)])

        assert result.exit_code == 0, result.output
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["language"] == "en"
        assert len(data["segments"]) == 2

    def test_no_output_file_prints_json_panel(self, fake_wav, sample_transcription_result):
        runner = CliRunner()
        with patch("src.cli.transcribe", return_value=sample_transcription_result):
            result = runner.invoke(cli, ["transcribe", str(fake_wav)])

        assert result.exit_code == 0, result.output
        assert "TranscriptionResult JSON" in result.output

    def test_device_option_forwarded(self, fake_wav, sample_transcription_result):
        runner = CliRunner()
        with patch("src.cli.transcribe", return_value=sample_transcription_result) as mock_tx:
            result = runner.invoke(cli, ["transcribe", str(fake_wav), "-d", "cpu"])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_tx.call_args
        assert kwargs.get("device") == "cpu"

    def test_compute_type_forwarded(self, fake_wav, sample_transcription_result):
        runner = CliRunner()
        with patch("src.cli.transcribe", return_value=sample_transcription_result) as mock_tx:
            result = runner.invoke(cli, ["transcribe", str(fake_wav), "--compute-type", "int8"])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_tx.call_args
        assert kwargs.get("compute_type") == "int8"

    def test_batch_size_forwarded(self, fake_wav, sample_transcription_result):
        runner = CliRunner()
        with patch("src.cli.transcribe", return_value=sample_transcription_result) as mock_tx:
            result = runner.invoke(cli, ["transcribe", str(fake_wav), "--batch-size", "8"])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_tx.call_args
        assert kwargs.get("batch_size") == 8

    def test_language_forwarded(self, fake_wav, sample_transcription_result):
        runner = CliRunner()
        with patch("src.cli.transcribe", return_value=sample_transcription_result) as mock_tx:
            result = runner.invoke(cli, ["transcribe", str(fake_wav), "--language", "fr"])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_tx.call_args
        assert kwargs.get("language") == "fr"

    def test_diarization_file_forwarded(self, fake_wav, sample_result_with_speakers, tmp_path):
        diar_file = tmp_path / "diarization.json"
        diar_file.write_text("{}")  # content doesn't matter — transcribe is mocked
        runner = CliRunner()
        with patch("src.cli.transcribe", return_value=sample_result_with_speakers) as mock_tx:
            result = runner.invoke(cli, ["transcribe", str(fake_wav), "--diarization-file", str(diar_file)])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_tx.call_args
        assert kwargs.get("diarization_file") is not None

    def test_speaker_column_shown_when_diarization_applied(self, fake_wav, sample_result_with_speakers):
        runner = CliRunner()
        with patch("src.cli.transcribe", return_value=sample_result_with_speakers):
            result = runner.invoke(cli, ["transcribe", str(fake_wav)])

        assert result.exit_code == 0, result.output
        assert "Speaker" in result.output
        assert "SPEAKER_00" in result.output

    def test_invalid_compute_type_rejected(self, fake_wav):
        runner = CliRunner()
        result = runner.invoke(cli, ["transcribe", str(fake_wav), "--compute-type", "bfloat16"])
        assert result.exit_code != 0
