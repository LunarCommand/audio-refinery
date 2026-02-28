"""CLI tests for the `audio-refinery diarize` subcommand."""

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.cli import cli
from src.diarizer import DiarizationError
from src.models.audio import AudioFileInfo
from src.models.diarization import DiarizationResult, SpeakerSegment


@pytest.fixture
def sample_diarization_result(fake_wav):
    now = datetime.now(timezone.utc)
    info = AudioFileInfo(
        path=fake_wav,
        sample_rate=44100,
        channels=1,
        duration_seconds=7.0,
        frames=308700,
        format_str="WAV",
        subtype="PCM_16",
    )
    return DiarizationResult(
        input_file=fake_wav,
        input_info=info,
        segments=[
            SpeakerSegment(speaker_label="SPEAKER_00", start_seconds=0.0, end_seconds=2.0),
            SpeakerSegment(speaker_label="SPEAKER_01", start_seconds=2.5, end_seconds=5.0),
            SpeakerSegment(speaker_label="SPEAKER_00", start_seconds=5.5, end_seconds=7.0),
        ],
        num_speakers=2,
        processing_time_seconds=1.23,
        started_at=now,
        completed_at=now,
    )


class TestDiarizeCommand:
    def test_happy_path(self, fake_wav, sample_diarization_result):
        runner = CliRunner()
        with patch("src.cli.diarize", return_value=sample_diarization_result):
            result = runner.invoke(cli, ["diarize", str(fake_wav)])

        assert result.exit_code == 0, result.output
        assert "Diarization Complete" in result.output
        assert "Speaker Breakdown" in result.output
        assert "SPEAKER_00" in result.output
        assert "SPEAKER_01" in result.output

    def test_diarization_error_exits_nonzero(self, fake_wav):
        runner = CliRunner()
        with patch("src.cli.diarize", side_effect=DiarizationError("HuggingFace token not found")):
            result = runner.invoke(cli, ["diarize", str(fake_wav)])

        assert result.exit_code == 1
        assert "HuggingFace token not found" in result.output

    def test_file_not_found_exits_nonzero(self, tmp_path):
        runner = CliRunner()
        missing = tmp_path / "no_such_file.wav"
        result = runner.invoke(cli, ["diarize", str(missing)])
        # Click checks file existence before calling the command
        assert result.exit_code != 0

    def test_output_file_written(self, fake_wav, sample_diarization_result, tmp_path):
        out_file = tmp_path / "result.json"
        runner = CliRunner()
        with patch("src.cli.diarize", return_value=sample_diarization_result):
            result = runner.invoke(cli, ["diarize", str(fake_wav), "-o", str(out_file)])

        assert result.exit_code == 0, result.output
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["num_speakers"] == 2
        assert len(data["segments"]) == 3

    def test_min_max_speakers_forwarded_to_diarize(self, fake_wav, sample_diarization_result):
        runner = CliRunner()
        with patch("src.cli.diarize", return_value=sample_diarization_result) as mock_diarize:
            result = runner.invoke(
                cli,
                ["diarize", str(fake_wav), "--min-speakers", "2", "--max-speakers", "4"],
            )

        assert result.exit_code == 0, result.output
        _, kwargs = mock_diarize.call_args
        assert kwargs.get("min_speakers") == 2
        assert kwargs.get("max_speakers") == 4

    def test_device_option_forwarded(self, fake_wav, sample_diarization_result):
        runner = CliRunner()
        with patch("src.cli.diarize", return_value=sample_diarization_result) as mock_diarize:
            result = runner.invoke(cli, ["diarize", str(fake_wav), "-d", "cpu"])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_diarize.call_args
        assert kwargs.get("device") == "cpu"

    def test_hf_token_option_forwarded(self, fake_wav, sample_diarization_result):
        runner = CliRunner()
        with patch("src.cli.diarize", return_value=sample_diarization_result) as mock_diarize:
            result = runner.invoke(cli, ["diarize", str(fake_wav), "--hf-token", "hf_test"])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_diarize.call_args
        assert kwargs.get("hf_token") == "hf_test"

    def test_no_output_file_prints_json_panel(self, fake_wav, sample_diarization_result):
        runner = CliRunner()
        with patch("src.cli.diarize", return_value=sample_diarization_result):
            result = runner.invoke(cli, ["diarize", str(fake_wav)])

        assert result.exit_code == 0, result.output
        assert "DiarizationResult JSON" in result.output
