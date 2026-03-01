"""CLI tests for the `audio-refinery sentiment` subcommand."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.cli import cli
from src.models.sentiment import SegmentSentiment, SentimentResult, SentimentScore
from src.sentiment_analyzer import SentimentError


@pytest.fixture
def sample_sentiment_result(tmp_path: Path):
    now = datetime.now(UTC)
    tx_file = tmp_path / "transcription_001.json"
    tx_file.write_text("{}")  # content doesn't matter — analyze_sentiment is mocked
    return SentimentResult(
        transcription_file=tx_file,
        segments=[
            SegmentSentiment(
                start=0.0,
                end=2.0,
                text=" Hello world",
                speaker="SPEAKER_00",
                scores=[
                    SentimentScore(label="positive", score=0.8),
                    SentimentScore(label="neutral", score=0.15),
                    SentimentScore(label="negative", score=0.05),
                ],
                primary_sentiment="positive",
            ),
            SegmentSentiment(
                start=3.0,
                end=5.0,
                text=" Goodbye",
                scores=[
                    SentimentScore(label="neutral", score=0.6),
                    SentimentScore(label="positive", score=0.3),
                    SentimentScore(label="negative", score=0.1),
                ],
                primary_sentiment="neutral",
            ),
        ],
        device="cpu",
        processing_time_seconds=0.25,
        started_at=now,
        completed_at=now,
    )


class TestSentimentCommand:
    def test_happy_path(self, tmp_path: Path, sample_sentiment_result):
        tx_file = tmp_path / "transcription_001.json"
        tx_file.write_text("{}")
        runner = CliRunner()

        with (
            patch("src.cli.analyze_sentiment", return_value=sample_sentiment_result),
            patch("src.cli.merge_sentiment_into_transcription") as mock_merge,
        ):
            result = runner.invoke(cli, ["sentiment", str(tx_file)])

        assert result.exit_code == 0, result.output
        assert "Sentiment Analysis Complete" in result.output
        assert "Sentiment Distribution" in result.output
        assert "Segment Preview" in result.output
        mock_merge.assert_called_once_with(tx_file, sample_sentiment_result)

    def test_output_file_written(self, tmp_path: Path, sample_sentiment_result):
        tx_file = tmp_path / "transcription_001.json"
        tx_file.write_text("{}")
        out_file = tmp_path / "sentiment.json"
        runner = CliRunner()

        with (
            patch("src.cli.analyze_sentiment", return_value=sample_sentiment_result),
            patch("src.cli.merge_sentiment_into_transcription"),
        ):
            result = runner.invoke(cli, ["sentiment", str(tx_file), "-o", str(out_file)])

        assert result.exit_code == 0, result.output
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert len(data["segments"]) == 2
        assert data["device"] == "cpu"

    def test_no_output_file_prints_json_panel(self, tmp_path: Path, sample_sentiment_result):
        tx_file = tmp_path / "transcription_001.json"
        tx_file.write_text("{}")
        runner = CliRunner()

        with (
            patch("src.cli.analyze_sentiment", return_value=sample_sentiment_result),
            patch("src.cli.merge_sentiment_into_transcription"),
        ):
            result = runner.invoke(cli, ["sentiment", str(tx_file)])

        assert result.exit_code == 0, result.output
        assert "SentimentResult JSON" in result.output

    def test_sentiment_error_exits_nonzero(self, tmp_path: Path):
        tx_file = tmp_path / "transcription_001.json"
        tx_file.write_text("{}")
        runner = CliRunner()

        with patch("src.cli.analyze_sentiment", side_effect=SentimentError("no usable text")):
            result = runner.invoke(cli, ["sentiment", str(tx_file)])

        assert result.exit_code == 1
        assert "no usable text" in result.output

    def test_file_not_found_exits_nonzero(self, tmp_path: Path):
        missing = tmp_path / "no_such_file.json"
        runner = CliRunner()

        with patch("src.cli.analyze_sentiment", side_effect=FileNotFoundError("not found")):
            result = runner.invoke(cli, ["sentiment", str(missing)], catch_exceptions=False)

        # Click itself rejects missing files with exists=True
        assert result.exit_code != 0

    def test_model_option_forwarded(self, tmp_path: Path, sample_sentiment_result):
        tx_file = tmp_path / "transcription_001.json"
        tx_file.write_text("{}")
        runner = CliRunner()

        with (
            patch("src.cli.analyze_sentiment", return_value=sample_sentiment_result) as mock_analyze,
            patch("src.cli.merge_sentiment_into_transcription"),
        ):
            runner.invoke(cli, ["sentiment", str(tx_file), "--model", "my/custom-model"])

        _, kwargs = mock_analyze.call_args
        assert kwargs["model"] == "my/custom-model"

    def test_device_option_forwarded(self, tmp_path: Path, sample_sentiment_result):
        tx_file = tmp_path / "transcription_001.json"
        tx_file.write_text("{}")
        runner = CliRunner()

        with (
            patch("src.cli.analyze_sentiment", return_value=sample_sentiment_result) as mock_analyze,
            patch("src.cli.merge_sentiment_into_transcription"),
        ):
            runner.invoke(cli, ["sentiment", str(tx_file), "-d", "cpu"])

        _, kwargs = mock_analyze.call_args
        assert kwargs["device"] == "cpu"

    def test_merge_not_called_on_sentiment_error(self, tmp_path: Path):
        tx_file = tmp_path / "transcription_001.json"
        tx_file.write_text("{}")
        runner = CliRunner()

        with (
            patch("src.cli.analyze_sentiment", side_effect=SentimentError("boom")),
            patch("src.cli.merge_sentiment_into_transcription") as mock_merge,
        ):
            result = runner.invoke(cli, ["sentiment", str(tx_file)])

        assert result.exit_code == 1
        mock_merge.assert_not_called()

    def test_transcription_updated_in_place_message(self, tmp_path: Path, sample_sentiment_result):
        tx_file = tmp_path / "transcription_001.json"
        tx_file.write_text("{}")
        runner = CliRunner()

        with (
            patch("src.cli.analyze_sentiment", return_value=sample_sentiment_result),
            patch("src.cli.merge_sentiment_into_transcription"),
        ):
            result = runner.invoke(cli, ["sentiment", str(tx_file)])

        assert result.exit_code == 0, result.output
        assert "Transcription updated in place" in result.output
