"""CLI tests using Click's CliRunner."""

from datetime import UTC

from click.testing import CliRunner

from src.cli import cli


class TestSeparateCommand:
    def test_success(self, fake_wav, tmp_output_dir, mock_audio_info, mocker):
        """CLI prints success output on successful separation."""
        from datetime import datetime

        from src.models.audio import SeparationResult

        mock_result = SeparationResult(
            input_file=fake_wav,
            input_info=mock_audio_info,
            vocals_path=tmp_output_dir / "htdemucs" / fake_wav.stem / "vocals.wav",
            no_vocals_path=tmp_output_dir / "htdemucs" / fake_wav.stem / "no_vocals.wav",
            output_dir=tmp_output_dir,
            model_name="htdemucs",
            device="cpu",
            processing_time_seconds=1.23,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

        mocker.patch("src.cli.separate", return_value=mock_result)

        runner = CliRunner()
        result = runner.invoke(cli, ["separate", str(fake_wav), "-o", str(tmp_output_dir), "-d", "cpu"])

        assert result.exit_code == 0
        assert "Separation Complete" in result.output

    def test_separation_error(self, fake_wav, tmp_output_dir, mocker):
        """CLI prints error panel and exits 1 on SeparationError."""
        from src.separator import SeparationError

        mocker.patch(
            "src.cli.separate",
            side_effect=SeparationError("Demucs failed", returncode=1, stderr="GPU error details"),
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["separate", str(fake_wav), "-o", str(tmp_output_dir)])

        assert result.exit_code == 1
        assert "Demucs failed" in result.output

    def test_file_not_found(self, tmp_path):
        """CLI errors on non-existent input file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["separate", str(tmp_path / "nonexistent.wav")])

        assert result.exit_code != 0

    def test_help(self):
        """CLI --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["separate", "--help"])

        assert result.exit_code == 0
        assert "OUTPUT" in result.output or "output" in result.output.lower()
