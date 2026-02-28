"""Unit tests for the Demucs separator module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.separator import (
    SeparationError,
    build_demucs_command,
    predict_output_paths,
    separate,
)


class TestBuildDemucsCommand:
    def test_basic_command(self, tmp_path: Path):
        cmd = build_demucs_command(
            input_file=tmp_path / "audio.wav",
            output_dir=tmp_path / "output",
            model="htdemucs",
            device="cuda",
        )
        assert cmd[0] == "demucs"
        assert "-n" in cmd
        assert "htdemucs" in cmd
        assert "--two-stems=vocals" in cmd
        assert "-o" in cmd
        assert str(tmp_path / "output") in cmd
        assert "-d" in cmd
        assert "cuda" in cmd
        assert str(tmp_path / "audio.wav") == cmd[-1]

    def test_cpu_device(self, tmp_path: Path):
        cmd = build_demucs_command(
            input_file=tmp_path / "audio.wav",
            output_dir=tmp_path / "output",
            device="cpu",
        )
        assert "cpu" in cmd

    def test_segment_option(self, tmp_path: Path):
        cmd = build_demucs_command(
            input_file=tmp_path / "audio.wav",
            output_dir=tmp_path / "output",
            segment=40,
        )
        segment_idx = cmd.index("--segment")
        assert cmd[segment_idx + 1] == "40"

    def test_no_segment_by_default(self, tmp_path: Path):
        cmd = build_demucs_command(
            input_file=tmp_path / "audio.wav",
            output_dir=tmp_path / "output",
        )
        assert "--segment" not in cmd


class TestPredictOutputPaths:
    def test_standard_paths(self, tmp_path: Path):
        vocals, no_vocals = predict_output_paths(
            input_file=Path("/audio/vinyl_side_a.wav"),
            output_dir=tmp_path / "output",
            model="htdemucs",
        )
        assert vocals == tmp_path / "output" / "htdemucs" / "vinyl_side_a" / "vocals.wav"
        assert no_vocals == tmp_path / "output" / "htdemucs" / "vinyl_side_a" / "no_vocals.wav"

    def test_different_model(self, tmp_path: Path):
        vocals, no_vocals = predict_output_paths(
            input_file=Path("/audio/track.wav"),
            output_dir=tmp_path / "output",
            model="htdemucs_ft",
        )
        assert vocals.parent.parent.name == "htdemucs_ft"


class TestSeparate:
    def test_success(self, fake_wav, tmp_output_dir, mock_audio_info, mocker):
        """Successful separation with mocked subprocess."""
        mocker.patch("src.separator.shutil.which", return_value="/usr/bin/demucs")
        mocker.patch("src.separator.probe_audio_file", return_value=mock_audio_info)

        # Create expected output files
        vocals_path, no_vocals_path = predict_output_paths(fake_wav, tmp_output_dir)
        vocals_path.parent.mkdir(parents=True)
        vocals_path.touch()
        no_vocals_path.touch()

        mock_run = mocker.patch(
            "src.separator.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="", stderr=""),
        )

        result = separate(input_file=fake_wav, output_dir=tmp_output_dir, device="cpu")

        assert result.vocals_path == vocals_path
        assert result.no_vocals_path == no_vocals_path
        assert result.device == "cpu"
        assert result.model_name == "htdemucs"
        assert result.processing_time_seconds >= 0
        mock_run.assert_called_once()

    def test_demucs_not_installed(self, fake_wav, tmp_output_dir, mocker):
        """Raises SeparationError when demucs is not on PATH."""
        mocker.patch("src.separator.shutil.which", return_value=None)

        with pytest.raises(SeparationError, match="not installed"):
            separate(input_file=fake_wav, output_dir=tmp_output_dir)

    def test_input_file_not_found(self, tmp_output_dir):
        """Raises FileNotFoundError for missing input."""
        with pytest.raises(FileNotFoundError):
            separate(input_file=Path("/nonexistent/audio.wav"), output_dir=tmp_output_dir)

    def test_demucs_failure(self, fake_wav, tmp_output_dir, mock_audio_info, mocker):
        """Raises SeparationError when demucs returns non-zero."""
        mocker.patch("src.separator.shutil.which", return_value="/usr/bin/demucs")
        mocker.patch("src.separator.probe_audio_file", return_value=mock_audio_info)
        mocker.patch(
            "src.separator.subprocess.run",
            return_value=MagicMock(returncode=1, stdout="", stderr="CUDA out of memory"),
        )

        with pytest.raises(SeparationError, match="return code 1") as exc_info:
            separate(input_file=fake_wav, output_dir=tmp_output_dir)
        assert exc_info.value.returncode == 1
        assert "CUDA out of memory" in exc_info.value.stderr

    def test_missing_output_files(self, fake_wav, tmp_output_dir, mock_audio_info, mocker):
        """Raises SeparationError when expected output files don't exist."""
        mocker.patch("src.separator.shutil.which", return_value="/usr/bin/demucs")
        mocker.patch("src.separator.probe_audio_file", return_value=mock_audio_info)
        mocker.patch(
            "src.separator.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="", stderr=""),
        )

        with pytest.raises(SeparationError, match="vocals output not found"):
            separate(input_file=fake_wav, output_dir=tmp_output_dir)
