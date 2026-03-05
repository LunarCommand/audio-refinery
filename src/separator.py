"""Demucs vocal separation — subprocess wrapper.

Runs Demucs htdemucs as a subprocess to ensure clean GPU memory release
between pipeline stages. All functions are pure/deterministic except
separate() which shells out to Demucs.
"""

import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import soundfile as sf

from src.models.audio import AudioFileInfo, SeparationResult

DEFAULT_OUTPUT_DIR = Path("/mnt/fast_scratch/demucs_output")
DEFAULT_MODEL = "htdemucs"
DEFAULT_DEVICE = "cuda"


class SeparationError(Exception):
    """Raised when the Demucs subprocess fails."""

    def __init__(self, message: str, returncode: int | None = None, stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


def probe_audio_file(path: Path) -> AudioFileInfo:
    """Read audio file metadata using soundfile."""
    info = sf.info(str(path))
    return AudioFileInfo(
        path=path,
        sample_rate=info.samplerate,
        channels=info.channels,
        duration_seconds=info.duration,
        frames=info.frames,
        format_str=info.format,
        subtype=info.subtype,
    )


def build_demucs_command(
    input_file: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    segment: int | None = None,
) -> list[str]:
    """Build the Demucs CLI command as a list of strings."""
    cmd = [
        "demucs",
        "-n",
        model,
        "--two-stems=vocals",
        "-o",
        str(output_dir),
        "-d",
        device,
    ]
    if segment is not None:
        cmd.extend(["--segment", str(segment)])
    cmd.append(str(input_file))
    return cmd


def predict_output_paths(
    input_file: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model: str = DEFAULT_MODEL,
) -> tuple[Path, Path]:
    """Predict where Demucs will write its output stems.

    Returns (vocals_path, no_vocals_path).
    Demucs writes to: output_dir / model / track_name / {vocals,no_vocals}.wav
    """
    track_name = input_file.stem
    stem_dir = output_dir / model / track_name
    return stem_dir / "vocals.wav", stem_dir / "no_vocals.wav"


def separate(
    input_file: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = DEFAULT_DEVICE,
    segment: int | None = None,
    model: str = DEFAULT_MODEL,
) -> SeparationResult:
    """Run Demucs vocal separation on an audio file.

    Args:
        input_file: Path to the input WAV file.
        output_dir: Directory for Demucs output.
        device: Compute device ('cuda' or 'cpu').
        segment: Optional segment size for VRAM optimization.
        model: Demucs model name.

    Returns:
        SeparationResult with full provenance of the run.

    Raises:
        SeparationError: If Demucs is not installed, fails, or output files are missing.
        FileNotFoundError: If the input file does not exist.
    """
    input_file = Path(input_file).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not shutil.which("demucs"):
        raise SeparationError("Demucs is not installed or not on PATH")

    input_info = probe_audio_file(input_file)
    cmd = build_demucs_command(input_file, output_dir, model, device, segment)
    vocals_path, no_vocals_path = predict_output_paths(input_file, output_dir, model)

    started_at = datetime.now(UTC)
    t0 = time.monotonic()

    result = subprocess.run(cmd, capture_output=True, text=True)

    processing_time = time.monotonic() - t0
    completed_at = datetime.now(UTC)

    if result.returncode != 0:
        raise SeparationError(
            f"Demucs failed with return code {result.returncode}",
            returncode=result.returncode,
            stderr=result.stderr,
        )

    if not vocals_path.exists():
        raise SeparationError(f"Expected vocals output not found: {vocals_path}")
    if not no_vocals_path.exists():
        raise SeparationError(f"Expected no_vocals output not found: {no_vocals_path}")

    return SeparationResult(
        input_file=input_file,
        input_info=input_info,
        vocals_path=vocals_path,
        no_vocals_path=no_vocals_path,
        output_dir=output_dir,
        model_name=model,
        device=device,
        processing_time_seconds=round(processing_time, 2),
        started_at=started_at,
        completed_at=completed_at,
        segment=segment,
    )
