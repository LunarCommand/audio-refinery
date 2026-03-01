"""WhisperX transcription — Python API wrapper.

Runs WhisperX (Whisper Large-v3 + Wav2Vec2 forced alignment) in-process.
GPU memory is released naturally when the process exits; each pipeline
step is meant to be run as its own CLI invocation.
"""

import contextlib
import logging
import os
import re
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

from src.models.audio import AudioFileInfo
from src.models.transcription import TranscriptionResult, TranscriptSegment, WordSegment
from src.separator import probe_audio_file

# Loggers that produce noisy INFO/WARNING output during model loading.
# These are version-mismatch and deprecation notices from third-party internals
# that the user cannot act on — silence them to keep the CLI output clean.
_NOISY_LOGGERS = [
    "whisperx",
    "whisperx.asr",
    "whisperx.vads",
    "whisperx.vads.pyannote",
    "pytorch_lightning",
    "pytorch_lightning.utilities",
    "pytorch_lightning.utilities.migration",
    "lightning",
    "lightning.pytorch",
    "pyannote.audio",
]


@contextlib.contextmanager
def _suppress_output():
    """Redirect stdout and stderr to /dev/null.

    Catches print() output from third-party libraries (pytorch_lightning checkpoint
    upgrade notices, pyannote version mismatch messages) that bypass Python logging.
    """
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        yield


DEFAULT_MODEL = "large-v3"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_BATCH_SIZE = 16
DEFAULT_LANGUAGE = "en"


class TranscriptionError(Exception):
    """Raised when transcription fails."""

    def __init__(self, message: str):
        super().__init__(message)


def _parse_whisperx_device(device: str) -> tuple[str, int]:
    """Split a PyTorch-style device string for ctranslate2's separate API.

    ctranslate2 (whisperx's backend) takes device and device_index as separate
    parameters and does not accept 'cuda:N' as a combined string.
    PyTorch-style load_align_model and align() still use the original string.

    Examples:
        'cuda:0' -> ('cuda', 0)
        'cuda:1' -> ('cuda', 1)
        'cuda'   -> ('cuda', 0)
        'cpu'    -> ('cpu', 0)
    """
    m = re.match(r"^cuda:(\d+)$", device)
    if m:
        return "cuda", int(m.group(1))
    return device, 0


def transcribe(
    input_file: Path,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    language: str = DEFAULT_LANGUAGE,
    diarization_file: Path | None = None,
    model: str = DEFAULT_MODEL,
    _whisperx_model=None,
) -> TranscriptionResult:
    """Run WhisperX transcription on an audio file.

    Args:
        input_file: Path to the input audio file (typically the vocals.wav from separation).
        device: Compute device ('cuda', 'cpu', or 'cuda:N').
        compute_type: CTranslate2 compute type ('float16', 'int8', or 'float32').
        batch_size: Batch size for transcription throughput.
        language: Language code (e.g. 'en') or 'auto' for detection.
        diarization_file: Optional path to a DiarizationResult JSON from step 2.
            When provided, speaker labels are merged into the transcript output.
        model: Whisper model size (default: 'large-v3').

    Returns:
        TranscriptionResult with full provenance of the transcription run.

    Raises:
        FileNotFoundError: If input_file does not exist.
        TranscriptionError: If whisperx is not installed, the diarization file is
            missing or invalid, or transcription fails at runtime.
    """
    input_file = Path(input_file).resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        import whisperx
    except ImportError as exc:
        raise TranscriptionError(
            "whisperx is not installed. Install it with:\n"
            "  uv pip install setuptools\n"
            "  uv pip install --no-deps --no-build-isolation "
            '"whisperx @ git+https://github.com/m-bain/whisperX.git@main"'
        ) from exc

    input_info: AudioFileInfo = probe_audio_file(input_file)

    # Load and validate diarization data if provided
    diarize_df = None
    if diarization_file is not None:
        diarization_file = Path(diarization_file).resolve()
        if not diarization_file.exists():
            raise TranscriptionError(f"Diarization file not found: {diarization_file}")
        try:
            import pandas as pd

            from src.models.diarization import DiarizationResult

            diarization_result = DiarizationResult.model_validate_json(diarization_file.read_text())
            diarize_df = pd.DataFrame(
                [
                    {"start": seg.start_seconds, "end": seg.end_seconds, "speaker": seg.speaker_label}
                    for seg in diarization_result.segments
                ]
            )
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Failed to load diarization file '{diarization_file}': {exc}") from exc

    # Pass None to whisperx for auto-detection; otherwise pass the language code directly.
    wx_language = None if language == "auto" else language

    started_at = datetime.now(UTC)
    t0 = time.monotonic()

    # Silence noisy version-mismatch and deprecation log output from third-party internals.
    for _logger_name in _NOISY_LOGGERS:
        logging.getLogger(_logger_name).setLevel(logging.ERROR)

    # ctranslate2 (whisperx backend) doesn't accept 'cuda:N' — split into device + index.
    # load_align_model and align use PyTorch directly and understand 'cuda:N' fine.
    ct2_device, ct2_device_index = _parse_whisperx_device(device)

    alignment_fallback = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            if _whisperx_model is None:
                with _suppress_output():
                    wx_model = whisperx.load_model(
                        model,
                        ct2_device,
                        device_index=ct2_device_index,
                        compute_type=compute_type,
                        language=wx_language,
                    )
            else:
                wx_model = _whisperx_model
            audio = whisperx.load_audio(str(input_file))
            raw_result = wx_model.transcribe(audio, batch_size=batch_size, language=wx_language)
        except Exception as exc:
            raise TranscriptionError(f"WhisperX transcription failed: {exc}") from exc

        detected_language = raw_result.get("language", language if language != "auto" else "en")

        # Forced alignment for word-level timestamps via Wav2Vec2
        try:
            align_model, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
            aligned = whisperx.align(
                raw_result["segments"], align_model, metadata, audio, device, return_char_alignments=False
            )
        except Exception:
            alignment_fallback = True
            aligned = raw_result

        # Speaker assignment via diarization merge
        if diarize_df is not None:
            try:
                aligned = whisperx.assign_word_speakers(diarize_df, aligned)
            except Exception as exc:
                raise TranscriptionError(f"Speaker assignment failed: {exc}") from exc

    processing_time = time.monotonic() - t0
    completed_at = datetime.now(UTC)

    segments = _build_segments(aligned.get("segments", []))

    return TranscriptionResult(
        input_file=input_file,
        input_info=input_info,
        language=detected_language,
        language_probability=raw_result.get("language_probability"),
        segments=segments,
        model_name=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        processing_time_seconds=round(processing_time, 2),
        started_at=started_at,
        completed_at=completed_at,
        diarization_applied=diarize_df is not None,
        diarization_file=diarization_file,
        alignment_fallback=alignment_fallback,
    )


def _build_segments(raw_segments: list[dict]) -> list[TranscriptSegment]:
    """Convert whisperx segment dicts to TranscriptSegment models."""
    segments = []
    for seg in raw_segments:
        words = []
        for w in seg.get("words", []):
            words.append(
                WordSegment(
                    word=w.get("word", ""),
                    start=w.get("start") or 0.0,
                    end=w.get("end") or 0.0,
                    score=w.get("score"),
                    speaker=w.get("speaker"),
                )
            )
        segments.append(
            TranscriptSegment(
                text=seg.get("text", ""),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                words=words,
                speaker=seg.get("speaker"),
            )
        )
    return segments
