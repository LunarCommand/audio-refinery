"""Pyannote speaker diarization — Python API wrapper.

Runs pyannote.audio's speaker-diarization-3.1 pipeline in-process.
GPU memory is released naturally when the process exits; each pipeline
step is meant to be run as its own CLI invocation.
"""

import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import huggingface_hub
from dotenv import load_dotenv

from src.models.audio import AudioFileInfo
from src.models.diarization import DiarizationResult, SpeakerSegment
from src.separator import probe_audio_file

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Compatibility patch: pyannote.audio 3.1.x calls hf_hub_download() with the
# deprecated `use_auth_token` kwarg that huggingface_hub 1.x removed entirely.
# We patch huggingface_hub.hf_hub_download HERE, at module level, so that when
# pyannote modules are imported inside load_pipeline() their local
# `from huggingface_hub import hf_hub_download` bindings pick up the wrapper.
# ---------------------------------------------------------------------------
_original_hf_hub_download = huggingface_hub.hf_hub_download


def _hf_hub_download_compat(*args, **kwargs):
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None and "token" not in kwargs:
        kwargs["token"] = use_auth_token
    return _original_hf_hub_download(*args, **kwargs)


huggingface_hub.hf_hub_download = _hf_hub_download_compat


class DiarizationError(Exception):
    """Raised when diarization fails."""

    def __init__(self, message: str):
        super().__init__(message)


def _resolve_hf_token(hf_token: str | None) -> str:
    """Return the HuggingFace token, loading from environment if not provided.

    Raises DiarizationError if no token is found.
    """
    if hf_token:
        return hf_token

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise DiarizationError(
            "HuggingFace token not found. Set HF_TOKEN in your environment or .env file.\n"
            "Setup steps:\n"
            "  1. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  2. Accept the model license at https://huggingface.co/pyannote/segmentation-3.0\n"
            "  3. Create a read-only token at https://huggingface.co/settings/tokens\n"
            "  4. Add HF_TOKEN=hf_your_token to your .env file"
        )
    return token


def load_pipeline(model: str, device: str, hf_token: str):
    """Load and return a pyannote Pipeline, moved to the specified device.

    Args:
        model: HuggingFace model ID, e.g. 'pyannote/speaker-diarization-3.1'.
        device: 'cuda' or 'cpu'.
        hf_token: HuggingFace access token.

    Returns:
        A loaded pyannote Pipeline ready for inference.

    Raises:
        DiarizationError: If the pipeline cannot be loaded.
    """
    try:
        import torch

        # Pass the token via environment variable rather than as a kwarg.
        # pyannote's from_pretrained() signature varies across versions, but
        # huggingface_hub always reads HF_TOKEN from the environment automatically.
        os.environ["HF_TOKEN"] = hf_token

        # Suppress noisy deprecation/reproducibility warnings from pyannote and
        # torchaudio internals that the user cannot act on.  The import itself
        # triggers the torchaudio backend warning at module-load time, so both
        # the import and the .to() call must be inside the catch_warnings block.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(model)
            pipeline.to(torch.device(device))

        return pipeline
    except Exception as exc:
        raise DiarizationError(f"Failed to load Pyannote pipeline '{model}': {exc}") from exc


def diarize(
    input_file: Path,
    device: str = DEFAULT_DEVICE,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    hf_token: str | None = None,
    model: str = DEFAULT_MODEL,
    _pipeline=None,
) -> DiarizationResult:
    """Run Pyannote speaker diarization on an audio file.

    Args:
        input_file: Path to the input audio file (typically the vocals.wav from separation).
        device: Compute device ('cuda' or 'cpu').
        min_speakers: Optional lower bound on speaker count.
        max_speakers: Optional upper bound on speaker count.
        hf_token: HuggingFace token. If None, reads from HF_TOKEN env var.
        model: Pyannote model ID.

    Returns:
        DiarizationResult with full provenance of the diarization run.

    Raises:
        FileNotFoundError: If input_file does not exist.
        DiarizationError: If no HF token is found, the pipeline fails to load,
            or diarization fails at runtime.
    """
    input_file = Path(input_file).resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    input_info: AudioFileInfo = probe_audio_file(input_file)
    if _pipeline is not None:
        pipeline = _pipeline
    else:
        token = _resolve_hf_token(hf_token)
        pipeline = load_pipeline(model, device, token)

    started_at = datetime.now(timezone.utc)
    t0 = time.monotonic()

    try:
        kwargs: dict = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="TensorFloat-32")
            annotation = pipeline(str(input_file), **kwargs)
    except Exception as exc:
        raise DiarizationError(f"Pyannote pipeline failed: {exc}") from exc

    processing_time = time.monotonic() - t0
    completed_at = datetime.now(timezone.utc)

    segments: list[SpeakerSegment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            SpeakerSegment(
                speaker_label=speaker,
                start_seconds=round(turn.start, 6),
                end_seconds=round(turn.end, 6),
            )
        )

    unique_speakers = len({seg.speaker_label for seg in segments})

    return DiarizationResult(
        input_file=input_file,
        input_info=input_info,
        segments=segments,
        num_speakers=unique_speakers,
        model_name=model,
        device=device,
        processing_time_seconds=round(processing_time, 2),
        started_at=started_at,
        completed_at=completed_at,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
