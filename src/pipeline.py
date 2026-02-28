"""Batch audio processing pipeline: vocal separation → speaker diarization → transcription.

Processes all WAV files in a source directory through an interleaved per-file loop:
  each file is carried through all active stages (separation → diarization → transcription,
  and eventually emotion analysis and audio event detection) before moving to the next file.

Models are loaded once before the loop begins. Ghost-track stems (vocals.wav,
no_vocals.wav) are cleaned up from the RAM disk as soon as each file no longer needs them,
bounding scratch-disk usage to roughly one file's worth of data at any time.

Stage runner functions (run_separation_stage, run_diarization_stage,
run_transcription_stage) are kept as a lower-level API for testing and potential
single-stage CLI use-cases.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

try:
    import torch as _torch

    _has_torch = True
except ImportError:
    _has_torch = False

from src.diarizer import (
    DEFAULT_DEVICE as DIARIZER_DEFAULT_DEVICE,
)
from src.diarizer import (
    DEFAULT_MODEL as DIARIZER_DEFAULT_MODEL,
)
from src.diarizer import (
    DiarizationError,
    _resolve_hf_token,
    diarize,
    load_pipeline,
)
from src.sentiment_analyzer import (
    DEFAULT_MODEL as DEFAULT_SENTIMENT_MODEL,
)
from src.sentiment_analyzer import (
    SentimentError,
    analyze_sentiment,
    load_sentiment_pipeline,
    merge_sentiment_into_transcription,
)
from src.separator import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    SeparationError,
    separate,
)
from src.transcriber import (
    _NOISY_LOGGERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_LANGUAGE,
    TranscriptionError,
    _parse_whisperx_device,
    _suppress_output,
    transcribe,
)
from src.transcriber import (
    DEFAULT_MODEL as TRANSCRIBER_DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_DIR = Path("/audio/test/extracted")
DEFAULT_DIARIZATION_DIR = Path("/audio/test/diarization")
DEFAULT_TRANSCRIPTION_DIR = Path("/audio/test/transcription")
DEFAULT_SENTIMENT_DIR = Path("/audio/test/sentiment")


def _reset_vram(device: str) -> None:
    """Reset peak VRAM stats for the given device before a stage begins."""
    if _has_torch and device.startswith("cuda"):
        _torch.cuda.reset_peak_memory_stats(device)


def _read_vram(device: str) -> int | None:
    """Return peak VRAM allocated (bytes) since the last reset, or None on CPU."""
    if _has_torch and device.startswith("cuda"):
        return _torch.cuda.max_memory_allocated(device)
    return None


# Callback for individual stage runner functions: (content_id, index, total)
ProgressCallback = Callable[[str, int, int], None]

# Callback for the interleaved run_pipeline(): (content_id, stage_name, file_index, total_files)
PipelineProgressCallback = Callable[[str, str, int, int], None]


@dataclass
class FileOutcome:
    """Result for a single file within one pipeline stage."""

    content_id: str
    stage: str
    success: bool
    skipped: bool = False
    error: str | None = None
    processing_time_seconds: float = 0.0
    # Cost / usage metrics (populated for actively processed files; zero for skipped).
    input_file_bytes: int = 0
    output_file_bytes: int = 0
    audio_duration_seconds: float = 0.0
    rtf: float | None = None
    peak_vram_bytes: int | None = None
    word_count: int = 0
    segment_count: int = 0


@dataclass
class StageResult:
    """Aggregated outcomes for one pipeline stage."""

    outcomes: list[FileOutcome] = field(default_factory=list)

    @property
    def n_succeeded(self) -> int:
        return sum(1 for o in self.outcomes if o.success and not o.skipped)

    @property
    def n_skipped(self) -> int:
        return sum(1 for o in self.outcomes if o.skipped)

    @property
    def n_failed(self) -> int:
        return sum(1 for o in self.outcomes if not o.success)

    @property
    def succeeded_ids(self) -> set[str]:
        """Content IDs of all successful outcomes (processed + skipped)."""
        return {o.content_id for o in self.outcomes if o.success}

    @property
    def skipped_ids(self) -> list[str]:
        """Content IDs of outcomes that were skipped (output already existed)."""
        return [o.content_id for o in self.outcomes if o.skipped]

    @property
    def failed_outcomes(self) -> list[FileOutcome]:
        return [o for o in self.outcomes if not o.success]

    @property
    def total_input_bytes(self) -> int:
        return sum(o.input_file_bytes for o in self.outcomes if o.success and not o.skipped)

    @property
    def total_output_bytes(self) -> int:
        return sum(o.output_file_bytes for o in self.outcomes if o.success and not o.skipped)

    @property
    def avg_rtf(self) -> float | None:
        rtfs = [o.rtf for o in self.outcomes if o.rtf is not None and o.success and not o.skipped]
        return round(sum(rtfs) / len(rtfs), 3) if rtfs else None

    @property
    def peak_vram_bytes(self) -> int | None:
        vrams = [
            o.peak_vram_bytes for o in self.outcomes if o.peak_vram_bytes is not None and o.success and not o.skipped
        ]
        return max(vrams) if vrams else None

    @property
    def total_words(self) -> int:
        return sum(o.word_count for o in self.outcomes if o.success and not o.skipped)

    @property
    def total_segments(self) -> int:
        return sum(o.segment_count for o in self.outcomes if o.success and not o.skipped)


@dataclass
class PipelineResult:
    """Aggregated results for a complete pipeline run."""

    total_discovered: int
    separation: StageResult = field(default_factory=StageResult)
    diarization: StageResult = field(default_factory=StageResult)
    transcription: StageResult = field(default_factory=StageResult)
    sentiment: StageResult = field(default_factory=StageResult)
    total_processing_time_seconds: float = 0.0
    total_audio_duration_seconds: float = 0.0
    total_words: int = 0
    total_segments: int = 0


def discover_files(source_dir: Path) -> list[tuple[str, Path]]:
    """Discover WAV files in source_dir and extract content IDs.

    Expects filenames of the form ``audio_<content_id>.wav``.

    Returns:
        Sorted list of ``(content_id, wav_path)`` tuples.
    """
    files = []
    for wav_path in sorted(source_dir.glob("audio_*.wav")):
        stem = wav_path.stem
        if stem.startswith("audio_"):
            content_id = stem[len("audio_") :]
            files.append((content_id, wav_path))
    return files


def partition_ids(ids: list[str], n: int = 2) -> list[list[str]]:
    """Split content_ids into n interleaved partitions for multi-worker runs.

    Worker i gets positions i, i+n, i+2n, ... This distributes workload more evenly
    than a naive chunked split when file durations correlate with naming. With n=2,
    behavior is identical to the previous dual-worker implementation.

    Args:
        ids: Ordered list of content_id strings (typically from discover_files()).
        n: Number of partitions (workers). Defaults to 2.

    Returns:
        List of n lists of content_id strings. Partition i contains elements at
        positions i, i+n, i+2n, ... Partitions may differ in length by at most 1.
    """
    return [[ids[i] for i in range(start, len(ids), n)] for start in range(n)]


def _vocals_path(content_id: str, demucs_output_dir: Path, model: str = DEFAULT_MODEL) -> Path:
    """Predict where Demucs will write vocals.wav for a given content_id."""
    return demucs_output_dir / model / f"audio_{content_id}" / "vocals.wav"


def _no_vocals_path(content_id: str, demucs_output_dir: Path, model: str = DEFAULT_MODEL) -> Path:
    """Predict where Demucs will write no_vocals.wav for a given content_id."""
    return demucs_output_dir / model / f"audio_{content_id}" / "no_vocals.wav"


def _diarization_path(content_id: str, diarization_dir: Path) -> Path:
    return diarization_dir / f"diarization_{content_id}.json"


def _transcription_path(content_id: str, transcription_dir: Path) -> Path:
    return transcription_dir / f"transcription_{content_id}.json"


def _sentiment_path(content_id: str, sentiment_dir: Path) -> Path:
    return sentiment_dir / f"sentiment_{content_id}.json"


def _file_complete(path: Path) -> bool:
    """Return True if the path exists and is non-empty."""
    return path.exists() and path.stat().st_size > 0


def _cleanup_stem(path: Path) -> None:
    """Delete a stem file and its parent directory if it becomes empty."""
    if path.exists():
        path.unlink()
    try:
        path.parent.rmdir()
    except OSError:
        pass


def _load_whisperx_model(
    model: str, ct2_device: str, ct2_device_index: int, compute_type: str, wx_language: str | None
):
    """Load a WhisperX model with output suppression.

    Extracted as a named function so tests can patch it without importing whisperx.

    Raises:
        ImportError: If whisperx is not installed.
        Exception: If model loading fails for any other reason.
    """
    import whisperx

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _suppress_output():
            return whisperx.load_model(
                model,
                ct2_device,
                device_index=ct2_device_index,
                compute_type=compute_type,
                language=wx_language,
            )


# ---------------------------------------------------------------------------
# Stage runner functions — lower-level API, each processes all files for one
# stage. Used by tests and available for single-stage invocations.
# ---------------------------------------------------------------------------


def run_separation_stage(
    files: list[tuple[str, Path]],
    demucs_output_dir: Path,
    device: str = DEFAULT_DEVICE,
    segment: int | None = None,
    resume: bool = True,
    on_file: ProgressCallback | None = None,
) -> StageResult:
    """Run Demucs vocal separation on all files.

    Args:
        files: List of (content_id, wav_path) from discover_files().
        demucs_output_dir: Root directory for Demucs output stems.
        device: Compute device ('cuda', 'cuda:N', or 'cpu').
        segment: Optional segment size in seconds for VRAM optimization.
        resume: If True, skip files whose vocals.wav already exist and are non-empty.
        on_file: Called with (content_id, index, total) before processing each file.

    Returns:
        StageResult with a FileOutcome for every input file.
    """
    result = StageResult()
    total = len(files)

    for i, (content_id, wav_path) in enumerate(files):
        if on_file:
            on_file(content_id, i, total)

        vocals = _vocals_path(content_id, demucs_output_dir)
        if resume and _file_complete(vocals):
            result.outcomes.append(FileOutcome(content_id=content_id, stage="separate", success=True, skipped=True))
            continue

        try:
            sep = separate(input_file=wav_path, output_dir=demucs_output_dir, device=device, segment=segment)
            result.outcomes.append(
                FileOutcome(
                    content_id=content_id,
                    stage="separate",
                    success=True,
                    processing_time_seconds=sep.processing_time_seconds,
                )
            )
        except (SeparationError, FileNotFoundError, Exception) as exc:
            stderr_detail = (
                f" — stderr: {exc.stderr.strip()}" if isinstance(exc, SeparationError) and exc.stderr else ""
            )
            logger.warning("Separation failed for %s: %s%s", content_id, exc, stderr_detail)
            result.outcomes.append(FileOutcome(content_id=content_id, stage="separate", success=False, error=str(exc)))

    return result


def run_diarization_stage(
    files: list[tuple[str, Path]],
    separation: StageResult,
    demucs_output_dir: Path,
    diarization_dir: Path,
    device: str = DIARIZER_DEFAULT_DEVICE,
    hf_token: str | None = None,
    resume: bool = True,
    on_file: ProgressCallback | None = None,
) -> StageResult:
    """Run Pyannote speaker diarization on all successfully separated files.

    Loads the Pyannote model once and reuses it across all files for efficiency.

    Args:
        files: Full list of (content_id, wav_path) — stage filters by separation.succeeded_ids.
        separation: Results from run_separation_stage().
        demucs_output_dir: Root directory containing Demucs vocals stems.
        diarization_dir: Directory to write diarization_<content_id>.json files.
        device: Compute device.
        hf_token: HuggingFace token (overrides HF_TOKEN env var).
        resume: If True, skip files whose diarization JSON already exists and are non-empty.
        on_file: Called with (content_id, index, total) before processing each file.

    Returns:
        StageResult with a FileOutcome for every eligible file.
    """
    result = StageResult()
    eligible = [(cid, wav) for cid, wav in files if cid in separation.succeeded_ids]

    if not eligible:
        return result

    diarization_dir.mkdir(parents=True, exist_ok=True)

    # Avoid loading the model at all if every output is already present.
    if resume and all(_file_complete(_diarization_path(cid, diarization_dir)) for cid, _ in eligible):
        for content_id, _ in eligible:
            result.outcomes.append(FileOutcome(content_id=content_id, stage="diarize", success=True, skipped=True))
        return result

    # Load Pyannote model once for the whole stage.
    try:
        token = _resolve_hf_token(hf_token)
        pipeline = load_pipeline(DIARIZER_DEFAULT_MODEL, device, token)
    except DiarizationError as exc:
        for content_id, _ in eligible:
            result.outcomes.append(FileOutcome(content_id=content_id, stage="diarize", success=False, error=str(exc)))
        return result

    for i, (content_id, _) in enumerate(eligible):
        if on_file:
            on_file(content_id, i, len(eligible))

        diar_path = _diarization_path(content_id, diarization_dir)
        if resume and _file_complete(diar_path):
            result.outcomes.append(FileOutcome(content_id=content_id, stage="diarize", success=True, skipped=True))
            continue

        vocals = _vocals_path(content_id, demucs_output_dir)
        try:
            diar = diarize(input_file=vocals, device=device, hf_token=hf_token, _pipeline=pipeline)
            diar_path.write_text(diar.model_dump_json(indent=2))
            result.outcomes.append(
                FileOutcome(
                    content_id=content_id,
                    stage="diarize",
                    success=True,
                    processing_time_seconds=diar.processing_time_seconds,
                )
            )
        except (DiarizationError, FileNotFoundError, Exception) as exc:
            logger.warning("Diarization failed for %s: %s", content_id, exc)
            result.outcomes.append(FileOutcome(content_id=content_id, stage="diarize", success=False, error=str(exc)))

    return result


def run_transcription_stage(
    files: list[tuple[str, Path]],
    diarization: StageResult,
    demucs_output_dir: Path,
    diarization_dir: Path,
    transcription_dir: Path,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    language: str = DEFAULT_LANGUAGE,
    resume: bool = True,
    on_file: ProgressCallback | None = None,
) -> StageResult:
    """Run WhisperX transcription on all successfully diarized files.

    Loads the WhisperX model once and reuses it across all files for efficiency.

    Args:
        files: Full list of (content_id, wav_path) — stage filters by diarization.succeeded_ids.
        diarization: Results from run_diarization_stage().
        demucs_output_dir: Root directory containing Demucs vocals stems.
        diarization_dir: Directory containing diarization_<content_id>.json files.
        transcription_dir: Directory to write transcription_<content_id>.json files.
        device: Compute device.
        compute_type: CTranslate2 compute types ('float16', 'int8', 'float32').
        batch_size: WhisperX batch size.
        language: Language code (e.g. 'en') or 'auto'.
        resume: If True, skip files whose transcription JSON already exists and are non-empty.
        on_file: Called with (content_id, index, total) before processing each file.

    Returns:
        StageResult with a FileOutcome for every eligible file.
    """
    result = StageResult()
    eligible = [(cid, wav) for cid, wav in files if cid in diarization.succeeded_ids]

    if not eligible:
        return result

    transcription_dir.mkdir(parents=True, exist_ok=True)

    # Avoid loading the model at all if every output is already present.
    if resume and all(_file_complete(_transcription_path(cid, transcription_dir)) for cid, _ in eligible):
        for content_id, _ in eligible:
            result.outcomes.append(FileOutcome(content_id=content_id, stage="transcribe", success=True, skipped=True))
        return result

    wx_language = None if language == "auto" else language
    ct2_device, ct2_device_index = _parse_whisperx_device(device)

    for _logger_name in _NOISY_LOGGERS:
        logging.getLogger(_logger_name).setLevel(logging.ERROR)

    # Load WhisperX model once for the whole stage.
    try:
        wx_model = _load_whisperx_model(
            TRANSCRIBER_DEFAULT_MODEL, ct2_device, ct2_device_index, compute_type, wx_language
        )
    except ImportError as exc:
        for content_id, _ in eligible:
            result.outcomes.append(
                FileOutcome(
                    content_id=content_id,
                    stage="transcribe",
                    success=False,
                    error=f"whisperx not installed: {exc}",
                )
            )
        return result
    except Exception as exc:
        for content_id, _ in eligible:
            result.outcomes.append(
                FileOutcome(
                    content_id=content_id,
                    stage="transcribe",
                    success=False,
                    error=f"Failed to load WhisperX model: {exc}",
                )
            )
        return result

    for i, (content_id, _) in enumerate(eligible):
        if on_file:
            on_file(content_id, i, len(eligible))

        tx_path = _transcription_path(content_id, transcription_dir)
        if resume and _file_complete(tx_path):
            result.outcomes.append(FileOutcome(content_id=content_id, stage="transcribe", success=True, skipped=True))
            continue

        vocals = _vocals_path(content_id, demucs_output_dir)
        diar_path = _diarization_path(content_id, diarization_dir)
        diar_file = diar_path if _file_complete(diar_path) else None

        try:
            tx = transcribe(
                input_file=vocals,
                device=device,
                compute_type=compute_type,
                batch_size=batch_size,
                language=language,
                diarization_file=diar_file,
                _whisperx_model=wx_model,
            )
            tx_path.write_text(tx.model_dump_json(indent=2))
            result.outcomes.append(
                FileOutcome(
                    content_id=content_id,
                    stage="transcribe",
                    success=True,
                    processing_time_seconds=tx.processing_time_seconds,
                )
            )
        except (TranscriptionError, FileNotFoundError, Exception) as exc:
            logger.warning("Transcription failed for %s: %s", content_id, exc)
            result.outcomes.append(
                FileOutcome(content_id=content_id, stage="transcribe", success=False, error=str(exc))
            )

    return result


# ---------------------------------------------------------------------------
# Interleaved pipeline — the primary batch processing entry point.
# ---------------------------------------------------------------------------


def run_pipeline(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    demucs_output_dir: Path = DEFAULT_OUTPUT_DIR,
    diarization_dir: Path = DEFAULT_DIARIZATION_DIR,
    transcription_dir: Path = DEFAULT_TRANSCRIPTION_DIR,
    sentiment_dir: Path = DEFAULT_SENTIMENT_DIR,
    device: str = DEFAULT_DEVICE,
    segment: int | None = None,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    language: str = DEFAULT_LANGUAGE,
    hf_token: str | None = None,
    resume: bool = True,
    enable_sentiment: bool = False,
    enable_emotion: bool = False,
    enable_events: bool = False,
    keep_scratch: bool = False,
    on_progress: PipelineProgressCallback | None = None,
    manifest: list[str] | None = None,
    whisper_model: str = TRANSCRIBER_DEFAULT_MODEL,
) -> PipelineResult:
    """Run the full pipeline on all WAV files in source_dir, interleaved per file.

    Each file is carried through all active stages (separation → diarization →
    transcription, and in future steps: emotion analysis → audio event detection)
    before processing begins on the next file. All models are loaded once before
    the loop starts.

    Ghost-track stems are cleaned up from the scratch directory as soon as they are
    no longer needed (vocals.wav after transcription; no_vocals.wav immediately after
    separation unless --events is requested, in which case it is kept until the CLAP
    step — not yet implemented). This bounds RAM disk usage to roughly one file's
    worth of data at any time regardless of total file count.

    Args:
        source_dir: Directory containing audio_<content_id>.wav files.
        demucs_output_dir: Demucs output root directory (RAM disk strongly recommended).
        diarization_dir: Directory for diarization_<content_id>.json files.
        transcription_dir: Directory for transcription_<content_id>.json files.
        device: GPU device string ('cuda', 'cuda:N', or 'cpu').
        segment: Demucs segment size in seconds (VRAM optimisation).
        compute_type: WhisperX CTranslate2 compute type.
        batch_size: WhisperX segment batch size.
        language: Language code (e.g. 'en') or 'auto'.
        hf_token: HuggingFace token for Pyannote model download.
        resume: Skip files with existing outputs.
        enable_sentiment: Enable step 4 — Text Sentiment Analysis. Runs after
            transcription as a separate post-loop pass; no audio file required.
        enable_emotion: Enable step 5 — Speech Emotion Recognition (not yet implemented).
        enable_events: Enable step 6 — Audio Event Detection via CLAP (not yet implemented).
            When True, no_vocals.wav is retained on the scratch disk until step 6 runs.
        keep_scratch: Retain all Demucs stems on disk after the run (default: delete them).
        on_progress: Called with (content_id, stage_name, file_index, total_files) when
            each active stage begins for a file.
        manifest: If provided, only process files whose content_id is in this list.
            Files discovered on disk but not in the manifest are silently skipped.
            Used by ``pipeline-parallel`` to assign disjoint file sets to each worker.
        whisper_model: WhisperX model name (e.g. ``"large-v3"``, ``"distil-large-v3"``).
            Defaults to :data:`TRANSCRIBER_DEFAULT_MODEL`.

    Returns:
        PipelineResult with per-stage outcomes and total wall-clock runtime.
    """
    t0 = time.monotonic()

    files = discover_files(source_dir)
    if manifest is not None:
        manifest_set = set(manifest)
        files = [(cid, p) for cid, p in files if cid in manifest_set]
    result = PipelineResult(total_discovered=len(files))

    if not files:
        result.total_processing_time_seconds = 0.0
        return result

    diarization_dir.mkdir(parents=True, exist_ok=True)
    transcription_dir.mkdir(parents=True, exist_ok=True)

    total = len(files)

    # ── Pass 1: classify files and clean up stale scratch data ───────────
    # Files whose transcription output already exists are fully done — record
    # all three stages as skipped and clean up any ghost tracks they left behind
    # from a previous run (e.g. if the old architecture left stems on the RAM disk).
    pending: list[tuple[str, Path]] = []
    for content_id, wav_path in files:
        tx_path = _transcription_path(content_id, transcription_dir)
        if resume and _file_complete(tx_path):
            logger.info("Skipping %s — transcription already complete (all stages)", content_id)
            result.separation.outcomes.append(
                FileOutcome(content_id=content_id, stage="separate", success=True, skipped=True)
            )
            result.diarization.outcomes.append(
                FileOutcome(content_id=content_id, stage="diarize", success=True, skipped=True)
            )
            result.transcription.outcomes.append(
                FileOutcome(content_id=content_id, stage="transcribe", success=True, skipped=True)
            )
            if not keep_scratch:
                _cleanup_stem(_vocals_path(content_id, demucs_output_dir))
                if not enable_events:
                    _cleanup_stem(_no_vocals_path(content_id, demucs_output_dir))
        else:
            pending.append((content_id, wav_path))

    # ── Load sentiment model if needed ────────────────────────────────────
    # Text-only; independent of audio model availability. Loaded before the
    # early-return check so a sentiment-only run (all audio done, sentiment
    # pending) falls through to Pass 3 rather than returning early.
    sentiment_pipeline_obj = None
    sentiment_load_error: str | None = None
    if enable_sentiment:
        try:
            sentiment_pipeline_obj = load_sentiment_pipeline(DEFAULT_SENTIMENT_MODEL, "cpu")
        except SentimentError as exc:
            sentiment_load_error = str(exc)

    # Build a lookup so on_progress can report the file's position within the full
    # discovered list (not just within the pending subset).
    file_index_map = {cid: i for i, (cid, _) in enumerate(files)}

    if not pending:
        if not enable_sentiment:
            result.total_processing_time_seconds = round(time.monotonic() - t0, 2)
            return result
        # Sentiment only (all audio stages complete) — fall through to Pass 3.

    if pending:
        # ── Load audio models once for all pending files ───────────────────
        diar_pipeline_obj = None
        diar_load_error: str | None = None
        try:
            token = _resolve_hf_token(hf_token)
            diar_pipeline_obj = load_pipeline(DIARIZER_DEFAULT_MODEL, device, token)
        except DiarizationError as exc:
            diar_load_error = str(exc)

        wx_language = None if language == "auto" else language
        ct2_device, ct2_device_index = _parse_whisperx_device(device)
        for _logger_name in _NOISY_LOGGERS:
            logging.getLogger(_logger_name).setLevel(logging.ERROR)

        wx_model = None
        wx_load_error: str | None = None
        try:
            wx_model = _load_whisperx_model(whisper_model, ct2_device, ct2_device_index, compute_type, wx_language)
        except ImportError as exc:
            wx_load_error = f"whisperx not installed: {exc}"
        except Exception as exc:
            wx_load_error = f"Failed to load WhisperX model: {exc}"

        # Step 5 (SER) and step 6 (CLAP) model loads will be added here when implemented.
        # ser_model = _load_ser_model(device) if enable_emotion else None
        # clap_model = _load_clap_model(device) if enable_events else None

        # ── Pass 2: process each pending file through all active stages ────
        for content_id, wav_path in pending:
            file_idx = file_index_map[content_id]
            vocals = _vocals_path(content_id, demucs_output_dir)
            no_vocals = _no_vocals_path(content_id, demucs_output_dir)
            diar_path = _diarization_path(content_id, diarization_dir)
            tx_path = _transcription_path(content_id, transcription_dir)

            # ── Step 1: Vocal separation ──────────────────────────────────
            if resume and _file_complete(vocals):
                logger.info("Skipping separation for %s — vocals.wav already exists", content_id)
                result.separation.outcomes.append(
                    FileOutcome(content_id=content_id, stage="separate", success=True, skipped=True)
                )
            else:
                if on_progress:
                    on_progress(content_id, "separate", file_idx, total)
                try:
                    _reset_vram(device)
                    sep = separate(input_file=wav_path, output_dir=demucs_output_dir, device=device, segment=segment)
                    # Stat both stems before ghost-track cleanup so we capture their sizes.
                    _sep_in = wav_path.stat().st_size
                    _sep_out = (vocals.stat().st_size if vocals.exists() else 0) + (
                        no_vocals.stat().st_size if no_vocals.exists() else 0
                    )
                    _sep_dur = sep.input_info.duration_seconds
                    _sep_rtf = round(sep.processing_time_seconds / _sep_dur, 3) if _sep_dur > 0 else None
                    _sep_vram = _read_vram(device)
                    # no_vocals.wav is only needed by step 6 (CLAP). Delete it now unless
                    # events are enabled, to keep RAM disk usage as low as possible.
                    if not keep_scratch and not enable_events:
                        _cleanup_stem(no_vocals)
                    result.separation.outcomes.append(
                        FileOutcome(
                            content_id=content_id,
                            stage="separate",
                            success=True,
                            processing_time_seconds=sep.processing_time_seconds,
                            input_file_bytes=_sep_in,
                            output_file_bytes=_sep_out,
                            audio_duration_seconds=_sep_dur,
                            rtf=_sep_rtf,
                            peak_vram_bytes=_sep_vram,
                        )
                    )
                except (SeparationError, FileNotFoundError, Exception) as exc:
                    stderr_detail = (
                        f" — stderr: {exc.stderr.strip()}" if isinstance(exc, SeparationError) and exc.stderr else ""
                    )
                    logger.warning("Separation failed for %s: %s%s", content_id, exc, stderr_detail)
                    result.separation.outcomes.append(
                        FileOutcome(content_id=content_id, stage="separate", success=False, error=str(exc))
                    )
                    continue

            # ── Step 2: Speaker diarization ───────────────────────────────
            if resume and _file_complete(diar_path):
                logger.info("Skipping diarization for %s — diarization JSON already exists", content_id)
                result.diarization.outcomes.append(
                    FileOutcome(content_id=content_id, stage="diarize", success=True, skipped=True)
                )
            elif diar_load_error:
                result.diarization.outcomes.append(
                    FileOutcome(content_id=content_id, stage="diarize", success=False, error=diar_load_error)
                )
                continue
            else:
                if on_progress:
                    on_progress(content_id, "diarize", file_idx, total)
                try:
                    _reset_vram(device)
                    diar = diarize(input_file=vocals, device=device, hf_token=hf_token, _pipeline=diar_pipeline_obj)
                    diar_path.write_text(diar.model_dump_json(indent=2))
                    _diar_dur = diar.input_info.duration_seconds
                    result.diarization.outcomes.append(
                        FileOutcome(
                            content_id=content_id,
                            stage="diarize",
                            success=True,
                            processing_time_seconds=diar.processing_time_seconds,
                            input_file_bytes=vocals.stat().st_size if vocals.exists() else 0,
                            output_file_bytes=diar_path.stat().st_size,
                            audio_duration_seconds=_diar_dur,
                            rtf=round(diar.processing_time_seconds / _diar_dur, 3) if _diar_dur > 0 else None,
                            peak_vram_bytes=_read_vram(device),
                            segment_count=len(diar.segments),
                        )
                    )
                except (DiarizationError, FileNotFoundError, Exception) as exc:
                    logger.warning("Diarization failed for %s: %s", content_id, exc)
                    result.diarization.outcomes.append(
                        FileOutcome(content_id=content_id, stage="diarize", success=False, error=str(exc))
                    )
                    continue

            # ── Step 3: Transcription ─────────────────────────────────────
            if on_progress:
                on_progress(content_id, "transcribe", file_idx, total)

            if wx_load_error:
                result.transcription.outcomes.append(
                    FileOutcome(content_id=content_id, stage="transcribe", success=False, error=wx_load_error)
                )
                continue

            diar_file = diar_path if _file_complete(diar_path) else None
            try:
                _reset_vram(device)
                tx = transcribe(
                    input_file=vocals,
                    device=device,
                    compute_type=compute_type,
                    batch_size=batch_size,
                    language=language,
                    diarization_file=diar_file,
                    _whisperx_model=wx_model,
                )
                tx_path.write_text(tx.model_dump_json(indent=2))
                _tx_dur = tx.input_info.duration_seconds
                result.transcription.outcomes.append(
                    FileOutcome(
                        content_id=content_id,
                        stage="transcribe",
                        success=True,
                        processing_time_seconds=tx.processing_time_seconds,
                        input_file_bytes=vocals.stat().st_size if vocals.exists() else 0,
                        output_file_bytes=tx_path.stat().st_size,
                        audio_duration_seconds=_tx_dur,
                        rtf=round(tx.processing_time_seconds / _tx_dur, 3) if _tx_dur > 0 else None,
                        peak_vram_bytes=_read_vram(device),
                        word_count=sum(len(seg.words) for seg in tx.segments),
                        segment_count=len(tx.segments),
                    )
                )
            except (TranscriptionError, FileNotFoundError, Exception) as exc:
                logger.warning("Transcription failed for %s: %s", content_id, exc)
                result.transcription.outcomes.append(
                    FileOutcome(content_id=content_id, stage="transcribe", success=False, error=str(exc))
                )
                continue

            # ── Step 5: Speech Emotion Recognition (SER) — not yet implemented ─
            # Scaffold: model load goes in the audio model-loading block above.
            # if enable_emotion and ser_model is not None:
            #     if on_progress:
            #         on_progress(content_id, "emotion", file_idx, total)
            #     ... run SER on vocals, write emotion JSON ...

            # ── Ghost track cleanup: vocals.wav ───────────────────────────
            # vocals.wav consumed by steps 2, 3 (and step 5/SER when implemented).
            # Step 4 (sentiment) is text-only — cleanup position is unchanged.
            if not keep_scratch:
                _cleanup_stem(vocals)

            # ── Step 6: Audio Event Detection (CLAP) — not yet implemented ─
            # Scaffold: model load goes in the audio model-loading block above.
            # if enable_events and clap_model is not None:
            #     if on_progress:
            #         on_progress(content_id, "events", file_idx, total)
            #     ... run CLAP on no_vocals, write events JSON ...
            #     if not keep_scratch:
            #         _cleanup_stem(no_vocals)

    # ── Pass 3: Text Sentiment Analysis ───────────────────────────────────────
    # Text-only step — reads the transcription JSON, produces per-segment scores.
    # Runs after all audio stages so it covers both freshly-transcribed files
    # (processed in Pass 2) and previously-transcribed files (fast-tracked in
    # Pass 1). This means a first run with --sentiment on a previously-transcribed
    # base-dir produces sentiment for all files without re-running audio stages.
    if enable_sentiment:
        sentiment_dir.mkdir(parents=True, exist_ok=True)
        for content_id, _ in files:
            sentiment_path = _sentiment_path(content_id, sentiment_dir)
            tx_path = _transcription_path(content_id, transcription_dir)
            file_idx = file_index_map[content_id]

            if resume and _file_complete(sentiment_path):
                result.sentiment.outcomes.append(
                    FileOutcome(content_id=content_id, stage="sentiment", success=True, skipped=True)
                )
            elif content_id not in result.transcription.succeeded_ids:
                result.sentiment.outcomes.append(
                    FileOutcome(
                        content_id=content_id,
                        stage="sentiment",
                        success=False,
                        error="transcription stage failed",
                    )
                )
            elif sentiment_load_error:
                result.sentiment.outcomes.append(
                    FileOutcome(content_id=content_id, stage="sentiment", success=False, error=sentiment_load_error)
                )
            else:
                if on_progress:
                    on_progress(content_id, "sentiment", file_idx, total)
                try:
                    sent = analyze_sentiment(
                        transcription_file=tx_path,
                        device="cpu",
                        _sentiment_pipeline=sentiment_pipeline_obj,
                    )
                    sentiment_path.write_text(sent.model_dump_json(indent=2))
                    merge_sentiment_into_transcription(tx_path, sent)
                    result.sentiment.outcomes.append(
                        FileOutcome(
                            content_id=content_id,
                            stage="sentiment",
                            success=True,
                            processing_time_seconds=sent.processing_time_seconds,
                            input_file_bytes=tx_path.stat().st_size if tx_path.exists() else 0,
                            output_file_bytes=sentiment_path.stat().st_size,
                            segment_count=len(sent.segments),
                        )
                    )
                except (SentimentError, FileNotFoundError, Exception) as exc:
                    logger.warning("Sentiment analysis failed for %s: %s", content_id, exc)
                    result.sentiment.outcomes.append(
                        FileOutcome(content_id=content_id, stage="sentiment", success=False, error=str(exc))
                    )

    result.total_processing_time_seconds = round(time.monotonic() - t0, 2)
    # Aggregate run-level cost/usage metrics from transcription outcomes
    # (transcription has audio_duration, word_count, segment_count for all processed files).
    result.total_audio_duration_seconds = sum(
        o.audio_duration_seconds for o in result.transcription.outcomes if o.success and not o.skipped
    )
    result.total_words = result.transcription.total_words
    result.total_segments = result.transcription.total_segments
    # Signal completion so callers (e.g. pipeline-parallel's live display) can mark
    # this worker as done rather than stuck on the last file's final stage.
    if on_progress and total > 0:
        on_progress("", "done", total, total)
    return result
