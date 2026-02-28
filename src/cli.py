"""Audio Refinery CLI — Click command group with Rich output."""

import json
import os
import re
import subprocess
import sys
import threading
import time

# Force PCI bus order so CUDA device indices match nvidia-smi numbering.
# Must be set before any CUDA context is created.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.diarizer import (
    DEFAULT_DEVICE as DIARIZER_DEFAULT_DEVICE,
)
from src.diarizer import (
    DEFAULT_MODEL as DIARIZER_DEFAULT_MODEL,
)
from src.diarizer import (
    DiarizationError,
    diarize,
)
from src.gpu_utils import (
    detect_gpu_order,
    load_tflops_table,
    lookup_tflops,
    query_compute_processes,
    query_gpu_info,
    query_gpu_temperature,
)
from src.notifier import notify_pipeline_complete, notify_pipeline_parallel_complete, notify_thermal_shutdown
from src.pipeline import (
    FileOutcome as _FileOutcome,
)
from src.pipeline import (
    run_pipeline,
)
from src.sentiment_analyzer import (
    SentimentError,
    analyze_sentiment,
    merge_sentiment_into_transcription,
)
from src.separator import DEFAULT_DEVICE, DEFAULT_OUTPUT_DIR, SeparationError, separate
from src.transcriber import (
    DEFAULT_BATCH_SIZE as TRANSCRIBER_DEFAULT_BATCH_SIZE,
)
from src.transcriber import (
    DEFAULT_COMPUTE_TYPE as TRANSCRIBER_DEFAULT_COMPUTE_TYPE,
)
from src.transcriber import (
    DEFAULT_LANGUAGE as TRANSCRIBER_DEFAULT_LANGUAGE,
)
from src.transcriber import (
    TranscriptionError,
    transcribe,
)

console = Console()


def _validate_device(ctx, param, value: str) -> str:
    if value == "cpu" or value == "cuda" or re.match(r"^cuda:\d+$", value):
        return value
    raise click.BadParameter("must be 'cpu', 'cuda', or 'cuda:N' (e.g. 'cuda:0')")


def _warn_if_gpu_busy(devices: list[str]) -> None:
    """Check CUDA devices for active compute processes before a job starts.

    Prints a Rich-formatted warning listing each occupied device's processes
    and prompts the user to confirm or abort.  Silently skips CPU devices and
    continues without prompting when all devices are free or nvidia-smi is
    unavailable.
    """
    busy: list[tuple[str, list]] = []
    for device in devices:
        if device == "cpu":
            continue
        idx = int(device.split(":")[1]) if ":" in device else 0
        procs = query_compute_processes(idx)
        if procs:
            busy.append((device, procs))

    if not busy:
        return

    lines = []
    for device, procs in busy:
        total_mib = sum(p.used_memory_mib for p in procs)
        lines.append(f"[bold]{device}[/bold] — {len(procs)} process(es), ~{total_mib} MiB VRAM in use")
        for p in procs:
            lines.append(f"  [dim]PID {p.pid}:[/dim] {p.used_memory_mib} MiB")

    console.print(
        Panel(
            "\n".join(lines),
            title="[yellow bold]GPU(s) In Use[/yellow bold]",
            border_style="yellow",
        )
    )
    if not click.confirm("Proceed anyway?", default=False):
        console.print("[dim]Aborted.[/dim]")
        sys.exit(0)


def _fmt_temp(temp: int | None, limit: int) -> str:
    """Format a GPU temperature with colour coding relative to the safety limit."""
    if temp is None:
        return "[dim]—[/dim]"
    if temp >= limit:
        return f"[bold red]{temp}°C[/bold red]"
    if temp >= limit - 10:
        return f"[yellow]{temp}°C[/yellow]"
    return f"[green]{temp}°C[/green]"


def _run_temp_guard(device: str, limit: int, stop: threading.Event) -> None:
    """Daemon thread: polls GPU temperature every 5 s; calls os._exit(1) if the
    threshold is exceeded.  Set *stop* to terminate cleanly once the run finishes."""
    import os

    idx = int(device.split(":")[1]) if ":" in device else 0
    while not stop.wait(5.0):
        temp = query_gpu_temperature(idx)
        if temp is not None and temp >= limit:
            console.print(f"\n[bold red]Thermal shutdown: {device} at {temp}°C (limit {limit}°C). Exiting.[/bold red]")
            notify_thermal_shutdown(device, temp, limit)
            os._exit(1)


def _fmt_time(seconds: float) -> str:
    """Format seconds as 'Xm Ys' (>= 60s) or 'X.Xs' (< 60s). Returns '—' for zero."""
    if seconds == 0.0:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s"


@click.group()
def cli():
    """Audio Refinery — Audio processing pipeline."""


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
    help="Output directory for separated stems.",
)
@click.option(
    "-d",
    "--device",
    type=str,
    callback=_validate_device,
    default=DEFAULT_DEVICE,
    show_default=True,
    help="Compute device: 'cpu', 'cuda', or 'cuda:N' (e.g. 'cuda:0').",
)
@click.option(
    "--segment",
    type=int,
    default=None,
    help="Segment size in seconds for VRAM optimization.",
)
def separate_cmd(input_file: str, output_dir: str, device: str, segment: int | None):
    """Run Demucs vocal separation on an audio file."""
    from pathlib import Path

    input_path = Path(input_file)
    output_path = Path(output_dir)

    console.print(
        Panel(
            f"[bold]Input:[/bold]  {input_path}\n"
            f"[bold]Output:[/bold] {output_path}\n"
            f"[bold]Device:[/bold] {device}" + (f"\n[bold]Segment:[/bold] {segment}s" if segment else ""),
            title="[blue bold]Audio Refinery — Demucs Vocal Separation[/blue bold]",
            border_style="blue",
        )
    )

    _warn_if_gpu_busy([device])

    with console.status("[bold green]Running Demucs separation...", spinner="dots"):
        try:
            result = separate(
                input_file=input_path,
                output_dir=output_path,
                device=device,
                segment=segment,
            )
        except (SeparationError, FileNotFoundError) as e:
            console.print(Panel(f"[bold]{e}[/bold]", title="[red bold]Error[/red bold]", border_style="red"))
            if isinstance(e, SeparationError) and e.stderr:
                console.print(Panel(e.stderr.strip(), title="[red]stderr[/red]", border_style="red dim"))
            sys.exit(1)

    table = Table(title="Separation Complete", border_style="green")
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("Input", str(result.input_file))
    table.add_row("Duration", f"{result.input_info.duration_seconds:.1f}s")
    table.add_row("Sample Rate", f"{result.input_info.sample_rate} Hz")
    table.add_row("Channels", str(result.input_info.channels))
    table.add_row("Vocals", str(result.vocals_path))
    table.add_row("No Vocals", str(result.no_vocals_path))
    table.add_row("Processing Time", f"{result.processing_time_seconds:.2f}s")
    table.add_row("Device", result.device)
    table.add_row("Model", result.model_name)
    console.print(table)

    console.print(
        Panel(
            result.model_dump_json(indent=2),
            title="[dim]SeparationResult JSON[/dim]",
            border_style="dim",
        )
    )


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option(
    "-d",
    "--device",
    type=str,
    callback=_validate_device,
    default=DIARIZER_DEFAULT_DEVICE,
    show_default=True,
    help="Compute device: 'cpu', 'cuda', or 'cuda:N' (e.g. 'cuda:0').",
)
@click.option(
    "--min-speakers",
    type=int,
    default=None,
    help="Minimum number of speakers (optional hint for Pyannote).",
)
@click.option(
    "--max-speakers",
    type=int,
    default=None,
    help="Maximum number of speakers (optional hint for Pyannote).",
)
@click.option(
    "--hf-token",
    default=None,
    help="HuggingFace token (overrides HF_TOKEN env var).",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(dir_okay=False, resolve_path=True),
    default=None,
    help="Write DiarizationResult JSON to this file.",
)
def diarize_cmd(
    input_file: str,
    device: str,
    min_speakers: int | None,
    max_speakers: int | None,
    hf_token: str | None,
    output_file: str | None,
):
    """Run Pyannote speaker diarization on an audio file."""
    from pathlib import Path

    input_path = Path(input_file)

    speaker_hints = ""
    if min_speakers is not None:
        speaker_hints += f"\n[bold]Min speakers:[/bold] {min_speakers}"
    if max_speakers is not None:
        speaker_hints += f"\n[bold]Max speakers:[/bold] {max_speakers}"

    console.print(
        Panel(
            f"[bold]Input:[/bold]  {input_path}\n"
            f"[bold]Device:[/bold] {device}"
            f"[bold]Model:[/bold]  {DIARIZER_DEFAULT_MODEL}" + speaker_hints,
            title="[blue bold]Audio Refinery — Pyannote Speaker Diarization[/blue bold]",
            border_style="blue",
        )
    )

    _warn_if_gpu_busy([device])

    with console.status("[bold green]Running Pyannote diarization...", spinner="dots"):
        try:
            result = diarize(
                input_file=input_path,
                device=device,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                hf_token=hf_token,
            )
        except (DiarizationError, FileNotFoundError) as e:
            console.print(Panel(f"[bold]{e}[/bold]", title="[red bold]Error[/red bold]", border_style="red"))
            sys.exit(1)

    # Speaker summary table
    summary = Table(title="Diarization Complete", border_style="green")
    summary.add_column("Property", style="bold")
    summary.add_column("Value")
    summary.add_row("Input", str(result.input_file))
    summary.add_row("Duration", f"{result.input_info.duration_seconds:.1f}s")
    summary.add_row("Speakers detected", str(result.num_speakers))
    summary.add_row("Total segments", str(len(result.segments)))
    total_speech = sum(s.duration_seconds for s in result.segments)
    summary.add_row("Total speech", f"{total_speech:.1f}s")
    summary.add_row("Processing time", f"{result.processing_time_seconds:.2f}s")
    summary.add_row("Device", result.device)
    console.print(summary)

    # Per-speaker breakdown
    from collections import defaultdict

    speaker_durations: dict[str, float] = defaultdict(float)
    speaker_counts: dict[str, int] = defaultdict(int)
    for seg in result.segments:
        speaker_durations[seg.speaker_label] += seg.duration_seconds
        speaker_counts[seg.speaker_label] += 1

    breakdown = Table(title="Speaker Breakdown", border_style="green")
    breakdown.add_column("Speaker", style="bold")
    breakdown.add_column("Segments", justify="right")
    breakdown.add_column("Total Duration", justify="right")
    for label in sorted(speaker_durations):
        breakdown.add_row(label, str(speaker_counts[label]), f"{speaker_durations[label]:.2f}s")
    console.print(breakdown)

    json_str = result.model_dump_json(indent=2)
    if output_file:
        Path(output_file).write_text(json_str)
        console.print(f"[green]DiarizationResult written to:[/green] {output_file}")
    else:
        console.print(
            Panel(
                json_str,
                title="[dim]DiarizationResult JSON[/dim]",
                border_style="dim",
            )
        )


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option(
    "-d",
    "--device",
    type=str,
    callback=_validate_device,
    default=DIARIZER_DEFAULT_DEVICE,
    show_default=True,
    help="Compute device: 'cpu', 'cuda', or 'cuda:N' (e.g. 'cuda:0').",
)
@click.option(
    "--compute-type",
    type=click.Choice(["float16", "int8_float16", "int8", "float32"]),
    default=TRANSCRIBER_DEFAULT_COMPUTE_TYPE,
    show_default=True,
    help="CTranslate2 compute type.",
)
@click.option(
    "--batch-size",
    type=int,
    default=TRANSCRIBER_DEFAULT_BATCH_SIZE,
    show_default=True,
    help="Batch size for transcription.",
)
@click.option(
    "--language",
    default=TRANSCRIBER_DEFAULT_LANGUAGE,
    show_default=True,
    help="Language code (e.g. 'en', 'fr') or 'auto' for detection.",
)
@click.option(
    "--diarization-file",
    type=click.Path(dir_okay=False, resolve_path=True),
    default=None,
    help="DiarizationResult JSON from step 2 for speaker assignment.",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(dir_okay=False, resolve_path=True),
    default=None,
    help="Write TranscriptionResult JSON to this file.",
)
def transcribe_cmd(
    input_file: str,
    device: str,
    compute_type: str,
    batch_size: int,
    language: str,
    diarization_file: str | None,
    output_file: str | None,
):
    """Run WhisperX transcription on an audio file."""
    from pathlib import Path

    input_path = Path(input_file)
    diarization_path = Path(diarization_file) if diarization_file else None

    info_lines = (
        f"[bold]Input:[/bold]        {input_path}\n"
        f"[bold]Device:[/bold]       {device}\n"
        f"[bold]Compute type:[/bold] {compute_type}\n"
        f"[bold]Batch size:[/bold]   {batch_size}\n"
        f"[bold]Language:[/bold]     {language}"
    )
    if diarization_path:
        info_lines += f"\n[bold]Diarization:[/bold]  {diarization_path}"

    console.print(
        Panel(
            info_lines,
            title="[blue bold]Audio Refinery — WhisperX Transcription[/blue bold]",
            border_style="blue",
        )
    )

    _warn_if_gpu_busy([device])

    with console.status("[bold green]Running WhisperX transcription...", spinner="dots"):
        try:
            result = transcribe(
                input_file=input_path,
                device=device,
                compute_type=compute_type,
                batch_size=batch_size,
                language=language,
                diarization_file=diarization_path,
            )
        except (TranscriptionError, FileNotFoundError) as e:
            console.print(Panel(f"[bold]{e}[/bold]", title="[red bold]Error[/red bold]", border_style="red"))
            sys.exit(1)

    if result.alignment_fallback:
        console.print(
            Panel(
                "Wav2Vec2 alignment model unavailable for this language.\n"
                "Word-level timestamps are approximate (Whisper raw output).\n"
                "Install nltk to enable alignment: [bold]uv pip install nltk[/bold]",
                title="[yellow bold]Alignment Warning[/yellow bold]",
                border_style="yellow",
            )
        )

    # Summary table
    summary = Table(title="Transcription Complete", border_style="green")
    summary.add_column("Property", style="bold")
    summary.add_column("Value")
    summary.add_row("Input", str(result.input_file))
    summary.add_row("Duration", f"{result.input_info.duration_seconds:.1f}s")
    summary.add_row("Language", result.language)
    summary.add_row("Segments", str(len(result.segments)))
    total_words = sum(len(s.words) for s in result.segments)
    summary.add_row("Words", str(total_words))
    summary.add_row("Diarization applied", "yes" if result.diarization_applied else "no")
    summary.add_row("Processing time", f"{result.processing_time_seconds:.2f}s")
    summary.add_row("Device", result.device)
    console.print(summary)

    # Transcript preview
    if result.segments:
        preview = Table(title="Transcript Preview (first 10 segments)", border_style="green")
        preview.add_column("Start", justify="right")
        preview.add_column("End", justify="right")
        if result.diarization_applied:
            preview.add_column("Speaker")
        preview.add_column("Text")
        for seg in result.segments[:10]:
            text_preview = seg.text[:60] + ("..." if len(seg.text) > 60 else "")
            if result.diarization_applied:
                preview.add_row(f"{seg.start:.2f}s", f"{seg.end:.2f}s", seg.speaker or "—", text_preview)
            else:
                preview.add_row(f"{seg.start:.2f}s", f"{seg.end:.2f}s", text_preview)
        console.print(preview)

    json_str = result.model_dump_json(indent=2)
    if output_file:
        Path(output_file).write_text(json_str)
        console.print(f"[green]TranscriptionResult written to:[/green] {output_file}")
    else:
        console.print(
            Panel(
                json_str,
                title="[dim]TranscriptionResult JSON[/dim]",
                border_style="dim",
            )
        )


@cli.command("sentiment")
@click.argument("transcription_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option(
    "--model",
    default="cardiffnlp/twitter-roberta-base-sentiment-latest",
    show_default=True,
    help="HuggingFace text-classification model name.",
)
@click.option(
    "-d",
    "--device",
    type=str,
    callback=_validate_device,
    default="cpu",
    show_default=True,
    help="Compute device: 'cpu', 'cuda', or 'cuda:N'.",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(dir_okay=False, resolve_path=True),
    default=None,
    help="Write SentimentResult JSON to this file.",
)
def sentiment_cmd(transcription_file: str, model: str, device: str, output_file: str | None):
    """Run text sentiment analysis on each segment in a transcription.

    TRANSCRIPTION_FILE is the TranscriptionResult JSON produced by
    'audio-refinery transcribe'. No audio file is required.

    Results are written to --output-file (or printed if omitted). The
    TRANSCRIPTION_FILE is also updated in place with sentiment fields merged
    into each segment, giving a single enriched output for downstream use.
    """
    from pathlib import Path

    tx_path = Path(transcription_file)

    console.print(
        Panel(
            f"[bold]Transcription:[/bold] {tx_path}\n"
            f"[bold]Model:[/bold]         {model}\n"
            f"[bold]Device:[/bold]        {device}",
            title="[blue bold]Audio Refinery — Text Sentiment Analysis[/blue bold]",
            border_style="blue",
        )
    )

    with console.status("[bold green]Running sentiment analysis...", spinner="dots"):
        try:
            result = analyze_sentiment(transcription_file=tx_path, device=device, model=model)
        except (SentimentError, FileNotFoundError) as e:
            console.print(Panel(f"[bold]{e}[/bold]", title="[red bold]Error[/red bold]", border_style="red"))
            sys.exit(1)

    # Summary table
    n_analyzed = len(result.segments)
    summary = Table(title="Sentiment Analysis Complete", border_style="green")
    summary.add_column("Property", style="bold")
    summary.add_column("Value")
    summary.add_row("Segments analyzed", str(n_analyzed))
    summary.add_row("Processing time", f"{result.processing_time_seconds:.3f}s")
    summary.add_row("Model", result.model_name)
    summary.add_row("Device", result.device)
    console.print(summary)

    # Sentiment distribution
    from collections import Counter, defaultdict

    label_counts: Counter = Counter(seg.primary_sentiment for seg in result.segments)
    label_totals: dict = defaultdict(float)
    label_n: Counter = Counter()
    for seg in result.segments:
        for sc in seg.scores:
            label_totals[sc.label] += sc.score
            label_n[sc.label] += 1

    dist = Table(title="Sentiment Distribution", border_style="green")
    dist.add_column("Label", style="bold")
    dist.add_column("Count", justify="right")
    dist.add_column("Mean Score", justify="right")
    for label in sorted(label_counts, key=lambda x: label_counts[x], reverse=True):
        mean_score = label_totals[label] / label_n[label] if label_n[label] else 0.0
        dist.add_row(label, str(label_counts[label]), f"{mean_score:.3f}")
    console.print(dist)

    # Segment preview
    if result.segments:
        preview = Table(title="Segment Preview (first 10)", border_style="green")
        preview.add_column("Start", justify="right")
        preview.add_column("End", justify="right")
        preview.add_column("Speaker")
        preview.add_column("Text")
        preview.add_column("Sentiment")
        preview.add_column("Confidence", justify="right")
        for seg in result.segments[:10]:
            text_preview = (seg.text or "")[:60] + ("..." if seg.text and len(seg.text) > 60 else "")
            top = seg.scores[0]
            preview.add_row(
                f"{seg.start:.2f}s",
                f"{seg.end:.2f}s",
                seg.speaker or "—",
                text_preview,
                seg.primary_sentiment,
                f"{top.score:.3f}",
            )
        console.print(preview)

    # Merge sentiment into the source transcription file
    try:
        merge_sentiment_into_transcription(tx_path, result)
        console.print(f"[green]Transcription updated in place:[/green] {tx_path}")
    except Exception as e:
        console.print(Panel(f"[bold]Merge failed: {e}[/bold]", title="[red bold]Error[/red bold]", border_style="red"))
        sys.exit(1)

    json_str = result.model_dump_json(indent=2)
    if output_file:
        from pathlib import Path as _Path

        _Path(output_file).write_text(json_str)
        console.print(f"[green]SentimentResult written to:[/green] {output_file}")
    else:
        console.print(
            Panel(
                json_str,
                title="[dim]SentimentResult JSON[/dim]",
                border_style="dim",
            )
        )


@cli.command()
@click.option(
    "--base-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    required=True,
    help=(
        "Base directory for the pipeline run. "
        "Source WAV files must exist in <base>/extracted. "
        "Output subdirectories (diarization/, transcription/, summary/) "
        "are created automatically."
    ),
)
@click.option(
    "-d",
    "--device",
    type=str,
    callback=_validate_device,
    default=DEFAULT_DEVICE,
    show_default=True,
    help="Compute device: 'cpu', 'cuda', or 'cuda:N' (e.g. 'cuda:0').",
)
@click.option(
    "--segment",
    type=int,
    default=None,
    help="Demucs segment size in seconds for VRAM optimisation.",
)
@click.option(
    "--compute-type",
    type=click.Choice(["float16", "int8_float16", "int8", "float32"]),
    default=TRANSCRIBER_DEFAULT_COMPUTE_TYPE,
    show_default=True,
    help="WhisperX CTranslate2 compute type.",
)
@click.option(
    "--batch-size",
    type=int,
    default=TRANSCRIBER_DEFAULT_BATCH_SIZE,
    show_default=True,
    help="WhisperX batch size.",
)
@click.option(
    "--language",
    default=TRANSCRIBER_DEFAULT_LANGUAGE,
    show_default=True,
    help="Language code (e.g. 'en') or 'auto' for detection.",
)
@click.option(
    "--hf-token",
    default=None,
    help="HuggingFace token for Pyannote (overrides HF_TOKEN env var).",
)
@click.option(
    "--no-resume",
    is_flag=True,
    default=False,
    help="Re-process all files, ignoring existing outputs.",
)
@click.option(
    "--keep-scratch",
    is_flag=True,
    default=False,
    help="Keep Demucs stems on the scratch disk after the run (default: delete per file).",
)
@click.option(
    "--whisper-model",
    default="large-v3",
    show_default=True,
    help=(
        "WhisperX model name. Any valid HuggingFace model name is accepted. "
        "Well-tested variants: large-v3 (default, highest accuracy), "
        "distil-large-v3 (~2× faster, minimal accuracy delta), "
        "medium.en (~3.2× faster, English-only), medium (~3× faster, multilingual)."
    ),
)
@click.option(
    "--temp-limit",
    type=int,
    default=80,
    show_default=True,
    help=(
        "GPU temperature limit in °C. A background thread monitors the device every 5 s "
        "and shuts down the pipeline if this threshold is exceeded. Set to 0 to disable."
    ),
)
@click.option(
    "--demucs-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    default=None,
    hidden=True,
    help=(
        "Override the Demucs scratch directory. When set, the RAM disk check and "
        "interactive prompt are skipped entirely. Used internally by pipeline-parallel."
    ),
)
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    hidden=True,
    help=(
        "Path to a plaintext file of content_ids to process (one per line). "
        "Files not listed are skipped. Used internally by pipeline-parallel."
    ),
)
@click.option(
    "--summary-file",
    type=click.Path(dir_okay=False, resolve_path=True),
    default=None,
    hidden=True,
    help=(
        "Write the pipeline summary JSON to this path instead of "
        "<base>/summary/pipeline_summary.json. Used internally by pipeline-parallel."
    ),
)
@click.option(
    "--progress-file",
    type=click.Path(dir_okay=False, resolve_path=True),
    default=None,
    hidden=True,
    help=(
        "Write per-file JSON progress updates to this path after each stage completion. "
        "Used internally by pipeline-parallel to drive its live status display."
    ),
)
@click.option(
    "--sentiment",
    is_flag=True,
    default=False,
    help="Enable step 4: Text Sentiment Analysis. Runs on the transcript JSON; no audio required.",
)
@click.option(
    "--emotion",
    is_flag=True,
    default=False,
    help="Enable step 5: Speech Emotion Recognition (SER). [not yet implemented]",
)
@click.option(
    "--events",
    is_flag=True,
    default=False,
    help=(
        "Enable step 6: Audio Event Detection via CLAP. [not yet implemented] "
        "When set, no_vocals.wav is retained on the scratch disk for future processing."
    ),
)
def pipeline(
    base_dir: str,
    device: str,
    segment: int | None,
    compute_type: str,
    batch_size: int,
    language: str,
    hf_token: str | None,
    no_resume: bool,
    keep_scratch: bool,
    whisper_model: str,
    temp_limit: int,
    demucs_dir: str | None,
    manifest: str | None,
    summary_file: str | None,
    progress_file: str | None,
    sentiment: bool,
    emotion: bool,
    events: bool,
):
    """Run the full audio processing pipeline on all WAV files in <base>/extracted.

    Each file is carried through all active stages (separation → diarization →
    transcription) before moving to the next file. All models are loaded once at
    startup. Ghost-track stems are cleaned up from the RAM disk as soon as they
    are no longer needed, keeping scratch-disk usage bounded to roughly one file
    at a time regardless of how many files are in the run.

    Steps 5 (--emotion, Speech Emotion Recognition) and 6 (--events, Audio Event
    Detection via CLAP) are scaffolded but not yet implemented.

    Directory layout derived from --base-dir:
      <base>/extracted/     — input WAV files (must exist)
      <base>/diarization/   — diarization JSON files (created if absent)
      <base>/transcription/ — transcription JSON output (created if absent)
      <base>/sentiment/     — sentiment JSON files (created if --sentiment is set)
      <base>/summary/       — pipeline run summary (created if absent)

    Demucs scratch (priority order):
      /mnt/fast_scratch/demucs  — RAM disk (used automatically if mounted)
      <base>/demucs             — disk fallback (requires confirmation)
    """
    from pathlib import Path

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from src.pipeline import discover_files

    base_path = Path(base_dir)
    source_path = base_path / "extracted"
    diar_path = base_path / "diarization"
    tx_path = base_path / "transcription"
    sentiment_dir = base_path / "sentiment"
    summary_dir = base_path / "summary"
    resume = not no_resume

    if not source_path.exists():
        console.print(
            Panel(
                f"[bold]Source directory not found:[/bold] {source_path}\n"
                "Create the directory and place audio_<content_id>.wav files inside it.",
                title="[red bold]Error[/red bold]",
                border_style="red",
            )
        )
        sys.exit(1)

    # GPU availability check — skipped when called as a pipeline-parallel worker
    # (demucs_dir is set by the parent, which already ran the check for both GPUs).
    if demucs_dir is None:
        _warn_if_gpu_busy([device])

    # Resolve Demucs scratch location.
    # --demucs-dir (set by pipeline-parallel) bypasses the interactive check entirely.
    if demucs_dir is not None:
        demucs_path = Path(demucs_dir)
        demucs_on_ramdisk = demucs_path.is_mount()
    else:
        fast_scratch = Path("/mnt/fast_scratch")
        if fast_scratch.is_mount():
            demucs_path = fast_scratch / "demucs"
            demucs_on_ramdisk = True
        else:
            console.print(
                Panel(
                    "[bold yellow]/mnt/fast_scratch is not mounted.[/bold yellow]\n\n"
                    "The RAM disk is not available. Without it, Demucs scratch files will be\n"
                    "written to local storage, which is slower and increases SSD wear.\n\n"
                    f"  Fallback path: [bold]{base_path / 'demucs'}[/bold]\n\n"
                    "To mount the RAM disk before running:\n"
                    "  [dim]sudo mount -t tmpfs -o size=32G,mode=1777 tmpfs /mnt/fast_scratch[/dim]",
                    title="[yellow bold]RAM Disk Not Available[/yellow bold]",
                    border_style="yellow",
                )
            )
            if not click.confirm("Continue using local storage for Demucs scratch?", default=False):
                console.print("[dim]Aborted.[/dim]")
                sys.exit(0)
            demucs_path = base_path / "demucs"
            demucs_on_ramdisk = False

    try:
        demucs_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        console.print(
            Panel(
                "[bold yellow]/mnt/fast_scratch is not writable.[/bold yellow]\n\n"
                "The RAM disk is mounted but the current user cannot write to it.\n"
                "Remount with open permissions:\n\n"
                "  [dim]sudo mount -o remount,mode=1777 /mnt/fast_scratch[/dim]\n\n"
                f"  Fallback path: [bold]{base_path / 'demucs'}[/bold]",
                title="[yellow bold]RAM Disk Not Writable[/yellow bold]",
                border_style="yellow",
            )
        )
        if not click.confirm("Continue using local storage for Demucs scratch?", default=False):
            console.print("[dim]Aborted.[/dim]")
            sys.exit(0)
        demucs_path = base_path / "demucs"
        demucs_on_ramdisk = False
        demucs_path.mkdir(parents=True, exist_ok=True)
    for path in [diar_path, tx_path, summary_dir]:
        path.mkdir(parents=True, exist_ok=True)
    if sentiment:
        sentiment_dir.mkdir(parents=True, exist_ok=True)

    # Warn about unimplemented steps that were requested.
    if emotion:
        console.print(
            Panel(
                "Step 5 (Speech Emotion Recognition) is not yet implemented.\n"
                "The --emotion flag is accepted but has no effect in this release.",
                title="[yellow bold]Step 5 Not Available[/yellow bold]",
                border_style="yellow",
            )
        )
    if events:
        console.print(
            Panel(
                "Step 6 (Audio Event Detection via CLAP) is not yet implemented.\n"
                "The --events flag is accepted; no_vocals.wav stems will be retained\n"
                "on the scratch disk for when CLAP support is added.",
                title="[yellow bold]Step 6 Not Available[/yellow bold]",
                border_style="yellow",
            )
        )

    scratch_suffix = "[dim](RAM disk)[/dim]" if demucs_on_ramdisk else "[yellow](disk — RAM disk not mounted)[/yellow]"
    scratch_label = f"{demucs_path} {scratch_suffix}"

    steps_active = "1 · Separate  2 · Diarize  3 · Transcribe"
    if sentiment:
        steps_active += "  4 · Sentiment"
    if emotion:
        steps_active += "  [dim]5 · Emotion (pending)[/dim]"
    if events:
        steps_active += "  [dim]6 · Events (pending)[/dim]"

    sentiment_line = f"\n[bold]Sentiment:[/bold]          {sentiment_dir}" if sentiment else ""
    console.print(
        Panel(
            f"[bold]Base dir:[/bold]           {base_path}\n"
            f"[bold]Source:[/bold]             {source_path}\n"
            f"[bold]Demucs scratch:[/bold]     {scratch_label}\n"
            f"[bold]Diarization:[/bold]        {diar_path}\n"
            f"[bold]Transcription:[/bold]      {tx_path}"
            f"{sentiment_line}\n"
            f"[bold]Summary:[/bold]            {summary_dir}\n"
            f"[bold]Device:[/bold]             {device}\n"
            f"[bold]Language:[/bold]           {language}\n"
            f"[bold]Steps:[/bold]              {steps_active}\n"
            f"[bold]Resume:[/bold]             {'yes' if resume else 'no'}",
            title="[blue bold]Audio Refinery — Batch Pipeline[/blue bold]",
            border_style="blue",
        )
    )

    with console.status("[bold]Discovering files..."):
        files = discover_files(source_path)

    manifest_ids: list[str] | None = None
    if manifest is not None:
        manifest_ids = [line.strip() for line in Path(manifest).read_text().splitlines() if line.strip()]
        manifest_set = set(manifest_ids)
        files = [(cid, p) for cid, p in files if cid in manifest_set]

    if not files:
        console.print("[yellow]No audio_*.wav files found in source directory.[/yellow]")
        return

    total = len(files)
    console.print(f"[bold]Discovered {total} audio files.[/bold]\n")

    _progress_file = Path(progress_file) if progress_file else None
    if _progress_file:
        _progress_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Temperature monitoring ──────────────────────────────────────────────
    _cuda_idx = int(device.split(":")[1]) if ":" in device else 0
    # Mutable dict used as a closure cell for the throttled temperature read in _on_progress.
    _temp_state: dict = {"value": None, "ts": 0.0, "readings": []}
    _stop_temp = threading.Event()
    if device != "cpu" and temp_limit > 0:
        threading.Thread(target=_run_temp_guard, args=(device, temp_limit, _stop_temp), daemon=True).start()

    t0 = time.monotonic()

    _stage_colours = {"separate": "blue", "diarize": "yellow", "transcribe": "green", "sentiment": "magenta"}
    _stage_labels = {"separate": "S1·Sep", "diarize": "S2·Diar", "transcribe": "S3·Tx", "sentiment": "S4·Sent"}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Starting…", total=total)

        def _on_progress(content_id: str, stage: str, i: int, n: int) -> None:
            if stage == "done":
                # Final signal emitted by run_pipeline after all files are processed.
                if not sys.stdout.isatty():
                    click.echo(f"[{time.strftime('%H:%M:%S')}] Done {i}/{n}")
                if _progress_file is not None:
                    _progress_file.write_text(json.dumps({"done": i, "total": n, "current": "", "stage": "done"}))
                return
            colour = _stage_colours.get(stage, "cyan")
            label = _stage_labels.get(stage, stage)
            # Update temperature reading at most once every 5 s.
            now = time.monotonic()
            if device != "cpu" and now - _temp_state["ts"] >= 5.0:
                _t = query_gpu_temperature(_cuda_idx)
                _temp_state["value"] = _t
                _temp_state["ts"] = now
                if _t is not None:
                    _temp_state["readings"].append(_t)
            temp_str = f" · {_fmt_temp(_temp_state['value'], temp_limit)}" if device != "cpu" else ""
            progress.update(task, completed=i, description=f"[{colour}]{label}[/{colour}] · {content_id}{temp_str}")
            # Plain-text fallback when stdout is redirected to a log file (no TTY).
            if not sys.stdout.isatty():
                click.echo(f"[{time.strftime('%H:%M:%S')}] {label} {i}/{n} · {content_id}")
            # Machine-readable progress snapshot for pipeline-parallel's live display.
            if _progress_file is not None:
                _progress_file.write_text(json.dumps({"done": i, "total": n, "current": content_id, "stage": label}))

        try:
            pipeline_result = run_pipeline(
                source_dir=source_path,
                demucs_output_dir=demucs_path,
                diarization_dir=diar_path,
                transcription_dir=tx_path,
                sentiment_dir=sentiment_dir,
                device=device,
                segment=segment,
                compute_type=compute_type,
                batch_size=batch_size,
                language=language,
                hf_token=hf_token,
                resume=resume,
                enable_sentiment=sentiment,
                enable_emotion=emotion,
                enable_events=events,
                keep_scratch=keep_scratch,
                on_progress=_on_progress,
                manifest=manifest_ids,
                whisper_model=whisper_model,
            )
        finally:
            _stop_temp.set()

        progress.update(task, completed=total, description="[green]Done")

    total_time = time.monotonic() - t0

    # ── GPU temperature summary (peak + avg over all readings collected) ───
    _readings = _temp_state.get("readings", [])
    _gpu_temp_stats: dict | None = None
    if _readings and device != "cpu":
        _gpu_temp_stats = {
            "device": device,
            "peak_celsius": max(_readings),
            "avg_celsius": round(sum(_readings) / len(_readings), 1),
            "sample_count": len(_readings),
        }

    sep_result = pipeline_result.separation
    diar_result = pipeline_result.diarization
    tx_result = pipeline_result.transcription
    sent_result = pipeline_result.sentiment

    # ── Per-stage timing (processed files only; skipped excluded) ─────────
    def _stage_stats(stage_result):
        times = [o.processing_time_seconds for o in stage_result.outcomes if o.success and not o.skipped]
        stage_total = sum(times)
        return stage_total, (stage_total / len(times) if times else 0.0)

    sep_time, sep_avg = _stage_stats(sep_result)
    diar_time, diar_avg = _stage_stats(diar_result)
    tx_time, tx_avg = _stage_stats(tx_result)
    sent_time, sent_avg = _stage_stats(sent_result)

    completed = tx_result.n_succeeded + tx_result.n_skipped
    pipeline_avg = total_time / completed if completed else 0.0

    # ── Summary table ─────────────────────────────────────────────────────
    summary = Table(title="Pipeline Summary", border_style="green")
    summary.add_column("Stage", style="bold")
    summary.add_column("Processed", justify="right")
    summary.add_column("Skipped", justify="right")
    summary.add_column("Failed", justify="right")
    summary.add_column("Stage Time", justify="right")
    summary.add_column("Avg/File", justify="right")
    summary.add_row(
        "Vocal separation",
        str(sep_result.n_succeeded),
        str(sep_result.n_skipped),
        str(sep_result.n_failed),
        _fmt_time(sep_time),
        _fmt_time(sep_avg),
    )
    summary.add_row(
        "Speaker diarization",
        str(diar_result.n_succeeded),
        str(diar_result.n_skipped),
        str(diar_result.n_failed),
        _fmt_time(diar_time),
        _fmt_time(diar_avg),
    )
    summary.add_row(
        "Transcription",
        str(tx_result.n_succeeded),
        str(tx_result.n_skipped),
        str(tx_result.n_failed),
        _fmt_time(tx_time),
        _fmt_time(tx_avg),
    )
    if sentiment:
        summary.add_row(
            "Text sentiment",
            str(sent_result.n_succeeded),
            str(sent_result.n_skipped),
            str(sent_result.n_failed),
            _fmt_time(sent_time),
            _fmt_time(sent_avg),
        )
    console.print(summary)
    console.print(f"[dim]Total: {_fmt_time(total_time)}  ·  Avg/file (full pipeline): {_fmt_time(pipeline_avg)}[/dim]")

    all_failures: list[_FileOutcome] = (
        sep_result.failed_outcomes
        + diar_result.failed_outcomes
        + tx_result.failed_outcomes
        + (sent_result.failed_outcomes if sentiment else [])
    )
    # Overwrite the progress file with the definitive failure count so the
    # pipeline-parallel coordinator can show "Done (N failed)" rather than a
    # misleading green "Done" when some files failed.
    if _progress_file is not None:
        _progress_file.write_text(
            json.dumps({"done": total, "total": total, "current": "", "stage": "done", "failures": len(all_failures)})
        )
    if all_failures:
        fail_table = Table(title="Failed Files", border_style="red")
        fail_table.add_column("Stage", style="bold")
        fail_table.add_column("Content ID")
        fail_table.add_column("Error")
        for f in all_failures:
            fail_table.add_row(f.stage, f.content_id, f.error or "unknown error")
        console.print(fail_table)

    # ── Persist summary JSON ───────────────────────────────────────────────
    from datetime import datetime, timezone

    stages_data = {
        "separation": {
            "processed": sep_result.n_succeeded,
            "skipped": sep_result.n_skipped,
            "skipped_ids": sep_result.skipped_ids,
            "failed": sep_result.n_failed,
            "stage_time_seconds": round(sep_time, 2),
            "avg_time_per_file_seconds": round(sep_avg, 2),
            "source_audio_bytes": sep_result.total_input_bytes,
            "output_bytes": sep_result.total_output_bytes,
            "avg_rtf": sep_result.avg_rtf,
            "peak_vram_bytes": sep_result.peak_vram_bytes,
        },
        "diarization": {
            "processed": diar_result.n_succeeded,
            "skipped": diar_result.n_skipped,
            "skipped_ids": diar_result.skipped_ids,
            "failed": diar_result.n_failed,
            "stage_time_seconds": round(diar_time, 2),
            "avg_time_per_file_seconds": round(diar_avg, 2),
            "input_bytes": diar_result.total_input_bytes,
            "output_bytes": diar_result.total_output_bytes,
            "avg_rtf": diar_result.avg_rtf,
            "peak_vram_bytes": diar_result.peak_vram_bytes,
            "total_segments": diar_result.total_segments,
        },
        "transcription": {
            "processed": tx_result.n_succeeded,
            "skipped": tx_result.n_skipped,
            "skipped_ids": tx_result.skipped_ids,
            "failed": tx_result.n_failed,
            "stage_time_seconds": round(tx_time, 2),
            "avg_time_per_file_seconds": round(tx_avg, 2),
            "input_bytes": tx_result.total_input_bytes,
            "output_bytes": tx_result.total_output_bytes,
            "avg_rtf": tx_result.avg_rtf,
            "peak_vram_bytes": tx_result.peak_vram_bytes,
            "total_words": tx_result.total_words,
            "total_segments": tx_result.total_segments,
        },
    }
    if sentiment:
        stages_data["sentiment"] = {
            "processed": sent_result.n_succeeded,
            "skipped": sent_result.n_skipped,
            "skipped_ids": sent_result.skipped_ids,
            "failed": sent_result.n_failed,
            "stage_time_seconds": round(sent_time, 2),
            "avg_time_per_file_seconds": round(sent_avg, 2),
            "input_bytes": sent_result.total_input_bytes,
            "output_bytes": sent_result.total_output_bytes,
            "total_segments": sent_result.total_segments,
        }

    summary_data = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_discovered": total,
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_file_seconds": round(pipeline_avg, 2),
        "total_audio_hours": round(pipeline_result.total_audio_duration_seconds / 3600, 4),
        "source_audio_bytes": sep_result.total_input_bytes,
        "total_words": pipeline_result.total_words,
        "total_segments": pipeline_result.total_segments,
        "gpu_temp_celsius": _gpu_temp_stats,
        "steps_enabled": {
            "separation": True,
            "diarization": True,
            "transcription": True,
            "sentiment": sentiment,
            "emotion": emotion,
            "events": events,
        },
        "stages": stages_data,
        "failures": [{"stage": f.stage, "content_id": f.content_id, "error": f.error} for f in all_failures],
    }
    resolved_summary_path = Path(summary_file) if summary_file else summary_dir / "pipeline_summary.json"
    resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_summary_path.write_text(json.dumps(summary_data, indent=2))
    console.print(f"[dim]Summary written to: {resolved_summary_path}[/dim]")

    if keep_scratch:
        console.print(f"[dim]Scratch kept: {demucs_path}[/dim]")

    colour = "green" if not all_failures else "yellow"
    console.print(f"\n[bold {colour}]Complete.[/bold {colour}] {completed}/{total} files transcribed successfully.")

    notify_pipeline_complete(
        device=device,
        total=total,
        completed=completed,
        failures=len(all_failures),
        elapsed_seconds=total_time,
    )

    if all_failures:
        sys.exit(1)


@cli.command("pipeline-parallel")
@click.option(
    "--base-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    required=True,
    help=(
        "Base directory for the pipeline run. "
        "Source WAV files must exist in <base>/extracted. "
        "Output subdirectories are created automatically."
    ),
)
@click.option(
    "--device",
    "devices",
    type=str,
    multiple=True,
    default=(),
    show_default=False,
    help=(
        "GPU device for a worker. Repeat for each worker: --device cuda:0 --device cuda:1. "
        "Order determines partition priority — first device gets the largest files. "
        "Defaults to all GPUs detected by nvidia-smi, ranked best-first."
    ),
)
@click.option(
    "--segment",
    type=int,
    default=None,
    help="Demucs segment size in seconds for VRAM optimisation.",
)
@click.option(
    "--compute-type",
    type=click.Choice(["float16", "int8_float16", "int8", "float32"]),
    default=TRANSCRIBER_DEFAULT_COMPUTE_TYPE,
    show_default=True,
    help="WhisperX CTranslate2 compute type.",
)
@click.option(
    "--batch-size",
    type=int,
    default=TRANSCRIBER_DEFAULT_BATCH_SIZE,
    show_default=True,
    help="WhisperX batch size.",
)
@click.option(
    "--language",
    default=TRANSCRIBER_DEFAULT_LANGUAGE,
    show_default=True,
    help="Language code (e.g. 'en') or 'auto' for detection.",
)
@click.option(
    "--hf-token",
    default=None,
    help="HuggingFace token for Pyannote (overrides HF_TOKEN env var).",
)
@click.option(
    "--no-resume",
    is_flag=True,
    default=False,
    help="Re-process all files, ignoring existing outputs.",
)
@click.option(
    "--keep-scratch",
    is_flag=True,
    default=False,
    help="Keep Demucs stems on the scratch disk after the run.",
)
@click.option(
    "--whisper-model",
    default="large-v3",
    show_default=True,
    help=(
        "WhisperX model name. Variants: large-v3 (default, highest accuracy), "
        "distil-large-v3 (~2× faster), medium.en (~3.2× faster, English-only), "
        "medium (~3× faster, multilingual)."
    ),
)
@click.option(
    "--temp-limit",
    type=int,
    default=80,
    show_default=True,
    help=(
        "GPU temperature limit in °C. Each worker monitors its own device every 5 s "
        "and shuts down if this threshold is exceeded. The coordinator also enforces it "
        "by terminating the offending worker subprocess. Set to 0 to disable."
    ),
)
@click.option(
    "--power-limit",
    type=int,
    default=None,
    help=(
        "Set nvidia-smi power limit (watts) on each GPU before launch. "
        "Recommended: 350. Requires passwordless sudo for nvidia-smi "
        "(see README for sudoers setup)."
    ),
)
@click.option(
    "--sentiment",
    is_flag=True,
    default=False,
    help="Enable step 4: Text Sentiment Analysis. Forwarded to each worker.",
)
def pipeline_parallel(
    base_dir: str,
    devices: tuple[str, ...],
    segment: int | None,
    compute_type: str,
    batch_size: int,
    language: str,
    hf_token: str | None,
    no_resume: bool,
    keep_scratch: bool,
    whisper_model: str,
    temp_limit: int,
    power_limit: int | None,
    sentiment: bool,
):
    """Launch N simultaneous pipeline workers, one per --device flag.

    Enumerates all files in <base>/extracted, splits them into N interleaved
    partitions, and runs one 'audio-refinery pipeline' process per device. Workers
    are labelled W0, W1, ..., WN-1. Each worker's combined stdout+stderr is
    written to <base>/logs/worker_N.log.

    After all workers finish, a combined failure report is printed and written
    to <base>/summary/combined_report.json.

    Directory layout derived from --base-dir:
      <base>/extracted/     — input WAV files (must exist)
      <base>/manifests/     — auto-generated per-worker manifests
      <base>/logs/          — per-worker log files
      <base>/diarization/   — diarization JSON output (created if absent)
      <base>/transcription/ — transcription JSON output (created if absent)
      <base>/summary/       — per-worker summaries + combined_report.json
    """
    from datetime import datetime, timezone
    from pathlib import Path

    from src.pipeline import discover_files, partition_ids

    # Auto-detect GPU order if no --device flags were given.
    if not devices:
        devices = detect_gpu_order()
        console.print(f"[dim]Auto-detected GPU order: {', '.join(devices)}[/dim]")

    # Validate each device string.
    for d in devices:
        _validate_device(None, None, d)

    base_path = Path(base_dir)
    source_path = base_path / "extracted"

    if not source_path.exists():
        console.print(
            Panel(
                f"[bold]Source directory not found:[/bold] {source_path}\n"
                "Create the directory and place audio_<content_id>.wav files inside it.",
                title="[red bold]Error[/red bold]",
                border_style="red",
            )
        )
        sys.exit(1)

    _warn_if_gpu_busy(list(devices))

    # ── Resolve Demucs scratch (interactive if needed; performed once here) ──
    fast_scratch = Path("/mnt/fast_scratch")
    if fast_scratch.is_mount():
        demucs_path = fast_scratch / "demucs"
        demucs_on_ramdisk = True
    else:
        console.print(
            Panel(
                "[bold yellow]/mnt/fast_scratch is not mounted.[/bold yellow]\n\n"
                "The RAM disk is not available. Without it, Demucs scratch files will be\n"
                "written to local storage, which is slower and increases SSD wear.\n\n"
                f"  Fallback path: [bold]{base_path / 'demucs'}[/bold]\n\n"
                "To mount the RAM disk before running:\n"
                "  [dim]sudo mount -t tmpfs -o size=32G,mode=1777 tmpfs /mnt/fast_scratch[/dim]",
                title="[yellow bold]RAM Disk Not Available[/yellow bold]",
                border_style="yellow",
            )
        )
        if not click.confirm("Continue using local storage for Demucs scratch?", default=False):
            console.print("[dim]Aborted.[/dim]")
            sys.exit(0)
        demucs_path = base_path / "demucs"
        demucs_on_ramdisk = False

    # ── Create working directories ──────────────────────────────────────────
    manifests_dir = base_path / "manifests"
    logs_dir = base_path / "logs"
    summary_dir = base_path / "summary"
    try:
        demucs_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        console.print(
            Panel(
                "[bold yellow]/mnt/fast_scratch is not writable.[/bold yellow]\n\n"
                "The RAM disk is mounted but the current user cannot write to it.\n"
                "Remount with open permissions:\n\n"
                "  [dim]sudo mount -o remount,mode=1777 /mnt/fast_scratch[/dim]\n\n"
                f"  Fallback path: [bold]{base_path / 'demucs'}[/bold]",
                title="[yellow bold]RAM Disk Not Writable[/yellow bold]",
                border_style="yellow",
            )
        )
        if not click.confirm("Continue using local storage for Demucs scratch?", default=False):
            console.print("[dim]Aborted.[/dim]")
            sys.exit(0)
        demucs_path = base_path / "demucs"
        demucs_on_ramdisk = False
        demucs_path.mkdir(parents=True, exist_ok=True)
    for path in [manifests_dir, logs_dir, summary_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # ── Discover and partition files ────────────────────────────────────────
    with console.status("[bold]Discovering files..."):
        all_files = discover_files(source_path)

    if not all_files:
        console.print("[yellow]No audio_*.wav files found in source directory.[/yellow]")
        return

    # Sort largest-first (LPT heuristic) before interleaving so that the
    # biggest files go to different workers, minimising the slower worker's total.
    all_files_sorted = sorted(all_files, key=lambda x: x[1].stat().st_size, reverse=True)
    all_ids = [cid for cid, _ in all_files_sorted]
    partitions = partition_ids(all_ids, n=len(devices))

    # ── Build worker dict list ──────────────────────────────────────────────
    workers: list[dict] = []
    for i, device in enumerate(devices):
        label = f"W{i}"
        workers.append(
            {
                "index": i,
                "label": label,
                "device": device,
                "ids": partitions[i],
                "manifest": manifests_dir / f"manifest_{i}.txt",
                "summary": summary_dir / f"worker_{i}.json",
                "log": logs_dir / f"worker_{i}.log",
                "progress": logs_dir / f"progress_{i}.json",
                "fh": None,
                "proc": None,
                "rc": None,
            }
        )

    # Write per-worker manifest files.
    for w in workers:
        w["manifest"].write_text("\n".join(w["ids"]) + ("\n" if w["ids"] else ""))

    # ── Optional power limit ────────────────────────────────────────────────
    if power_limit is not None:
        for device in devices:
            idx = device.split(":")[1] if ":" in device else "0"
            pl_result = subprocess.run(
                ["sudo", "nvidia-smi", "-i", idx, "-pl", str(power_limit)],
                capture_output=True,
                text=True,
            )
            if pl_result.returncode != 0:
                detail = pl_result.stderr.strip() or pl_result.stdout.strip() or "unknown error"
                console.print(f"[yellow]Warning: nvidia-smi power limit failed for {device}: {detail}[/yellow]")
                console.print("[yellow]  Tip: add a sudoers rule to allow passwordless nvidia-smi:[/yellow]")
                console.print(
                    "[yellow]  echo 'USER ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi'"
                    " | sudo tee /etc/sudoers.d/nvidia-smi[/yellow]"
                )
            else:
                console.print(f"[dim]Power limit {power_limit}W set on {device}.[/dim]")

    # ── Build worker commands ───────────────────────────────────────────────
    refinery_cmd = sys.argv[0]

    def _build_worker_cmd(device: str, manifest_path: Path, summary_path: Path, progress_path: Path) -> list[str]:
        cmd = [
            refinery_cmd,
            "pipeline",
            "--base-dir",
            str(base_path),
            "--device",
            device,
            "--demucs-dir",
            str(demucs_path),
            "--manifest",
            str(manifest_path),
            "--summary-file",
            str(summary_path),
            "--progress-file",
            str(progress_path),
            "--compute-type",
            compute_type,
            "--batch-size",
            str(batch_size),
            "--language",
            language,
            "--whisper-model",
            whisper_model,
        ]
        if segment is not None:
            cmd += ["--segment", str(segment)]
        if hf_token is not None:
            cmd += ["--hf-token", hf_token]
        if temp_limit > 0:
            cmd += ["--temp-limit", str(temp_limit)]
        else:
            cmd += ["--temp-limit", "0"]
        if no_resume:
            cmd.append("--no-resume")
        if keep_scratch:
            cmd.append("--keep-scratch")
        if sentiment:
            cmd.append("--sentiment")
        return cmd

    # ── Print launch summary ────────────────────────────────────────────────
    scratch_suffix = "(RAM disk)" if demucs_on_ramdisk else "(disk)"
    tflops_table = load_tflops_table()

    def _gpu_stat_line(device: str) -> str:
        idx = int(device.split(":")[1]) if ":" in device else 0
        info = query_gpu_info(idx)
        if info is None:
            return "[dim]GPU info unavailable[/dim]"
        vram_gb = round(info.vram_mib / 1024)
        tflops = lookup_tflops(info.name, tflops_table)
        if tflops is not None:
            return f"[dim]{info.name}  ·  {vram_gb} GB  ·  {tflops} TFLOPs[/dim]"
        return (
            f"[dim]{info.name}  ·  {vram_gb} GB  ·  {info.sm_clock_mhz} MHz[/dim]"
            f"  [yellow][heuristic — add to gpu_tflops.toml][/yellow]"
        )

    worker_lines = "\n".join(
        f"[bold]Worker {w['label']}:[/bold]  {w['device']}  ({len(w['ids'])} files)  →  {w['log']}\n"
        f"            {_gpu_stat_line(w['device'])}"
        for w in workers
    )
    console.print(
        Panel(
            f"[bold]Base dir:[/bold]       {base_path}\n"
            f"[bold]Total files:[/bold]    {len(all_ids)}\n"
            f"[bold]Demucs scratch:[/bold] {demucs_path} {scratch_suffix}\n"
            f"{worker_lines}\n"
            f"[bold]Model:[/bold]          {whisper_model}  [{compute_type}]",
            title="[blue bold]Audio Refinery — Parallel Pipeline[/blue bold]",
            border_style="blue",
            expand=False,
        )
    )
    log_lines = "\n".join(f"  [dim]tail -f {w['log']}[/dim]" for w in workers)
    console.print(f"[dim]Monitor workers in separate terminals:[/dim]\n{log_lines}\n")

    # ── Launch workers ──────────────────────────────────────────────────────
    from rich import box as rich_box
    from rich.live import Live

    t0 = time.monotonic()
    gpu_temps: dict[str, int | None] = {w["device"]: None for w in workers}
    gpu_temp_readings: dict[str, list[int]] = {w["device"]: [] for w in workers}

    def _read_progress(path: Path) -> dict:
        try:
            return json.loads(path.read_text())
        except Exception:
            return {"done": 0, "total": "?", "current": "—", "stage": "starting"}

    def _worker_status_table() -> Table:
        elapsed = time.monotonic() - t0
        tbl = Table(
            title=f"[bold]Parallel Pipeline[/bold]  [dim]{_fmt_time(elapsed)} elapsed[/dim]",
            title_justify="left",
            box=rich_box.SIMPLE,
            border_style="blue",
            show_header=True,
        )
        tbl.add_column("Worker", style="bold", width=8)
        tbl.add_column("Device", width=10)
        tbl.add_column("Temp", width=8)
        tbl.add_column("Stage", width=10)
        tbl.add_column("File")
        tbl.add_column("Progress", justify="right", width=10)
        for w in workers:
            p = _read_progress(w["progress"])
            done, total_w = p.get("done", 0), p.get("total", "?")
            stage_val = p.get("stage", "—")
            if stage_val == "done":
                n_failures = p.get("failures", None)
                stage_display = "[yellow]Done[/yellow]" if n_failures else "[green]Done[/green]"
                file_display = ""
            else:
                stage_display, file_display = stage_val, p.get("current", "—")
            temp_display = _fmt_temp(gpu_temps.get(w["device"]), temp_limit)
            tbl.add_row(w["label"], w["device"], temp_display, stage_display, file_display, f"{done}/{total_w}")
        return tbl

    try:
        for w in workers:
            w["fh"] = w["log"].open("w")
            w["proc"] = subprocess.Popen(
                _build_worker_cmd(w["device"], w["manifest"], w["summary"], w["progress"]),
                stdout=w["fh"],
                stderr=subprocess.STDOUT,
            )
            console.print(f"[dim]Worker {w['label']} PID: {w['proc'].pid}[/dim]")

        with Live(_worker_status_table(), console=console, refresh_per_second=2) as live:
            while True:
                for w in workers:
                    _idx = int(w["device"].split(":")[1]) if ":" in w["device"] else 0
                    _t = query_gpu_temperature(_idx)
                    gpu_temps[w["device"]] = _t
                    if _t is not None:
                        gpu_temp_readings[w["device"]].append(_t)
                live.update(_worker_status_table())
                for w in workers:
                    if w["rc"] is None:
                        w["rc"] = w["proc"].poll()
                # Coordinator-side thermal protection (belt-and-suspenders with worker's own guard).
                if temp_limit > 0:
                    for w in workers:
                        if w["rc"] is None:
                            _t = gpu_temps.get(w["device"])
                            if _t is not None and _t >= temp_limit:
                                live.console.print(
                                    f"[bold red]Thermal shutdown: Worker {w['label']} ({w['device']}) "
                                    f"at {_t}°C ≥ {temp_limit}°C — terminating.[/bold red]"
                                )
                                w["proc"].terminate()
                if all(w["rc"] is not None for w in workers):
                    break
                time.sleep(0.5)
    finally:
        for w in workers:
            if w["fh"] is not None:
                w["fh"].close()

    total_time = time.monotonic() - t0

    # ── Report exit status ──────────────────────────────────────────────────
    for w in workers:
        ok = w["rc"] == 0
        status = "[green]OK[/green]" if ok else f"[red]FAILED (exit {w['rc']})[/red]"
        console.print(f"Worker {w['label']} ({w['device']}): {status}")
    console.print(f"[dim]Total wall-clock time: {_fmt_time(total_time)}[/dim]\n")

    # ── Aggregate summaries ─────────────────────────────────────────────────
    def _load_summary(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    all_combined_failures: list[dict] = []
    worker_summaries: list[dict | None] = []
    for w in workers:
        summary = _load_summary(w["summary"])
        worker_summaries.append(summary)
        if summary:
            for f in summary.get("failures", []):
                all_combined_failures.append({"worker": w["label"], "device": w["device"], **f})

    _notif_processed = 0
    _notif_failures = 0

    if any(s for s in worker_summaries):
        combined = Table(title="Combined Pipeline Summary", border_style="green")
        combined.add_column("Stage", style="bold")
        combined.add_column("Processed", justify="right")
        combined.add_column("Skipped", justify="right")
        combined.add_column("Failed", justify="right")

        def _agg(key: str, sub: str) -> int:
            agg_total = 0
            for s in worker_summaries:
                if s:
                    agg_total += s.get("stages", {}).get(key, {}).get(sub, 0)
            return agg_total

        stage_rows = [
            ("separation", "Vocal separation"),
            ("diarization", "Speaker diarization"),
            ("transcription", "Transcription"),
        ]
        if sentiment:
            stage_rows.append(("sentiment", "Text sentiment"))
        for stage_key, stage_label in stage_rows:
            combined.add_row(
                stage_label,
                str(_agg(stage_key, "processed")),
                str(_agg(stage_key, "skipped")),
                str(_agg(stage_key, "failed")),
            )
        console.print(combined)

        all_failures_count = (
            _agg("separation", "failed") + _agg("diarization", "failed") + _agg("transcription", "failed")
        )
        total_processed = _agg("transcription", "processed") + _agg("transcription", "skipped")
        _notif_processed = total_processed
        _notif_failures = all_failures_count
        colour = "green" if (all(w["rc"] == 0 for w in workers) and all_failures_count == 0) else "yellow"

        # ── Timing summary ─────────────────────────────────────────────────
        combined_avg = total_time / total_processed if total_processed else 0.0
        for w, w_summary in zip(workers, worker_summaries):
            avg = w_summary.get("avg_time_per_file_seconds", 0.0) if w_summary else 0.0
            console.print(f"[dim]Worker {w['label']} ({w['device']}):  avg/file {_fmt_time(avg)}[/dim]")
        console.print(
            f"[dim]Combined:  avg/file {_fmt_time(combined_avg)}  ·  wall-clock {_fmt_time(total_time)}[/dim]"
        )

        console.print(f"[bold {colour}]Complete.[/bold {colour}] {total_processed}/{len(all_ids)} files transcribed.")

    # ── Combined failure report (printed) ──────────────────────────────────
    if all_combined_failures:
        fail_table = Table(
            title=f"Combined Failure Report ({len(all_combined_failures)} failures)",
            border_style="red",
        )
        fail_table.add_column("Worker", style="bold")
        fail_table.add_column("Device")
        fail_table.add_column("Stage")
        fail_table.add_column("Content ID")
        fail_table.add_column("Error")
        for f in all_combined_failures:
            fail_table.add_row(
                f.get("worker", ""),
                f.get("device", ""),
                f.get("stage", ""),
                f.get("content_id", ""),
                f.get("error", ""),
            )
        console.print(fail_table)

    # ── Write combined_report.json (always) ────────────────────────────────
    _gpu_temp_summaries: dict | None = {
        device: {
            "peak_celsius": max(readings),
            "avg_celsius": round(sum(readings) / len(readings), 1),
            "sample_count": len(readings),
        }
        for device, readings in gpu_temp_readings.items()
        if readings
    } or None
    combined_report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_discovered": len(all_ids),
        "total_time_seconds": round(total_time, 2),
        "total_audio_hours": round(sum(s.get("total_audio_hours", 0.0) for s in worker_summaries if s), 4),
        "source_audio_bytes": sum(s.get("source_audio_bytes", 0) for s in worker_summaries if s),
        "total_words": sum(s.get("total_words", 0) for s in worker_summaries if s),
        "total_segments": sum(s.get("total_segments", 0) for s in worker_summaries if s),
        "gpu_temp_celsius": _gpu_temp_summaries,
        "workers": [
            {
                "label": w["label"],
                "device": w["device"],
                "exit_code": w["rc"],
                "summary": worker_summaries[i],
            }
            for i, w in enumerate(workers)
        ],
        "combined_failures": all_combined_failures,
    }
    combined_report_path = summary_dir / "combined_report.json"
    combined_report_path.write_text(json.dumps(combined_report, indent=2))
    console.print(f"[dim]Combined report written to: {combined_report_path}[/dim]")

    worker_statuses = [(w["label"], w["device"], w["rc"] == 0) for w in workers]
    notify_pipeline_parallel_complete(
        worker_statuses=worker_statuses,
        total_discovered=len(all_ids),
        total_processed=_notif_processed,
        failures=_notif_failures,
        elapsed_seconds=total_time,
    )

    if not all(w["rc"] == 0 for w in workers):
        failed_logs = "\n".join(f"  {w['log']}" for w in workers)
        console.print(f"[dim]Worker logs retained for inspection:[/dim]\n{failed_logs}")
        sys.exit(1)


cli.add_command(separate_cmd, name="separate")
cli.add_command(diarize_cmd, name="diarize")
cli.add_command(transcribe_cmd, name="transcribe")
