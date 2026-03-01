"""Tests for the batch pipeline module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline import (
    FileOutcome,
    StageResult,
    _cleanup_stem,
    _diarization_path,
    _file_complete,
    _no_vocals_path,
    _transcription_path,
    _vocals_path,
    discover_files,
    run_diarization_stage,
    run_pipeline,
    run_separation_stage,
    run_transcription_stage,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_wav(path: Path) -> Path:
    """Write a minimal non-empty file so _file_complete() is satisfied."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 64)
    return path


def _sep_success(content_id: str) -> FileOutcome:
    return FileOutcome(content_id=content_id, stage="separate", success=True)


def _sep_skipped(content_id: str) -> FileOutcome:
    return FileOutcome(content_id=content_id, stage="separate", success=True, skipped=True)


def _sep_failed(content_id: str, error: str = "boom") -> FileOutcome:
    return FileOutcome(content_id=content_id, stage="separate", success=False, error=error)


@pytest.fixture()
def source_dir(tmp_path: Path) -> Path:
    d = tmp_path / "extracted"
    d.mkdir()
    for cid in ["abc123", "def456", "ghi789"]:
        _make_wav(d / f"audio_{cid}.wav")
    return d


@pytest.fixture()
def three_files(source_dir: Path) -> list[tuple[str, Path]]:
    return discover_files(source_dir)


@pytest.fixture()
def mock_sep_result() -> MagicMock:
    r = MagicMock()
    r.processing_time_seconds = 5.0
    return r


@pytest.fixture()
def mock_diar_result() -> MagicMock:
    r = MagicMock()
    r.processing_time_seconds = 3.0
    r.model_dump_json.return_value = '{"ok": true}'
    return r


@pytest.fixture()
def mock_tx_result() -> MagicMock:
    r = MagicMock()
    r.processing_time_seconds = 10.0
    r.model_dump_json.return_value = '{"ok": true}'
    return r


# ---------------------------------------------------------------------------
# discover_files
# ---------------------------------------------------------------------------


def test_discover_files_finds_all(source_dir: Path) -> None:
    files = discover_files(source_dir)
    assert len(files) == 3
    ids = {cid for cid, _ in files}
    assert ids == {"abc123", "def456", "ghi789"}


def test_discover_files_empty_dir(tmp_path: Path) -> None:
    assert discover_files(tmp_path) == []


def test_discover_files_ignores_non_matching(source_dir: Path) -> None:
    _make_wav(source_dir / "other.wav")
    (source_dir / "audio_something.mp3").write_bytes(b"x")
    files = discover_files(source_dir)
    assert len(files) == 3  # only the original three


def test_discover_files_sorted(source_dir: Path) -> None:
    files = discover_files(source_dir)
    names = [cid for cid, _ in files]
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# _file_complete
# ---------------------------------------------------------------------------


def test_file_complete_missing(tmp_path: Path) -> None:
    assert not _file_complete(tmp_path / "nope.json")


def test_file_complete_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.json"
    p.write_bytes(b"")
    assert not _file_complete(p)


def test_file_complete_non_empty(tmp_path: Path) -> None:
    p = tmp_path / "data.json"
    p.write_text('{"x": 1}')
    assert _file_complete(p)


# ---------------------------------------------------------------------------
# StageResult properties
# ---------------------------------------------------------------------------


def test_stage_result_counts() -> None:
    sr = StageResult(
        outcomes=[
            FileOutcome(content_id="a", stage="s", success=True),
            FileOutcome(content_id="b", stage="s", success=True, skipped=True),
            FileOutcome(content_id="c", stage="s", success=False, error="err"),
        ]
    )
    assert sr.n_succeeded == 1
    assert sr.n_skipped == 1
    assert sr.n_failed == 1
    assert sr.succeeded_ids == {"a", "b"}
    assert len(sr.failed_outcomes) == 1


# ---------------------------------------------------------------------------
# run_separation_stage
# ---------------------------------------------------------------------------


def test_separation_success(three_files: list, mock_sep_result: MagicMock, tmp_path: Path) -> None:
    with patch("src.pipeline.separate", return_value=mock_sep_result):
        result = run_separation_stage(three_files, demucs_output_dir=tmp_path / "demucs", resume=True)

    assert len(result.outcomes) == 3
    assert all(o.success for o in result.outcomes)
    assert all(not o.skipped for o in result.outcomes)


def test_separation_resume_skips_existing(three_files: list, tmp_path: Path) -> None:
    # Pre-create vocals for one file.
    cid = three_files[0][0]
    _make_wav(_vocals_path(cid, tmp_path / "demucs"))

    with patch(
        "src.pipeline.separate",
        return_value=MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0)),
    ) as mock_sep:
        result = run_separation_stage(three_files, demucs_output_dir=tmp_path / "demucs", resume=True)

    assert mock_sep.call_count == 2  # only the two without existing vocals
    skipped = [o for o in result.outcomes if o.skipped]
    assert len(skipped) == 1
    assert skipped[0].content_id == cid


def test_separation_no_resume_reprocesses(three_files: list, tmp_path: Path) -> None:
    # Even with existing vocals.wav, no-resume should process all.
    for cid, _ in three_files:
        _make_wav(_vocals_path(cid, tmp_path / "demucs"))

    with patch(
        "src.pipeline.separate",
        return_value=MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0)),
    ) as mock_sep:
        run_separation_stage(three_files, demucs_output_dir=tmp_path / "demucs", resume=False)

    assert mock_sep.call_count == 3


def test_separation_error_continues(three_files: list, tmp_path: Path) -> None:
    mock_ok = MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0))
    with patch("src.pipeline.separate", side_effect=[Exception("boom"), mock_ok, mock_ok]):
        result = run_separation_stage(three_files, demucs_output_dir=tmp_path / "demucs", resume=True)

    assert len(result.outcomes) == 3
    assert result.outcomes[0].success is False
    assert result.outcomes[0].error == "boom"
    assert result.outcomes[1].success is True
    assert result.outcomes[2].success is True


def test_separation_progress_callback(three_files: list, tmp_path: Path) -> None:
    calls: list[tuple] = []

    def cb(cid: str, i: int, n: int) -> None:
        calls.append((cid, i, n))

    with patch(
        "src.pipeline.separate",
        return_value=MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0)),
    ):
        run_separation_stage(three_files, demucs_output_dir=tmp_path / "demucs", on_file=cb)

    assert len(calls) == 3
    assert calls[0] == (three_files[0][0], 0, 3)


# ---------------------------------------------------------------------------
# run_diarization_stage
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_sep_result(three_files: list) -> StageResult:
    return StageResult(outcomes=[_sep_success(cid) for cid, _ in three_files])


def test_diarization_skips_failed_separation(three_files: list, tmp_path: Path) -> None:
    sep = StageResult(outcomes=[_sep_failed(cid) for cid, _ in three_files])

    with (
        patch("src.pipeline.load_pipeline") as mock_load,
        patch("src.pipeline.diarize") as mock_diar,
    ):
        result = run_diarization_stage(three_files, sep, tmp_path, tmp_path / "diar")

    mock_load.assert_not_called()
    mock_diar.assert_not_called()
    assert len(result.outcomes) == 0


def test_diarization_model_loaded_once(
    three_files: list, full_sep_result: StageResult, tmp_path: Path, mock_diar_result: MagicMock
) -> None:
    for cid, _ in three_files:
        _make_wav(_vocals_path(cid, tmp_path / "demucs"))

    mock_pipeline = MagicMock()
    with (
        patch("src.pipeline.load_pipeline", return_value=mock_pipeline) as mock_load,
        patch("src.pipeline._resolve_hf_token", return_value="tok"),
        patch("src.pipeline.diarize", return_value=mock_diar_result),
    ):
        result = run_diarization_stage(
            three_files, full_sep_result, tmp_path / "demucs", tmp_path / "diar", hf_token="tok"
        )

    mock_load.assert_called_once()
    assert len(result.outcomes) == 3
    assert all(o.success for o in result.outcomes)


def test_diarization_resume_skips_existing(three_files: list, full_sep_result: StageResult, tmp_path: Path) -> None:
    # Pre-create diarization JSON for one file.
    cid = three_files[0][0]
    p = _diarization_path(cid, tmp_path / "diar")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"ok":true}')

    mock_pipeline = MagicMock()
    with (
        patch("src.pipeline.load_pipeline", return_value=mock_pipeline),
        patch("src.pipeline._resolve_hf_token", return_value="tok"),
        patch(
            "src.pipeline.diarize",
            return_value=MagicMock(
                processing_time_seconds=1.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ) as mock_diar,
    ):
        result = run_diarization_stage(
            three_files, full_sep_result, tmp_path, tmp_path / "diar", resume=True, hf_token="tok"
        )

    assert mock_diar.call_count == 2  # one skipped
    skipped = [o for o in result.outcomes if o.skipped]
    assert len(skipped) == 1
    assert skipped[0].content_id == cid


def test_diarization_all_existing_skips_model_load(
    three_files: list, full_sep_result: StageResult, tmp_path: Path
) -> None:
    # Pre-create all diarization JSONs.
    for cid, _ in three_files:
        p = _diarization_path(cid, tmp_path / "diar")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"ok":true}')

    with (
        patch("src.pipeline.load_pipeline") as mock_load,
        patch("src.pipeline._resolve_hf_token"),
    ):
        result = run_diarization_stage(three_files, full_sep_result, tmp_path, tmp_path / "diar", resume=True)

    mock_load.assert_not_called()
    assert all(o.skipped for o in result.outcomes)


def test_diarization_model_load_failure_fails_all(
    three_files: list, full_sep_result: StageResult, tmp_path: Path
) -> None:
    from src.diarizer import DiarizationError

    with (
        patch("src.pipeline._resolve_hf_token", return_value="tok"),
        patch("src.pipeline.load_pipeline", side_effect=DiarizationError("no model")),
    ):
        result = run_diarization_stage(three_files, full_sep_result, tmp_path, tmp_path / "diar", hf_token="tok")

    assert all(not o.success for o in result.outcomes)
    assert all("no model" in o.error for o in result.outcomes)  # type: ignore[operator]


def test_diarization_error_continues(three_files: list, full_sep_result: StageResult, tmp_path: Path) -> None:
    for cid, _ in three_files:
        _make_wav(_vocals_path(cid, tmp_path / "demucs"))

    ok = MagicMock(
        processing_time_seconds=1.0,
        model_dump_json=lambda **kw: "{}",
        input_info=MagicMock(duration_seconds=10.0),
        segments=[],
    )
    with (
        patch("src.pipeline._resolve_hf_token", return_value="tok"),
        patch("src.pipeline.load_pipeline", return_value=MagicMock()),
        patch("src.pipeline.diarize", side_effect=[Exception("oops"), ok, ok]),
    ):
        result = run_diarization_stage(
            three_files, full_sep_result, tmp_path / "demucs", tmp_path / "diar", hf_token="tok"
        )

    assert result.outcomes[0].success is False
    assert result.outcomes[1].success is True
    assert result.outcomes[2].success is True


# ---------------------------------------------------------------------------
# run_transcription_stage
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_diar_result(three_files: list) -> StageResult:
    return StageResult(outcomes=[FileOutcome(content_id=cid, stage="diarize", success=True) for cid, _ in three_files])


def test_transcription_skips_failed_diarization(three_files: list, tmp_path: Path) -> None:
    diar = StageResult(outcomes=[FileOutcome(content_id=cid, stage="diarize", success=False) for cid, _ in three_files])

    with patch("src.pipeline._load_whisperx_model") as mock_load:
        result = run_transcription_stage(three_files, diar, tmp_path, tmp_path / "diar", tmp_path / "tx")

    mock_load.assert_not_called()
    assert len(result.outcomes) == 0


def test_transcription_model_loaded_once(
    three_files: list, full_diar_result: StageResult, tmp_path: Path, mock_tx_result: MagicMock
) -> None:
    for cid, _ in three_files:
        _make_wav(_vocals_path(cid, tmp_path / "demucs"))

    with (
        patch("src.pipeline._load_whisperx_model", return_value=MagicMock()) as mock_load,
        patch("src.pipeline.transcribe", return_value=mock_tx_result),
    ):
        result = run_transcription_stage(
            three_files, full_diar_result, tmp_path / "demucs", tmp_path / "diar", tmp_path / "tx"
        )

    mock_load.assert_called_once()
    assert len(result.outcomes) == 3
    assert all(o.success for o in result.outcomes)


def test_transcription_resume_skips_existing(
    three_files: list, full_diar_result: StageResult, tmp_path: Path, mock_tx_result: MagicMock
) -> None:
    cid = three_files[0][0]
    p = _transcription_path(cid, tmp_path / "tx")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"ok":true}')

    with (
        patch("src.pipeline._load_whisperx_model", return_value=MagicMock()),
        patch("src.pipeline.transcribe", return_value=mock_tx_result) as mock_tx,
    ):
        result = run_transcription_stage(
            three_files, full_diar_result, tmp_path, tmp_path / "diar", tmp_path / "tx", resume=True
        )

    assert mock_tx.call_count == 2
    skipped = [o for o in result.outcomes if o.skipped]
    assert len(skipped) == 1
    assert skipped[0].content_id == cid


def test_transcription_all_existing_skips_model_load(
    three_files: list, full_diar_result: StageResult, tmp_path: Path
) -> None:
    for cid, _ in three_files:
        p = _transcription_path(cid, tmp_path / "tx")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"ok":true}')

    with patch("src.pipeline._load_whisperx_model") as mock_load:
        result = run_transcription_stage(
            three_files, full_diar_result, tmp_path, tmp_path / "diar", tmp_path / "tx", resume=True
        )

    mock_load.assert_not_called()
    assert all(o.skipped for o in result.outcomes)


def test_transcription_model_load_failure_fails_all(
    three_files: list, full_diar_result: StageResult, tmp_path: Path
) -> None:
    with patch("src.pipeline._load_whisperx_model", side_effect=ImportError("no whisperx")):
        result = run_transcription_stage(three_files, full_diar_result, tmp_path, tmp_path / "diar", tmp_path / "tx")

    assert all(not o.success for o in result.outcomes)
    assert all("no whisperx" in o.error for o in result.outcomes)  # type: ignore[operator]


def test_transcription_error_continues(
    three_files: list, full_diar_result: StageResult, tmp_path: Path, mock_tx_result: MagicMock
) -> None:
    for cid, _ in three_files:
        _make_wav(_vocals_path(cid, tmp_path / "demucs"))

    with (
        patch("src.pipeline._load_whisperx_model", return_value=MagicMock()),
        patch("src.pipeline.transcribe", side_effect=[Exception("crash"), mock_tx_result, mock_tx_result]),
    ):
        result = run_transcription_stage(
            three_files, full_diar_result, tmp_path / "demucs", tmp_path / "diar", tmp_path / "tx"
        )

    assert result.outcomes[0].success is False
    assert result.outcomes[1].success is True
    assert result.outcomes[2].success is True


def test_transcription_passes_whisperx_model_to_transcribe(
    three_files: list, full_diar_result: StageResult, tmp_path: Path, mock_tx_result: MagicMock
) -> None:
    for cid, _ in three_files:
        _make_wav(_vocals_path(cid, tmp_path / "demucs"))

    fake_model = MagicMock(name="wx_model")

    with (
        patch("src.pipeline._load_whisperx_model", return_value=fake_model),
        patch("src.pipeline.transcribe", return_value=mock_tx_result) as mock_tx,
    ):
        run_transcription_stage(three_files, full_diar_result, tmp_path / "demucs", tmp_path / "diar", tmp_path / "tx")

    for c in mock_tx.call_args_list:
        assert c.kwargs["_whisperx_model"] is fake_model


# ---------------------------------------------------------------------------
# _cleanup_stem helper
# ---------------------------------------------------------------------------


def test_cleanup_stem_removes_file_and_empty_parent(tmp_path: Path) -> None:
    stem_dir = tmp_path / "model" / "track"
    stem_dir.mkdir(parents=True)
    f = stem_dir / "vocals.wav"
    f.write_bytes(b"x")
    _cleanup_stem(f)
    assert not f.exists()
    assert not stem_dir.exists()  # parent removed because it was empty


def test_cleanup_stem_leaves_nonempty_parent(tmp_path: Path) -> None:
    stem_dir = tmp_path / "model" / "track"
    stem_dir.mkdir(parents=True)
    f = stem_dir / "vocals.wav"
    f.write_bytes(b"x")
    (stem_dir / "no_vocals.wav").write_bytes(b"y")
    _cleanup_stem(f)
    assert not f.exists()
    assert stem_dir.exists()  # sibling kept the directory alive


def test_cleanup_stem_missing_file_is_noop(tmp_path: Path) -> None:
    _cleanup_stem(tmp_path / "nonexistent.wav")  # must not raise


# ---------------------------------------------------------------------------
# run_pipeline — interleaved per-file processing
# ---------------------------------------------------------------------------


def _patch_models(
    mock_sep_result=None,
    mock_diar_result=None,
    mock_tx_result=None,
    diar_side_effect=None,
    tx_side_effect=None,
):
    """Return a context manager that patches all three model entry-points."""
    from contextlib import ExitStack
    from unittest.mock import MagicMock, patch

    stack = ExitStack()

    sep_ret = mock_sep_result or MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0))
    stack.enter_context(patch("src.pipeline.separate", return_value=sep_ret))

    if diar_side_effect:
        stack.enter_context(patch("src.pipeline.load_pipeline", return_value=MagicMock()))
        stack.enter_context(patch("src.pipeline._resolve_hf_token", return_value="tok"))
        stack.enter_context(patch("src.pipeline.diarize", side_effect=diar_side_effect))
    else:
        diar_ret = mock_diar_result or MagicMock(
            processing_time_seconds=2.0,
            model_dump_json=lambda **kw: '{"ok":true}',
            input_info=MagicMock(duration_seconds=10.0),
            segments=[],
        )
        stack.enter_context(patch("src.pipeline.load_pipeline", return_value=MagicMock()))
        stack.enter_context(patch("src.pipeline._resolve_hf_token", return_value="tok"))
        stack.enter_context(patch("src.pipeline.diarize", return_value=diar_ret))

    if tx_side_effect:
        stack.enter_context(patch("src.pipeline._load_whisperx_model", return_value=MagicMock()))
        stack.enter_context(patch("src.pipeline.transcribe", side_effect=tx_side_effect))
    else:
        tx_ret = mock_tx_result or MagicMock(
            processing_time_seconds=5.0,
            model_dump_json=lambda **kw: '{"ok":true}',
            input_info=MagicMock(duration_seconds=10.0),
            segments=[],
        )
        stack.enter_context(patch("src.pipeline._load_whisperx_model", return_value=MagicMock()))
        stack.enter_context(patch("src.pipeline.transcribe", return_value=tx_ret))

    return stack


def test_run_pipeline_empty_source(tmp_path: Path) -> None:
    result = run_pipeline(source_dir=tmp_path / "empty")
    assert result.total_discovered == 0
    assert result.separation.outcomes == []


def test_run_pipeline_no_files_found(tmp_path: Path) -> None:
    (tmp_path / "extracted").mkdir()
    result = run_pipeline(source_dir=tmp_path / "extracted")
    assert result.total_discovered == 0


def test_run_pipeline_models_loaded_once(source_dir: Path, tmp_path: Path) -> None:
    """Pyannote and WhisperX should each be loaded exactly once regardless of file count."""
    with (
        patch(
            "src.pipeline.separate",
            return_value=MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0)),
        ),
        patch("src.pipeline.load_pipeline", return_value=MagicMock()) as mock_load_diar,
        patch("src.pipeline._resolve_hf_token", return_value="tok"),
        patch(
            "src.pipeline.diarize",
            return_value=MagicMock(
                processing_time_seconds=2.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ),
        patch("src.pipeline._load_whisperx_model", return_value=MagicMock()) as mock_load_wx,
        patch(
            "src.pipeline.transcribe",
            return_value=MagicMock(
                processing_time_seconds=5.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ),
    ):
        run_pipeline(
            source_dir=source_dir,
            demucs_output_dir=tmp_path / "demucs",
            diarization_dir=tmp_path / "diar",
            transcription_dir=tmp_path / "tx",
        )

    mock_load_diar.assert_called_once()
    mock_load_wx.assert_called_once()


def test_run_pipeline_all_three_stages_run_per_file(source_dir: Path, tmp_path: Path) -> None:
    """separate, diarize, and transcribe should each be called once per file."""
    with (
        patch(
            "src.pipeline.separate",
            return_value=MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0)),
        ) as mock_sep,
        patch("src.pipeline.load_pipeline", return_value=MagicMock()),
        patch("src.pipeline._resolve_hf_token", return_value="tok"),
        patch(
            "src.pipeline.diarize",
            return_value=MagicMock(
                processing_time_seconds=2.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ) as mock_diar,
        patch("src.pipeline._load_whisperx_model", return_value=MagicMock()),
        patch(
            "src.pipeline.transcribe",
            return_value=MagicMock(
                processing_time_seconds=5.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ) as mock_tx,
    ):
        result = run_pipeline(
            source_dir=source_dir,
            demucs_output_dir=tmp_path / "demucs",
            diarization_dir=tmp_path / "diar",
            transcription_dir=tmp_path / "tx",
        )

    assert result.total_discovered == 3
    assert mock_sep.call_count == 3
    assert mock_diar.call_count == 3
    assert mock_tx.call_count == 3
    assert result.separation.n_succeeded == 3
    assert result.diarization.n_succeeded == 3
    assert result.transcription.n_succeeded == 3


def test_run_pipeline_separation_failure_skips_diar_and_tx(source_dir: Path, tmp_path: Path) -> None:
    """A file that fails separation should not reach diarization or transcription."""
    ok = MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0))
    with (
        patch("src.pipeline.separate", side_effect=[Exception("sep boom"), ok, ok]),
        patch("src.pipeline.load_pipeline", return_value=MagicMock()),
        patch("src.pipeline._resolve_hf_token", return_value="tok"),
        patch(
            "src.pipeline.diarize",
            return_value=MagicMock(
                processing_time_seconds=2.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ) as mock_diar,
        patch("src.pipeline._load_whisperx_model", return_value=MagicMock()),
        patch(
            "src.pipeline.transcribe",
            return_value=MagicMock(
                processing_time_seconds=5.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ) as mock_tx,
    ):
        result = run_pipeline(
            source_dir=source_dir,
            demucs_output_dir=tmp_path / "demucs",
            diarization_dir=tmp_path / "diar",
            transcription_dir=tmp_path / "tx",
        )

    assert result.separation.n_failed == 1
    assert result.separation.n_succeeded == 2
    assert mock_diar.call_count == 2  # only for the two successful files
    assert mock_tx.call_count == 2


def test_run_pipeline_diarization_failure_skips_tx(source_dir: Path, tmp_path: Path) -> None:
    """A file that fails diarization should not be transcribed."""
    sep_ok = MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0))
    diar_ok = MagicMock(
        processing_time_seconds=2.0,
        model_dump_json=lambda **kw: "{}",
        input_info=MagicMock(duration_seconds=10.0),
        segments=[],
    )
    with (
        patch("src.pipeline.separate", return_value=sep_ok),
        patch("src.pipeline.load_pipeline", return_value=MagicMock()),
        patch("src.pipeline._resolve_hf_token", return_value="tok"),
        patch("src.pipeline.diarize", side_effect=[Exception("diar boom"), diar_ok, diar_ok]),
        patch("src.pipeline._load_whisperx_model", return_value=MagicMock()),
        patch(
            "src.pipeline.transcribe",
            return_value=MagicMock(
                processing_time_seconds=5.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ) as mock_tx,
    ):
        result = run_pipeline(
            source_dir=source_dir,
            demucs_output_dir=tmp_path / "demucs",
            diarization_dir=tmp_path / "diar",
            transcription_dir=tmp_path / "tx",
        )

    assert result.diarization.n_failed == 1
    assert mock_tx.call_count == 2


def test_run_pipeline_resume_skips_completed_files(source_dir: Path, tmp_path: Path) -> None:
    """Files whose transcription JSON already exists should be fully skipped."""
    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()
    files = discover_files(source_dir)
    # Pre-complete one file
    cid = files[0][0]
    (tx_dir / f"transcription_{cid}.json").write_text('{"ok":true}')

    with (
        patch(
            "src.pipeline.separate",
            return_value=MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0)),
        ) as mock_sep,
        patch("src.pipeline.load_pipeline", return_value=MagicMock()),
        patch("src.pipeline._resolve_hf_token", return_value="tok"),
        patch(
            "src.pipeline.diarize",
            return_value=MagicMock(
                processing_time_seconds=2.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ),
        patch("src.pipeline._load_whisperx_model", return_value=MagicMock()),
        patch(
            "src.pipeline.transcribe",
            return_value=MagicMock(
                processing_time_seconds=5.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ),
    ):
        result = run_pipeline(
            source_dir=source_dir,
            demucs_output_dir=tmp_path / "demucs",
            diarization_dir=tmp_path / "diar",
            transcription_dir=tx_dir,
            resume=True,
        )

    assert mock_sep.call_count == 2  # one file was fully skipped
    sep_skipped = [o for o in result.separation.outcomes if o.skipped]
    assert len(sep_skipped) == 1
    assert sep_skipped[0].content_id == cid


def test_run_pipeline_all_skipped_skips_model_load(source_dir: Path, tmp_path: Path) -> None:
    """If all files are already done, no models should be loaded at all."""
    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()
    for cid, _ in discover_files(source_dir):
        (tx_dir / f"transcription_{cid}.json").write_text('{"ok":true}')

    with (
        patch("src.pipeline.load_pipeline") as mock_diar_load,
        patch("src.pipeline._load_whisperx_model") as mock_wx_load,
    ):
        result = run_pipeline(
            source_dir=source_dir,
            demucs_output_dir=tmp_path / "demucs",
            diarization_dir=tmp_path / "diar",
            transcription_dir=tx_dir,
            resume=True,
        )

    mock_diar_load.assert_not_called()
    mock_wx_load.assert_not_called()
    assert all(o.skipped for o in result.separation.outcomes)
    assert all(o.skipped for o in result.diarization.outcomes)
    assert all(o.skipped for o in result.transcription.outcomes)


def test_run_pipeline_resume_false_reprocesses_all(source_dir: Path, tmp_path: Path) -> None:
    """resume=False should process all files even if output already exists."""
    tx_dir = tmp_path / "tx"
    tx_dir.mkdir()
    for cid, _ in discover_files(source_dir):
        (tx_dir / f"transcription_{cid}.json").write_text('{"ok":true}')

    with (
        patch(
            "src.pipeline.separate",
            return_value=MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0)),
        ) as mock_sep,
        patch("src.pipeline.load_pipeline", return_value=MagicMock()),
        patch("src.pipeline._resolve_hf_token", return_value="tok"),
        patch(
            "src.pipeline.diarize",
            return_value=MagicMock(
                processing_time_seconds=2.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ),
        patch("src.pipeline._load_whisperx_model", return_value=MagicMock()),
        patch(
            "src.pipeline.transcribe",
            return_value=MagicMock(
                processing_time_seconds=5.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ),
    ):
        run_pipeline(
            source_dir=source_dir,
            demucs_output_dir=tmp_path / "demucs",
            diarization_dir=tmp_path / "diar",
            transcription_dir=tx_dir,
            resume=False,
        )

    assert mock_sep.call_count == 3


def test_run_pipeline_no_vocals_deleted_when_events_disabled(source_dir: Path, tmp_path: Path) -> None:
    """no_vocals.wav should be deleted after separation when --events is not set."""
    demucs_dir = tmp_path / "demucs"
    cid = discover_files(source_dir)[0][0]
    no_vocals = _no_vocals_path(cid, demucs_dir)

    def _sep_side_effect(**kwargs):
        # Derive the content_id from the actual input file so each call creates
        # stems for the correct file, not always for the first cid.
        current_cid = kwargs["input_file"].stem[len("audio_") :]
        _make_wav(_vocals_path(current_cid, demucs_dir))
        _make_wav(_no_vocals_path(current_cid, demucs_dir))
        return MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0))

    with (
        patch("src.pipeline.separate", side_effect=_sep_side_effect),
        patch("src.pipeline.load_pipeline", return_value=MagicMock()),
        patch("src.pipeline._resolve_hf_token", return_value="tok"),
        patch(
            "src.pipeline.diarize",
            return_value=MagicMock(
                processing_time_seconds=2.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ),
        patch("src.pipeline._load_whisperx_model", return_value=MagicMock()),
        patch(
            "src.pipeline.transcribe",
            return_value=MagicMock(
                processing_time_seconds=5.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ),
    ):
        run_pipeline(
            source_dir=source_dir,
            demucs_output_dir=demucs_dir,
            diarization_dir=tmp_path / "diar",
            transcription_dir=tmp_path / "tx",
            enable_events=False,
            keep_scratch=False,
        )

    assert not no_vocals.exists()


def test_run_pipeline_no_vocals_kept_when_events_enabled(source_dir: Path, tmp_path: Path) -> None:
    """no_vocals.wav should be retained when --events is set (step 5 will consume it)."""
    demucs_dir = tmp_path / "demucs"
    cid = discover_files(source_dir)[0][0]
    no_vocals = _no_vocals_path(cid, demucs_dir)

    def _sep_side_effect(**kwargs):
        current_cid = kwargs["input_file"].stem[len("audio_") :]
        _make_wav(_vocals_path(current_cid, demucs_dir))
        _make_wav(_no_vocals_path(current_cid, demucs_dir))
        return MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0))

    with (
        patch("src.pipeline.separate", side_effect=_sep_side_effect),
        patch("src.pipeline.load_pipeline", return_value=MagicMock()),
        patch("src.pipeline._resolve_hf_token", return_value="tok"),
        patch(
            "src.pipeline.diarize",
            return_value=MagicMock(
                processing_time_seconds=2.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ),
        patch("src.pipeline._load_whisperx_model", return_value=MagicMock()),
        patch(
            "src.pipeline.transcribe",
            return_value=MagicMock(
                processing_time_seconds=5.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ),
    ):
        run_pipeline(
            source_dir=source_dir,
            demucs_output_dir=demucs_dir,
            diarization_dir=tmp_path / "diar",
            transcription_dir=tmp_path / "tx",
            enable_events=True,
            keep_scratch=False,
        )

    assert no_vocals.exists()


def test_run_pipeline_vocals_deleted_after_transcription(source_dir: Path, tmp_path: Path) -> None:
    """vocals.wav should be cleaned up after successful transcription."""
    demucs_dir = tmp_path / "demucs"
    cid = discover_files(source_dir)[0][0]
    vocals = _vocals_path(cid, demucs_dir)

    def _sep_side_effect(**kwargs):
        current_cid = kwargs["input_file"].stem[len("audio_") :]
        _make_wav(_vocals_path(current_cid, demucs_dir))
        return MagicMock(processing_time_seconds=1.0, input_info=MagicMock(duration_seconds=10.0))

    with (
        patch("src.pipeline.separate", side_effect=_sep_side_effect),
        patch("src.pipeline.load_pipeline", return_value=MagicMock()),
        patch("src.pipeline._resolve_hf_token", return_value="tok"),
        patch(
            "src.pipeline.diarize",
            return_value=MagicMock(
                processing_time_seconds=2.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ),
        patch("src.pipeline._load_whisperx_model", return_value=MagicMock()),
        patch(
            "src.pipeline.transcribe",
            return_value=MagicMock(
                processing_time_seconds=5.0,
                model_dump_json=lambda **kw: "{}",
                input_info=MagicMock(duration_seconds=10.0),
                segments=[],
            ),
        ),
    ):
        run_pipeline(
            source_dir=source_dir,
            demucs_output_dir=demucs_dir,
            diarization_dir=tmp_path / "diar",
            transcription_dir=tmp_path / "tx",
            keep_scratch=False,
        )

    assert not vocals.exists()
