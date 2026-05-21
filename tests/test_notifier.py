"""Tests for `src.notifier`.

The `_no_slack` autouse fixture in `tests/conftest.py` removes
`SLACK_WEBHOOK_URL` from the environment and patches out the
`_load_dotenv` call, so these tests verify the function shapes without
firing real webhooks.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src import notifier


@pytest.fixture
def captured_messages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace `_send` with a capture list so we can assert on payloads."""
    captured: list[str] = []
    monkeypatch.setattr(notifier, "_send", captured.append)
    return captured


def test_notify_job_failed_emits_job_id_stage_and_error(captured_messages: list[str]) -> None:
    notifier.notify_job_failed(
        job_id="rfj_abc123",
        stage="transcribe",
        input_uri="https://bucket.example.com/audio/x.wav",
        error="ValueError: bad sample rate",
    )
    assert len(captured_messages) == 1
    msg = captured_messages[0]
    assert "rfj_abc123" in msg
    assert "transcribe" in msg
    assert "ValueError: bad sample rate" in msg
    assert "https://bucket.example.com/audio/x.wav" in msg


def test_notify_job_failed_strips_presigned_query_string(captured_messages: list[str]) -> None:
    """Presigned URLs include long X-Amz-* query strings. The Slack message
    should show just the object path for legibility."""
    notifier.notify_job_failed(
        job_id="rfj_xyz",
        stage="download",
        input_uri="https://bucket.example.com/audio/x.wav?X-Amz-Signature=abc&X-Amz-Date=20260519",
        error="404 Not Found",
    )
    msg = captured_messages[0]
    # Path retained
    assert "https://bucket.example.com/audio/x.wav" in msg
    # Query stripped
    assert "X-Amz-Signature" not in msg
    assert "?" not in msg.split("input:", 1)[1].split("\n", 1)[0]


def test_notify_job_failed_silent_when_webhook_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity: with the autouse `_no_slack` fixture wiping SLACK_WEBHOOK_URL,
    `_send` is still callable but exits early. We re-import _send (not
    captured) and verify no HTTP call happens."""
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(notifier, "_load_dotenv", None)
    with patch("src.notifier.urllib.request.urlopen") as urlopen:
        notifier.notify_job_failed("rfj_x", "transcribe", "file:///x.wav", "boom")
    urlopen.assert_not_called()
