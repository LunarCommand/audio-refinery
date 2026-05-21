"""Tests for `src.service.uri_io`.

Covers the four code paths times the two URI schemes:
  fetch_input × {https, file}
  upload × {https, file}

Plus scheme validation, error mapping, and edge cases (missing files,
nonexistent parent directories, HTTP failure codes).

HTTPS calls are exercised through a real ``httpx.MockTransport`` so the
streaming and PUT paths execute end-to-end against an in-memory server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.service import uri_io
from src.service.uri_io import FetchError, UnsupportedScheme, UploadError

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _install_mock_transport(monkeypatch: pytest.MonkeyPatch, handler):
    """Patch ``httpx.stream`` and ``httpx.put`` to use a MockTransport with `handler`."""
    transport = httpx.MockTransport(handler)

    def _stream(method, url, **kwargs):
        client = httpx.Client(transport=transport)
        return client.stream(method, url, **kwargs)

    def _put(url, **kwargs):
        with httpx.Client(transport=transport) as client:
            return client.put(url, **kwargs)

    monkeypatch.setattr("src.service.uri_io.httpx.stream", _stream)
    monkeypatch.setattr("src.service.uri_io.httpx.put", _put)


# --------------------------------------------------------------------------
# validate_scheme
# --------------------------------------------------------------------------


def test_validate_scheme_accepts_https():
    uri_io.validate_scheme("https://example.com/a.json?X-Amz-...")


def test_validate_scheme_accepts_file():
    uri_io.validate_scheme("file:///inbox/audio.wav")


@pytest.mark.parametrize("uri", ["ftp://x/y", "s3://bucket/key", "http://x/y", "data:text/plain,xx"])
def test_validate_scheme_rejects_unsupported(uri: str):
    with pytest.raises(UnsupportedScheme):
        uri_io.validate_scheme(uri)


# --------------------------------------------------------------------------
# fetch_input — https://
# --------------------------------------------------------------------------


def test_fetch_input_https_streams_to_dest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    expected_body = b"fake-wav-bytes-" * 1000  # 15 KB; exercises chunked streaming

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://example.com/audio.wav"
        return httpx.Response(200, content=expected_body)

    _install_mock_transport(monkeypatch, handler)

    dest = tmp_path / "downloaded.wav"
    result = uri_io.fetch_input("https://example.com/audio.wav", dest)

    assert result == dest
    assert dest.read_bytes() == expected_body


def test_fetch_input_https_raises_on_4xx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    _install_mock_transport(monkeypatch, handler)

    with pytest.raises(FetchError, match="404"):
        uri_io.fetch_input("https://example.com/missing.wav", tmp_path / "x.wav")


def test_fetch_input_https_raises_on_network_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated DNS failure")

    _install_mock_transport(monkeypatch, handler)

    with pytest.raises(FetchError, match="failed"):
        uri_io.fetch_input("https://nowhere.invalid/x.wav", tmp_path / "x.wav")


# --------------------------------------------------------------------------
# fetch_input — file://
# --------------------------------------------------------------------------


def test_fetch_input_file_returns_original_path_no_copy(tmp_path: Path):
    source = tmp_path / "inbox" / "audio.wav"
    source.parent.mkdir()
    source.write_bytes(b"real-audio-data")

    dest = tmp_path / "should-not-be-written.wav"
    result = uri_io.fetch_input(f"file://{source}", dest)

    assert result == source
    # dest must not have been touched — file:// path is read in-place
    assert not dest.exists()


def test_fetch_input_file_raises_when_missing(tmp_path: Path):
    missing = tmp_path / "nope.wav"
    with pytest.raises(FetchError, match="does not exist"):
        uri_io.fetch_input(f"file://{missing}", tmp_path / "ignored.wav")


def test_fetch_input_file_raises_when_directory(tmp_path: Path):
    a_dir = tmp_path / "a_directory"
    a_dir.mkdir()
    with pytest.raises(FetchError, match="not a regular file"):
        uri_io.fetch_input(f"file://{a_dir}", tmp_path / "ignored.wav")


# --------------------------------------------------------------------------
# fetch_input — unsupported scheme
# --------------------------------------------------------------------------


def test_fetch_input_rejects_unsupported_scheme(tmp_path: Path):
    with pytest.raises(UnsupportedScheme, match="ftp"):
        uri_io.fetch_input("ftp://example.com/x.wav", tmp_path / "x.wav")


# --------------------------------------------------------------------------
# upload — https://
# --------------------------------------------------------------------------


def test_upload_https_puts_json_with_content_type(monkeypatch: pytest.MonkeyPatch):
    received: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["method"] = request.method
        received["url"] = str(request.url)
        received["content_type"] = request.headers.get("Content-Type")
        received["body"] = request.read()
        return httpx.Response(200)

    _install_mock_transport(monkeypatch, handler)

    payload = {"hello": "world", "n": 42}
    uri_io.upload("https://example.com/transcript.json", payload)

    assert received["method"] == "PUT"
    assert received["url"] == "https://example.com/transcript.json"
    assert received["content_type"] == "application/json"
    assert json.loads(received["body"]) == payload


def test_upload_https_custom_content_type(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers.get("Content-Type", "")
        return httpx.Response(200)

    _install_mock_transport(monkeypatch, handler)

    uri_io.upload("https://example.com/x.json", {"a": 1}, content_type="application/vnd.x+json")
    assert captured["content_type"] == "application/vnd.x+json"


def test_upload_https_raises_on_4xx(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    _install_mock_transport(monkeypatch, handler)

    with pytest.raises(UploadError, match="403"):
        uri_io.upload("https://example.com/x.json", {"a": 1})


def test_upload_https_raises_on_network_error(monkeypatch: pytest.MonkeyPatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated TCP reset")

    _install_mock_transport(monkeypatch, handler)

    with pytest.raises(UploadError, match="failed"):
        uri_io.upload("https://nowhere.invalid/x.json", {"a": 1})


# --------------------------------------------------------------------------
# upload — file://
# --------------------------------------------------------------------------


def test_upload_file_writes_json_to_local_path(tmp_path: Path):
    dest = tmp_path / "out" / "transcript.json"
    payload = {"schema_version": "1.0.0", "segments": []}

    uri_io.upload(f"file://{dest}", payload)

    assert dest.is_file()
    assert json.loads(dest.read_bytes()) == payload


def test_upload_file_creates_missing_parent_dirs(tmp_path: Path):
    dest = tmp_path / "deeply" / "nested" / "summary.json"
    assert not dest.parent.exists()

    uri_io.upload(f"file://{dest}", {"ok": True})

    assert dest.is_file()
    assert dest.parent.is_dir()


def test_upload_file_overwrites_existing(tmp_path: Path):
    dest = tmp_path / "x.json"
    dest.write_text('{"old": true}')

    uri_io.upload(f"file://{dest}", {"new": True})
    assert json.loads(dest.read_bytes()) == {"new": True}


# --------------------------------------------------------------------------
# upload — unsupported scheme
# --------------------------------------------------------------------------


def test_upload_rejects_unsupported_scheme():
    with pytest.raises(UnsupportedScheme, match="s3"):
        uri_io.upload("s3://bucket/key.json", {"a": 1})
