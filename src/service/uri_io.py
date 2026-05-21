"""URI fetch/upload across `https://` (presigned) and `file://` schemes.

Service-mode I/O helpers. Used by the worker to download per-job inputs and
upload per-job transcripts plus the per-batch summary. The same `upload()`
function serves both — caller decides what to PUT and where.

Bare `s3://` URIs are out of scope for v1; the production path is
caller-signed HTTPS PUT/GET URLs against any S3-compatible endpoint
(real S3, MinIO, R2, B2, GCS-with-S3-compat).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx


class UnsupportedScheme(ValueError):
    """Raised when a URI uses a scheme other than `https://` or `file://`."""


class FetchError(RuntimeError):
    """Raised when fetching an input URI fails (404, network error, missing file)."""


class UploadError(RuntimeError):
    """Raised when uploading to an output or summary URI fails."""


_SUPPORTED_SCHEMES = ("https", "file")


def _scheme_of(uri: str) -> str:
    return urlparse(uri).scheme.lower()


def _file_path_from_uri(uri: str) -> Path:
    """Decode a `file://` URI to a local absolute Path."""
    parsed = urlparse(uri)
    return Path(unquote(parsed.path))


def fetch_input(uri: str, dest: Path, *, timeout: float = 60.0) -> Path:
    """Fetch the audio input at `uri` to local disk and return the readable path.

    - `https://` URIs: stream the HTTP body into `dest`. Returns `dest`.
    - `file://` URIs: verify the file exists locally and return its path as-is
      (no copy — the worker reads it directly from the bind-mounted location).

    Args:
        uri: Input URI, either `https://...` or `file:///abs/path`.
        dest: Local destination path for downloads. Parent directory must exist.
              Ignored for `file://` URIs.
        timeout: Per-request timeout in seconds for HTTPS fetches.

    Returns:
        Local Path the caller can hand to the pipeline.

    Raises:
        UnsupportedScheme: scheme is not `https` or `file`.
        FetchError: HTTP status >= 400, network error, or `file://` path missing.
    """
    scheme = _scheme_of(uri)

    if scheme == "https":
        try:
            with httpx.stream("GET", uri, timeout=timeout, follow_redirects=True) as resp:
                resp.raise_for_status()
                with dest.open("wb") as fh:
                    for chunk in resp.iter_bytes():
                        fh.write(chunk)
        except httpx.HTTPStatusError as e:
            raise FetchError(f"HTTPS GET {uri!r} returned {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise FetchError(f"HTTPS GET {uri!r} failed: {e!r}") from e
        return dest

    if scheme == "file":
        local = _file_path_from_uri(uri)
        if not local.is_file():
            raise FetchError(f"file:// path does not exist or is not a regular file: {local}")
        return local

    raise UnsupportedScheme(f"Unsupported URI scheme {scheme!r} in {uri!r}; expected one of {_SUPPORTED_SCHEMES}")


def upload(uri: str, payload: dict[str, Any], *, content_type: str = "application/json", timeout: float = 60.0) -> None:
    """Upload a JSON payload to `uri`.

    Used for both per-job transcripts and per-batch summaries.

    - `https://` URIs: HTTP PUT with the JSON-encoded body and the given
      Content-Type header.
    - `file://` URIs: write the JSON-encoded bytes to the local path,
      creating parent directories as needed.

    Args:
        uri: Destination URI, either `https://...` or `file:///abs/path`.
        payload: JSON-serializable dict to upload.
        content_type: MIME type for HTTPS uploads. Defaults to `application/json`.
        timeout: Per-request timeout in seconds for HTTPS uploads.

    Raises:
        UnsupportedScheme: scheme is not `https` or `file`.
        UploadError: HTTP status >= 400, network error, or local write failure.
    """
    scheme = _scheme_of(uri)
    body = json.dumps(payload).encode("utf-8")

    if scheme == "https":
        try:
            resp = httpx.put(uri, content=body, headers={"Content-Type": content_type}, timeout=timeout)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise UploadError(f"HTTPS PUT {uri!r} returned {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise UploadError(f"HTTPS PUT {uri!r} failed: {e!r}") from e
        return

    if scheme == "file":
        local = _file_path_from_uri(uri)
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(body)
        except OSError as e:
            raise UploadError(f"file:// write to {local} failed: {e!r}") from e
        return

    raise UnsupportedScheme(f"Unsupported URI scheme {scheme!r} in {uri!r}; expected one of {_SUPPORTED_SCHEMES}")


def validate_scheme(uri: str) -> None:
    """Raise `UnsupportedScheme` if `uri` does not use `https://` or `file://`.

    Cheap pre-check used by the HTTP layer to reject bad requests with 400
    before they hit the queue.
    """
    scheme = _scheme_of(uri)
    if scheme not in _SUPPORTED_SCHEMES:
        raise UnsupportedScheme(f"Unsupported URI scheme {scheme!r} in {uri!r}; expected one of {_SUPPORTED_SCHEMES}")


__all__ = [
    "FetchError",
    "UnsupportedScheme",
    "UploadError",
    "fetch_input",
    "upload",
    "validate_scheme",
]
