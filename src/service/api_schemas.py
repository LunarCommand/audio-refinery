"""HTTP transport schemas — request and response Pydantic models for the FastAPI app.

These differ from :mod:`src.service.schemas` in scope:

- ``schemas`` holds **content** schemas — what we serialize to disk: the
  combined transcript (uploaded per job to ``output_uri``) and the batch
  summary (uploaded per batch to ``summary_uri``). Their evolution is gated
  by what the pipeline produces.

- ``api_schemas`` (this module) holds **transport** schemas — what flows
  over the wire between the caller and the service. Their evolution is
  gated by the integration contract in ``_docs/refinery-integration.md``.

Keeping the two layers separate keeps diffs honest: a transcript-schema
bump (e.g., when v0.3.0 alignment splits aligned words into a separate
array) doesn't churn the HTTP wire format, and vice versa.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from src.service.uri_io import UnsupportedScheme, validate_scheme


def _validate_uri(value: str) -> str:
    """Pydantic-friendly wrapper around :func:`validate_scheme`.

    Translates :class:`UnsupportedScheme` into ``ValueError`` so pydantic
    surfaces it as a normal field-validation error (HTTP 422 by default).
    """
    try:
        validate_scheme(value)
    except UnsupportedScheme as exc:
        raise ValueError(str(exc)) from exc
    return value


# ---------------------------------------------------------------------------
# POST /transcribe — request
# ---------------------------------------------------------------------------


class JobRequest(BaseModel):
    """One job inside a ``POST /transcribe`` request."""

    input_uri: str = Field(min_length=1)
    output_uri: str = Field(min_length=1)

    @field_validator("input_uri", "output_uri")
    @classmethod
    def _check_scheme(cls, v: str) -> str:
        return _validate_uri(v)


class TranscribeRequest(BaseModel):
    """Body of ``POST /transcribe``. One ``summary_uri`` per batch."""

    summary_uri: str = Field(min_length=1)
    jobs: list[JobRequest] = Field(min_length=1)

    @field_validator("summary_uri")
    @classmethod
    def _check_summary_scheme(cls, v: str) -> str:
        return _validate_uri(v)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class TranscribeResponse(BaseModel):
    batch_id: str
    job_ids: list[str]


class JobStatusResponse(BaseModel):
    job_id: str
    batch_id: str
    status: str
    input_uri: str
    output_uri: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    stage: str | None = None
    error: str | None = None
    retryable: bool | None = None
    duration_seconds: float | None = None


class HealthResponse(BaseModel):
    status: str  # "ok" | "loading" | "failed"
    stage: str | None = None
    detail: str | None = None


__all__ = [
    "HealthResponse",
    "JobRequest",
    "JobStatusResponse",
    "TranscribeRequest",
    "TranscribeResponse",
]
