"""Bearer-token authentication for the service HTTP API.

Service auth is "is this caller allowed to use Refinery at all" — a single tier
of API keys validated against a comma-separated allowlist loaded from
``REFINERY_API_KEYS``. It has nothing to do with which S3 objects the caller
can read or write; that's the caller's presigned-URL story (see
``src/service/uri_io.py``).

Tokens are never logged in plaintext. The :func:`fingerprint` helper returns
the first 8 hex characters of the SHA-256 of the token — useful for audit log
correlation without exposing the token itself. The bearer dependency returns
the fingerprint on success so route handlers can include it in structured logs.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_security_scheme = HTTPBearer(auto_error=False)


class AllowlistError(ValueError):
    """Raised when ``REFINERY_API_KEYS`` is unset or only contains empty tokens."""


def fingerprint(token: str) -> str:
    """Return the first 8 hex chars of SHA-256(token).

    Stable for the same input; one-way (cannot reverse to recover the token).
    Used in audit logs so failed-auth events stay correlatable across requests
    without leaking the actual bearer.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]


def load_allowlist_from_env(env_var: str = "REFINERY_API_KEYS") -> set[str]:
    """Read the comma-separated allowlist from the environment.

    Empty entries (e.g. from a trailing comma) are dropped. Raises
    :class:`AllowlistError` if the resulting set is empty so the container
    fails fast at startup rather than serving requests no caller can satisfy.
    """
    raw = os.getenv(env_var, "")
    tokens = {t.strip() for t in raw.split(",") if t.strip()}
    if not tokens:
        raise AllowlistError(
            f"{env_var} is empty or unset; the service requires at least one bearer token in the allowlist"
        )
    return tokens


def make_bearer_dependency(allowlist: set[str]) -> Callable[..., str]:
    """Build a FastAPI dependency that validates the bearer token against ``allowlist``.

    The returned callable is suitable for use with ``Depends(...)``. On
    success it returns the token's :func:`fingerprint`. On any failure
    (missing header, wrong scheme, token not in the allowlist) it raises
    ``HTTPException(401)`` with a generic ``invalid_bearer`` detail — the
    same response shape regardless of which check failed, so callers can't
    distinguish "not in allowlist" from "header malformed."
    """
    # Capture a frozen copy so later mutations to the caller's set don't change auth behavior.
    frozen = frozenset(allowlist)

    def require_bearer(
        creds: HTTPAuthorizationCredentials | None = Depends(_security_scheme),
    ) -> str:
        if creds is None or creds.scheme.lower() != "bearer" or not creds.credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "invalid_bearer"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        if creds.credentials not in frozen:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "invalid_bearer"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return fingerprint(creds.credentials)

    return require_bearer


__all__ = [
    "AllowlistError",
    "fingerprint",
    "load_allowlist_from_env",
    "make_bearer_dependency",
]
