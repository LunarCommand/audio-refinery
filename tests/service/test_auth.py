"""Tests for `src.service.auth`.

The dependency callable is tested directly with synthesized
`HTTPAuthorizationCredentials` — no TestClient needed for the
allowlist/fingerprint behavior. A small in-app round-trip test through
FastAPI's TestClient covers the end-to-end shape (401 status,
WWW-Authenticate header, JSON body).
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from src.service.auth import (
    AllowlistError,
    fingerprint,
    load_allowlist_from_env,
    make_bearer_dependency,
)

# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_stable_across_calls():
    assert fingerprint("secret-key-1") == fingerprint("secret-key-1")


def test_fingerprint_differs_between_tokens():
    assert fingerprint("a") != fingerprint("b")


def test_fingerprint_does_not_leak_token():
    fp = fingerprint("the-real-secret")
    assert "the-real-secret" not in fp
    assert len(fp) == 8
    assert all(c in "0123456789abcdef" for c in fp)


# ---------------------------------------------------------------------------
# load_allowlist_from_env
# ---------------------------------------------------------------------------


def test_load_allowlist_parses_comma_separated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REFINERY_API_KEYS", "key1,key2,key3")
    assert load_allowlist_from_env() == {"key1", "key2", "key3"}


def test_load_allowlist_strips_whitespace(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REFINERY_API_KEYS", "  key1 , key2 ,  key3  ")
    assert load_allowlist_from_env() == {"key1", "key2", "key3"}


def test_load_allowlist_drops_empty_entries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REFINERY_API_KEYS", "key1,,key2,")
    assert load_allowlist_from_env() == {"key1", "key2"}


def test_load_allowlist_raises_when_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("REFINERY_API_KEYS", raising=False)
    with pytest.raises(AllowlistError, match="empty or unset"):
        load_allowlist_from_env()


def test_load_allowlist_raises_when_only_whitespace(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REFINERY_API_KEYS", " ,  , ")
    with pytest.raises(AllowlistError, match="empty or unset"):
        load_allowlist_from_env()


def test_load_allowlist_respects_alternative_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUSTOM_KEYS", "a,b")
    monkeypatch.delenv("REFINERY_API_KEYS", raising=False)
    assert load_allowlist_from_env("CUSTOM_KEYS") == {"a", "b"}


# ---------------------------------------------------------------------------
# make_bearer_dependency — direct invocation
# ---------------------------------------------------------------------------


def _creds(scheme: str, token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme=scheme, credentials=token)


def test_bearer_dependency_accepts_valid_token_returns_fingerprint():
    dep = make_bearer_dependency({"key1", "key2"})
    fp = dep(_creds("Bearer", "key1"))
    assert fp == fingerprint("key1")


def test_bearer_dependency_accepts_lowercase_bearer_scheme():
    dep = make_bearer_dependency({"key1"})
    fp = dep(_creds("bearer", "key1"))
    assert fp == fingerprint("key1")


def test_bearer_dependency_rejects_unknown_token():
    from fastapi import HTTPException

    dep = make_bearer_dependency({"key1"})
    with pytest.raises(HTTPException) as exc:
        dep(_creds("Bearer", "not-in-allowlist"))
    assert exc.value.status_code == 401
    assert exc.value.detail == {"error": "invalid_bearer"}


def test_bearer_dependency_rejects_missing_header():
    from fastapi import HTTPException

    dep = make_bearer_dependency({"key1"})
    with pytest.raises(HTTPException) as exc:
        dep(None)
    assert exc.value.status_code == 401


def test_bearer_dependency_rejects_non_bearer_scheme():
    from fastapi import HTTPException

    dep = make_bearer_dependency({"key1"})
    with pytest.raises(HTTPException) as exc:
        dep(_creds("Basic", "key1"))
    assert exc.value.status_code == 401


def test_bearer_dependency_rejects_empty_token():
    from fastapi import HTTPException

    dep = make_bearer_dependency({"key1"})
    with pytest.raises(HTTPException) as exc:
        dep(_creds("Bearer", ""))
    assert exc.value.status_code == 401


def test_bearer_dependency_frozen_against_post_construction_mutations():
    """Mutating the caller's set after building the dependency must not change auth behavior."""
    mutable: set[str] = {"key1"}
    dep = make_bearer_dependency(mutable)
    mutable.add("key2")  # would be a security regression if this took effect

    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        dep(_creds("Bearer", "key2"))
    # Original token still works.
    assert dep(_creds("Bearer", "key1")) == fingerprint("key1")


# ---------------------------------------------------------------------------
# End-to-end via FastAPI TestClient
# ---------------------------------------------------------------------------


def _make_app(allowlist: set[str]) -> FastAPI:
    app = FastAPI()
    require_bearer = make_bearer_dependency(allowlist)

    @app.get("/protected")
    def protected(fp: str = Depends(require_bearer)) -> dict[str, str]:
        return {"fingerprint": fp}

    @app.get("/open")
    def open_route() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_e2e_protected_route_with_valid_token():
    client = TestClient(_make_app({"good-key"}))
    resp = client.get("/protected", headers={"Authorization": "Bearer good-key"})
    assert resp.status_code == 200
    assert resp.json() == {"fingerprint": fingerprint("good-key")}


def test_e2e_protected_route_without_header_returns_401():
    client = TestClient(_make_app({"good-key"}))
    resp = client.get("/protected")
    assert resp.status_code == 401
    assert resp.json()["detail"] == {"error": "invalid_bearer"}
    assert resp.headers.get("www-authenticate", "").lower().startswith("bearer")


def test_e2e_protected_route_with_wrong_token_returns_401():
    client = TestClient(_make_app({"good-key"}))
    resp = client.get("/protected", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401


def test_e2e_open_route_remains_unauthenticated():
    """Sanity: the dependency is route-scoped, not global. Routes without
    Depends(require_bearer) stay open. This matches the /health design — no
    auth on the health endpoint."""
    client = TestClient(_make_app({"good-key"}))
    resp = client.get("/open")
    assert resp.status_code == 200
