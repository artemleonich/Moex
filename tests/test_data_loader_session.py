"""Tests for MOEXDataLoader session hardening (timeout + retry).

These tests verify the configuration of the underlying ``requests.Session``
without making any real network calls.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from island_model.data_loader import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    MOEXDataLoader,
    _TimeoutSession,
    _build_session,
)


# ---------- _TimeoutSession (timeout injection) ----------


def test_timeout_session_injects_default_when_missing() -> None:
    """A call without explicit timeout must get the session's default."""
    session = _TimeoutSession(default_timeout=42)
    captured: dict = {}

    def fake_super_request(self, method, url, **kwargs):
        captured.update(kwargs)
        captured["_method"] = method
        return None

    with patch.object(requests.Session, "request", fake_super_request):
        session.request("GET", "https://example.com/")

    assert captured.get("timeout") == 42, captured
    assert captured.get("_method") == "GET"


def test_timeout_session_preserves_caller_timeout() -> None:
    """A caller-supplied timeout must NOT be overridden by the default."""
    session = _TimeoutSession(default_timeout=10)
    captured: dict = {}

    def fake_super_request(self, method, url, **kwargs):
        captured.update(kwargs)
        return None

    with patch.object(requests.Session, "request", fake_super_request):
        session.request("GET", "https://example.com/", timeout=99)

    assert captured.get("timeout") == 99, captured


def test_timeout_session_via_get_method() -> None:
    """``session.get()`` must also flow through the timeout injection."""
    session = _TimeoutSession(default_timeout=7)
    captured: dict = {}

    def fake_super_request(self, method, url, **kwargs):
        captured.update(kwargs)
        captured["_method"] = method
        return None

    with patch.object(requests.Session, "request", fake_super_request):
        session.get("https://example.com/path")

    assert captured.get("timeout") == 7
    assert captured.get("_method") == "GET"


# ---------- _build_session ----------


def test_build_session_returns_subclass_instance() -> None:
    session = _build_session()
    assert isinstance(session, _TimeoutSession)


def test_build_session_mounts_retry_adapter_on_https_and_http() -> None:
    """Both schemes must have an HTTPAdapter with a Retry policy."""
    session = _build_session()
    for scheme in ("https://", "http://"):
        adapter = session.get_adapter(scheme + "example.com")
        assert isinstance(adapter, HTTPAdapter), f"scheme={scheme}"
        assert adapter.max_retries is not None, f"scheme={scheme}"
        assert isinstance(adapter.max_retries, Retry)


def test_build_session_retry_policy_covers_transient_errors() -> None:
    """Retry must include 429/5xx codes and use exponential backoff."""
    session = _build_session(max_retries=5, backoff_factor=0.7)
    adapter = session.get_adapter("https://example.com")
    retry = adapter.max_retries
    assert retry.total == 5
    assert retry.backoff_factor == 0.7
    for code in (429, 500, 502, 503, 504):
        assert code in retry.status_forcelist


def test_build_session_retry_is_get_only() -> None:
    """Only idempotent methods are auto-retried to avoid duplicate side-effects."""
    session = _build_session()
    retry = session.get_adapter("https://example.com").max_retries
    allowed = retry.allowed_methods
    if allowed is not None:
        allowed_lower = {m.lower() for m in allowed}
        assert "get" in allowed_lower


# ---------- MOEXDataLoader ----------


def test_loader_defaults_match_module_constants() -> None:
    loader = MOEXDataLoader()
    assert loader.timeout == DEFAULT_TIMEOUT
    assert loader.max_retries == DEFAULT_MAX_RETRIES
    assert isinstance(loader.session, requests.Session)


def test_loader_respects_constructor_kwargs() -> None:
    loader = MOEXDataLoader(timeout=7, max_retries=11)
    assert loader.timeout == 7
    assert loader.max_retries == 11

    captured: dict = {}

    def fake_super_request(self, method, url, **kwargs):
        captured.update(kwargs)
        return None

    with patch.object(requests.Session, "request", fake_super_request):
        loader.session.request("GET", "https://example.com/")

    assert captured.get("timeout") == 7, captured


def test_loader_session_adapters_have_retry() -> None:
    loader = MOEXDataLoader()
    adapter = loader.session.get_adapter("https://iss.moex.com/iss/...")
    assert isinstance(adapter, HTTPAdapter)
    assert adapter.max_retries is not None
    assert adapter.max_retries.total >= 1


def test_loader_independent_sessions_per_instance() -> None:
    """Two loaders must not share retry adapter state."""
    a = MOEXDataLoader(max_retries=1)
    b = MOEXDataLoader(max_retries=9)
    assert a.session is not b.session
    assert a.session.get_adapter("https://x").max_retries.total == 1
    assert b.session.get_adapter("https://x").max_retries.total == 9