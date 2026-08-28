"""Auth helper tests (no live mail or HTTP server)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "dashboard"))

from backend.accounts import (  # noqa: E402
    allowed_email,
    normalize_email,
    valid_email,
    valid_password,
)


def test_normalize_and_validate_email() -> None:
    assert normalize_email("  A@B.Com ") == "a@b.com"
    assert valid_email("user@example.com")
    assert not valid_email("not-an-email")


def test_password_rules() -> None:
    assert valid_password("short", "a@b.com") is not None
    assert valid_password("user@example.com", "user@example.com") is not None
    assert valid_password("  paddedpass12", "a@b.com") is not None
    assert valid_password("a-reasonable-password", "a@b.com") is None


def test_email_domain_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KITAB_ALLOWED_EMAIL_DOMAINS", "")
    assert allowed_email("anyone@gmail.com")
    monkeypatch.setenv("KITAB_ALLOWED_EMAIL_DOMAINS", "kitab-atlas.com, lab.org")
    assert allowed_email("me@lab.org")
    assert not allowed_email("me@gmail.com")
