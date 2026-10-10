"""Tests for the optional APP_PASSWORD login (app.auth)."""

import pytest

from tests.conftest import set_password


@pytest.fixture
def locked(client, monkeypatch):
    set_password(monkeypatch, "s3cret")
    monkeypatch.setattr("app.auth.time.sleep", lambda _: None)
    return client


def test_no_password_means_open(client):
    assert client.get("/api/session").json() == {"auth_required": False, "authenticated": True}
    assert client.get("/api/status").status_code == 200


def test_api_and_docs_locked_without_login(locked):
    assert locked.get("/api/session").json() == {"auth_required": True, "authenticated": False}
    for path in ["/api/status", "/api/documents", "/api/documents/a.pdf/file", "/docs", "/openapi.json"]:
        assert locked.get(path).status_code == 401, path
    assert locked.post("/api/ask", json={"question": "hi"}).status_code == 401


def test_frontend_and_health_stay_public(locked):
    assert locked.get("/").status_code == 200
    assert locked.get("/manifest.webmanifest").status_code == 200
    assert locked.get("/health").status_code == 200


def test_wrong_password_rejected(locked):
    r = locked.post("/api/login", json={"password": "nope"})
    assert r.status_code == 401
    assert locked.get("/api/status").status_code == 401


def test_login_then_logout(locked):
    r = locked.post("/api/login", json={"password": "s3cret"})
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert locked.get("/api/status").status_code == 200
    assert locked.get("/api/session").json()["authenticated"] is True

    locked.post("/api/logout")
    assert locked.get("/api/status").status_code == 401


def test_password_change_invalidates_sessions(locked, monkeypatch):
    locked.post("/api/login", json={"password": "s3cret"})
    set_password(monkeypatch, "new-password")
    assert locked.get("/api/status").status_code == 401
