"""Tests for LLM provider selection and error messages."""

import httpx
import openai
import pytest

from app.config import GEMINI_BASE_URL, GEMINI_DEFAULT_MODEL, resolve_llm
from app.rag.llm import friendly_error

LLM_VARS = ["GEMINI_API_KEY", "GEMINI_MODEL", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in LLM_VARS:
        monkeypatch.delenv(name, raising=False)


def test_no_keys_means_passage_mode():
    assert resolve_llm().provider == "none"


def test_gemini_key_uses_gemini_endpoint_and_default_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    llm = resolve_llm()
    assert (llm.provider, llm.api_key, llm.base_url, llm.model) == (
        "gemini", "AIza-test", GEMINI_BASE_URL, GEMINI_DEFAULT_MODEL)


def test_gemini_model_can_be_changed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    assert resolve_llm().model == "gemini-3.8-flash"


def test_gemini_takes_priority_over_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    assert resolve_llm().provider == "gemini"


def test_placeholder_key_is_ignored(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_key_here")
    monkeypatch.setenv("OPENAI_API_KEY", "your_key_here")
    assert resolve_llm().provider == "none"


def test_generic_and_openai_settings_still_work(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "llama3.2:3b")
    llm = resolve_llm()
    assert (llm.provider, llm.api_key, llm.model) == ("custom", None, "llama3.2:3b")

    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.delenv("LLM_MODEL")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert resolve_llm().provider == "openai"


def _api_error(cls, status, message):
    request = httpx.Request("POST", "https://example.test")
    return cls(message, response=httpx.Response(status, request=request), body=None)


def test_friendly_errors():
    bad_key = _api_error(openai.BadRequestError, 400, "Please pass a valid API key")
    assert "rejected the API key" in friendly_error(bad_key, "gemini", "m")

    quota = _api_error(openai.RateLimitError, 429, "quota exceeded")
    assert "usage limit reached" in friendly_error(quota, "gemini", "m")

    missing = _api_error(openai.NotFoundError, 404, "not found")
    assert '"old-model" was not found' in friendly_error(missing, "gemini", "old-model")
