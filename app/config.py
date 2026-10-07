"""Application configuration loaded from environment variables (.env)."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

# Project root = parent of the app/ package. All paths are resolved relative to it.
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

# Value shipped in .env.example; treat it as "no key configured".
_PLACEHOLDER_KEYS = {"", "your_key_here", "your_gemini_key_here"}

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash-lite"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


def _env(*names: str, default: str = "") -> str:
    """First non-empty value among names; placeholder keys count as empty."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value not in _PLACEHOLDER_KEYS:
            return value
    return default


@dataclass(frozen=True)
class LLMConfig:
    provider: str  # "gemini", "openai", "custom" or "none"
    api_key: str | None
    base_url: str | None
    model: str


def resolve_llm() -> LLMConfig:
    """Pick the answer-generating model from .env.

    Priority: GEMINI_API_KEY (free tier) > generic LLM_* settings > OPENAI_API_KEY.
    LLM_MODEL / LLM_BASE_URL, when set, override the provider defaults.
    """
    model_override = _env("LLM_MODEL")
    base_override = _env("LLM_BASE_URL")

    if gemini_key := _env("GEMINI_API_KEY"):
        return LLMConfig("gemini", gemini_key, base_override or GEMINI_BASE_URL,
                         _env("GEMINI_MODEL") or model_override or GEMINI_DEFAULT_MODEL)
    if (generic_key := _env("LLM_API_KEY")) or base_override:
        return LLMConfig("custom", generic_key or None, base_override or None,
                         model_override or OPENAI_DEFAULT_MODEL)
    if openai_key := _env("OPENAI_API_KEY"):
        return LLMConfig("openai", openai_key, None,
                         model_override or _env("OPENAI_MODEL") or OPENAI_DEFAULT_MODEL)
    return LLMConfig("none", None, None, "")


@dataclass(frozen=True)
class Settings:
    """Immutable settings object; read once at startup."""

    app_name: str = os.getenv("APP_NAME", "RAG Document Q&A")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    llm: LLMConfig = resolve_llm()
    documents_dir: Path = BASE_DIR / os.getenv("DOCUMENTS_DIR", "data/documents")
    chroma_dir: Path = BASE_DIR / os.getenv("CHROMA_DIR", "chroma_db")
    frontend_dir: Path = BASE_DIR / "frontend"
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))


settings = Settings()
