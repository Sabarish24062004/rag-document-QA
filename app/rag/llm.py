"""Answer generation from retrieved chunks."""

import logging
from typing import Protocol

from app.rag.models import SearchResult

logger = logging.getLogger(__name__)

NOT_FOUND = "I couldn't find the answer to that in the uploaded documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered document excerpts provided.

Rules:
- Base every statement on the excerpts. Do not use outside knowledge.
- Cite the excerpts you used with their numbers in square brackets, e.g. [1] or [2][3], right after the statement they support.
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- Be clear and concise. Use short paragraphs or bullet points where helpful."""


class GenerationError(RuntimeError):
    """The LLM call failed (bad key, network, quota, ...)."""


class Generator(Protocol):
    llm_used: bool

    def generate(self, question: str, sources: list[SearchResult]) -> str: ...


def build_context(sources: list[SearchResult]) -> str:
    return "\n\n".join(
        f"[{i}] (Source: {s.chunk.filename}, page {s.chunk.page_number})\n{s.chunk.text}"
        for i, s in enumerate(sources, start=1)
    )


def friendly_error(exc: Exception, provider: str, model: str) -> str:
    """Turn an OpenAI-SDK exception into a message a user can act on."""
    import openai

    detail = str(exc)
    name = "Gemini" if provider == "gemini" else "The language model"
    if isinstance(exc, openai.AuthenticationError) or "valid API key" in detail or "API key not valid" in detail or "API_KEY_INVALID" in detail:
        return f"{name} rejected the API key. Check the key in your .env file, then restart the server."
    if isinstance(exc, openai.RateLimitError):
        return (f"{name} usage limit reached (free-tier quotas are per minute and per day). "
                "Wait a minute and try again; daily quotas reset at midnight Pacific time.")
    if isinstance(exc, openai.NotFoundError):
        return f'Model "{model}" was not found. Set a current model name in .env and restart the server.'
    if isinstance(exc, openai.PermissionDeniedError):
        return f"{name} denied access: {detail}"
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return f"Could not reach {name}. Check your internet connection and try again."
    return f"The language model request failed: {detail}"


class OpenAIGenerator:
    """Works with any OpenAI-compatible chat API (Gemini, OpenAI, Groq, Ollama, ...)."""

    llm_used = True

    def __init__(
        self, api_key: str | None, model: str, base_url: str | None = None, provider: str = "custom"
    ) -> None:
        from openai import OpenAI

        # Local servers such as Ollama ignore the key, but the client requires one.
        self._client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=120, max_retries=2)
        self.model = model
        self.provider = provider

    def generate(self, question: str, sources: list[SearchResult]) -> str:
        user = f"Document excerpts:\n\n{build_context(sources)}\n\nQuestion: {question}"
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:
            logger.exception("LLM request failed")
            raise GenerationError(friendly_error(exc, self.provider, self.model)) from exc
        return (response.choices[0].message.content or "").strip() or NOT_FOUND


class ExtractiveGenerator:
    """Fallback when no LLM key is configured: return the best passages verbatim."""

    llm_used = False

    def generate(self, question: str, sources: list[SearchResult]) -> str:
        best = sources[:3]
        lines = [
            "No language model is configured (add GEMINI_API_KEY to .env for free AI answers), "
            "so here are the most relevant passages:"
        ]
        for i, s in enumerate(best, start=1):
            text = " ".join(s.chunk.text.split())  # PDF line breaks read badly inline
            excerpt = text if len(text) <= 500 else text[:500].rsplit(" ", 1)[0] + " …"
            lines.append(f"{excerpt} [{i}]")
        return "\n\n".join(lines)
