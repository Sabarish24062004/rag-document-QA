"""Shared fixtures: an offline pipeline with fake embedder/LLM and a temp vector DB."""

import hashlib
import math
import re
from dataclasses import replace

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import config
from app.api.routes import get_pipeline
from app.main import app
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import VectorStore


class FakeEmbedder:
    """Bag-of-words hashed into a fixed-size unit vector: texts sharing words are close."""

    dims = 256

    def embed(self, texts):
        vectors = []
        for text in texts:
            v = [0.0] * self.dims
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


class FakeGenerator:
    """Cites the first source, like a well-behaved LLM would."""

    llm_used = True
    model = "fake-model"

    def __init__(self):
        self.calls = []

    def generate(self, question, sources):
        self.calls.append((question, sources))
        return f"Answer based on the documents [1]."


def make_pdf(page_texts: list[str]) -> bytes:
    """Build a PDF in memory; an empty string produces a blank page."""
    pdf = pymupdf.open()
    for text in page_texts:
        page = pdf.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text)
    data = pdf.tobytes()
    pdf.close()
    return data


@pytest.fixture
def pipeline(tmp_path):
    return RAGPipeline(
        embedder=FakeEmbedder(),
        store=VectorStore(tmp_path / "chroma"),
        generator=FakeGenerator(),
        documents_dir=tmp_path / "docs",
        chunk_size=300,
        chunk_overlap=50,
        top_k=3,
    )


def set_password(monkeypatch, password: str) -> None:
    monkeypatch.setattr(config, "settings", replace(config.settings, app_password=password))


@pytest.fixture
def client(pipeline, monkeypatch):
    set_password(monkeypatch, "")  # independent of APP_PASSWORD in .env
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
