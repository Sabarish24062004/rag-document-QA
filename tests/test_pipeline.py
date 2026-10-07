"""Tests for the end-to-end pipeline, vector store and answer generation."""

import pytest

from app.rag.llm import NOT_FOUND, ExtractiveGenerator, build_context
from app.rag.models import Chunk, SearchResult
from app.rag.pdf_loader import EmptyPDFError
from tests.conftest import make_pdf

SOLAR = make_pdf([
    "Solar panels convert sunlight into electricity using photovoltaic cells.",
    "Wind turbines generate power from moving air.",
])
COOKING = make_pdf(["Bake the bread at 220 degrees for thirty minutes."])


def test_ingest_reports_counts_and_saves_file(pipeline):
    result = pipeline.ingest_pdf(SOLAR, "energy.pdf")

    assert result.filename == "energy.pdf"
    assert result.total_pages == 2
    assert result.chunks == 2
    assert not result.replaced
    assert (pipeline.documents_dir / "energy.pdf").read_bytes() == SOLAR
    assert pipeline.list_documents()[0] | {"uploaded_at": ""} == {
        "filename": "energy.pdf", "total_pages": 2, "chunks": 2, "uploaded_at": "",
    }


def test_reuploading_same_name_replaces_document(pipeline):
    pipeline.ingest_pdf(SOLAR, "doc.pdf")
    result = pipeline.ingest_pdf(COOKING, "doc.pdf")

    assert result.replaced
    docs = pipeline.list_documents()
    assert len(docs) == 1 and docs[0]["chunks"] == 1


def test_ask_retrieves_relevant_page_and_cites_it(pipeline):
    pipeline.ingest_pdf(SOLAR, "energy.pdf")
    pipeline.ingest_pdf(COOKING, "recipes.pdf")

    answer = pipeline.ask("How do wind turbines generate power?")

    top = answer.citations[0]
    assert (top.filename, top.page_number) == ("energy.pdf", 2)
    assert top.cited and top.number == 1
    assert not any(c.cited for c in answer.citations[1:])
    assert answer.llm_used
    # The generator received the retrieved chunks, best first.
    _, sources = pipeline.generator.calls[0]
    assert sources[0].chunk.text.startswith("Wind turbines")


def test_ask_can_be_limited_to_selected_documents(pipeline):
    pipeline.ingest_pdf(SOLAR, "energy.pdf")
    pipeline.ingest_pdf(COOKING, "recipes.pdf")

    answer = pipeline.ask("wind turbines", filenames=["recipes.pdf"])

    assert {c.filename for c in answer.citations} == {"recipes.pdf"}


def test_ask_with_no_documents_returns_not_found(pipeline):
    answer = pipeline.ask("anything?")

    assert answer.answer == NOT_FOUND
    assert answer.citations == ()
    assert pipeline.generator.calls == []


def test_delete_removes_vectors_and_file(pipeline):
    pipeline.ingest_pdf(SOLAR, "energy.pdf")

    assert pipeline.delete_document("energy.pdf")
    assert pipeline.list_documents() == []
    assert not (pipeline.documents_dir / "energy.pdf").exists()
    assert not pipeline.delete_document("energy.pdf")


def test_delete_ignores_directory_components(pipeline, tmp_path):
    outside = tmp_path / "keep.pdf"
    outside.write_bytes(b"x")

    pipeline.delete_document("../keep.pdf")

    assert outside.exists()


def test_empty_pdf_is_rejected_without_storing(pipeline):
    with pytest.raises(EmptyPDFError):
        pipeline.ingest_pdf(make_pdf([""]), "blank.pdf")
    assert pipeline.list_documents() == []


def _hit(text, page=1):
    return SearchResult(Chunk("id", "f.pdf", page, 0, text), 0.9)


def test_context_numbers_sources_with_filename_and_page():
    context = build_context([_hit("alpha", 2), _hit("beta", 5)])
    assert "[1] (Source: f.pdf, page 2)\nalpha" in context
    assert "[2] (Source: f.pdf, page 5)\nbeta" in context


def test_extractive_fallback_cites_passages():
    text = ExtractiveGenerator().generate("q", [_hit("first passage"), _hit("second passage")])
    assert "first passage [1]" in text and "second passage [2]" in text


def test_openai_generator_sends_context_and_parses_reply():
    from types import SimpleNamespace

    from app.rag.llm import OpenAIGenerator

    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        message = SimpleNamespace(content=" Two years [1]. ")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    gen = OpenAIGenerator(api_key="test-key", model="some-model")
    gen._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    assert gen.generate("How long?", [_hit("warranty is two years", 4)]) == "Two years [1]."
    assert sent["model"] == "some-model"
    assert sent["messages"][0]["role"] == "system"
    assert "[1] (Source: f.pdf, page 4)" in sent["messages"][1]["content"]
    assert "Question: How long?" in sent["messages"][1]["content"]


def test_openai_generator_supports_other_providers_and_keyless_local_servers():
    from app.rag.llm import OpenAIGenerator

    gen = OpenAIGenerator(api_key=None, model="llama3.2:3b", base_url="http://localhost:11434/v1")

    assert str(gen._client.base_url).rstrip("/") == "http://localhost:11434/v1"
    assert gen.model == "llama3.2:3b"
