"""Tests for chunking (app.rag.chunker)."""

import pytest

from app.rag.chunker import chunk_document, split_text
from app.rag.models import Document, Page


def test_short_text_is_a_single_chunk():
    assert split_text("Just one sentence.", 100, 20) == ["Just one sentence."]


def test_empty_text_gives_no_chunks():
    assert split_text("   ", 100, 20) == []


def test_chunks_respect_size_and_cover_all_words():
    words = [f"word{i}" for i in range(400)]
    text = " ".join(words)

    chunks = split_text(text, 200, 40)

    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert set(" ".join(chunks).split()) == set(words)


def test_consecutive_chunks_overlap_on_word_boundaries():
    text = " ".join(f"w{i}" for i in range(300))

    chunks = split_text(text, 150, 40)

    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split()[0] in prev.split()  # starts on a whole word seen before


def test_prefers_paragraph_breaks():
    para1 = "Alpha sentence. " * 8
    para2 = "Beta sentence. " * 8
    chunks = split_text(f"{para1.strip()}\n\n{para2.strip()}", 200, 0)

    assert chunks[0].endswith("Alpha sentence.")
    assert chunks[1].startswith("Beta")


def test_invalid_settings_raise():
    with pytest.raises(ValueError):
        split_text("text", 100, 100)


def test_chunk_document_keeps_filename_and_page_numbers():
    doc = Document(
        filename="a.pdf",
        total_pages=3,
        pages=(Page("a.pdf", 1, "x " * 200), Page("a.pdf", 3, "Last page")),
    )

    chunks = chunk_document(doc, 150, 20)

    assert {c.page_number for c in chunks} == {1, 3}
    assert all(c.filename == "a.pdf" for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[-1].text == "Last page" and chunks[-1].chunk_index == 0
