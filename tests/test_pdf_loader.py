"""Tests for PDF ingestion (app.rag.pdf_loader)."""

import pymupdf
import pytest

from app.rag.models import Document, Page
from app.rag.pdf_loader import EmptyPDFError, PDFLoadError, clean_text, load_pdf


def make_pdf(page_texts: list[str], password: str | None = None) -> bytes:
    """Build a PDF in memory; an empty string produces a blank page."""
    pdf = pymupdf.open()
    for text in page_texts:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    options = {}
    if password:
        options = {"encryption": pymupdf.PDF_ENCRYPT_AES_256, "user_pw": password, "owner_pw": password}
    data = pdf.tobytes(**options)
    pdf.close()
    return data


def test_extracts_text_page_by_page_with_metadata():
    doc = load_pdf(make_pdf(["First page", "Second page"]), filename="report.pdf")

    assert isinstance(doc, Document)
    assert doc.filename == "report.pdf"
    assert doc.total_pages == 2
    assert doc.pages == (
        Page(filename="report.pdf", page_number=1, text="First page"),
        Page(filename="report.pdf", page_number=2, text="Second page"),
    )


def test_loads_from_path_and_uses_file_name(tmp_path):
    path = tmp_path / "notes.pdf"
    path.write_bytes(make_pdf(["Hello from disk"]))

    doc = load_pdf(path)

    assert doc.filename == "notes.pdf"
    assert doc.pages[0].text == "Hello from disk"


def test_blank_pages_are_skipped_but_numbering_is_preserved():
    doc = load_pdf(make_pdf(["Intro", "", "Conclusion"]), filename="gaps.pdf")

    assert doc.total_pages == 3
    assert [p.page_number for p in doc.pages] == [1, 3]


def test_filename_is_stripped_of_directories():
    doc = load_pdf(make_pdf(["x"]), filename="uploads/sub/file.PDF")
    assert doc.filename == "file.PDF"


def test_pdf_with_no_text_raises_empty_error():
    with pytest.raises(EmptyPDFError):
        load_pdf(make_pdf(["", ""]), filename="blank.pdf")


def test_corrupt_bytes_raise_load_error():
    with pytest.raises(PDFLoadError):
        load_pdf(b"this is not a pdf", filename="broken.pdf")


def test_empty_bytes_raise_load_error():
    with pytest.raises(PDFLoadError):
        load_pdf(b"", filename="empty.pdf")


def test_password_protected_pdf_raises_load_error():
    with pytest.raises(PDFLoadError, match="password"):
        load_pdf(make_pdf(["secret"], password="pw"), filename="locked.pdf")


def test_non_pdf_extension_is_rejected():
    with pytest.raises(PDFLoadError, match="Not a PDF"):
        load_pdf(make_pdf(["x"]), filename="notes.txt")


def test_bytes_without_filename_are_rejected():
    with pytest.raises(PDFLoadError, match="filename"):
        load_pdf(make_pdf(["x"]))


def test_missing_file_raises_load_error(tmp_path):
    with pytest.raises(PDFLoadError, match="not found"):
        load_pdf(tmp_path / "missing.pdf")


def test_clean_text_normalises_whitespace():
    raw = "  Title\t\twith   gaps \r\n\r\n\r\n\r\nNext\x00 para  \n"
    assert clean_text(raw) == "Title with gaps\n\nNext para"
