"""PDF ingestion: extract text page by page with PyMuPDF."""

import logging
import re
from pathlib import Path

import pymupdf

from app.rag.models import Document, Page

logger = logging.getLogger(__name__)


class PDFLoadError(ValueError):
    """The input is not a readable PDF (wrong type, corrupt, or encrypted)."""


class EmptyPDFError(PDFLoadError):
    """The PDF opened but contains no extractable text (e.g. blank or scanned)."""


_SPACES = re.compile(r"[ \t\f\v ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalise whitespace while keeping paragraph breaks."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def load_pdf(source: str | Path | bytes, filename: str | None = None) -> Document:
    """Load a PDF from a file path or raw bytes and return its pages' text.

    filename is required when source is bytes (e.g. an uploaded file); for a
    path it defaults to the file's name. Pages without text are skipped.

    Raises PDFLoadError if the input can't be read as a PDF, and EmptyPDFError
    if it has no extractable text on any page.
    """
    if isinstance(source, bytes):
        if not filename:
            raise PDFLoadError("filename is required when loading PDF bytes")
        open_kwargs = {"stream": source, "filetype": "pdf"}
    else:
        path = Path(source)
        filename = filename or path.name
        if not path.is_file():
            raise PDFLoadError(f"File not found: {path}")
        open_kwargs = {"filename": str(path), "filetype": "pdf"}

    filename = Path(filename).name
    if Path(filename).suffix.lower() != ".pdf":
        raise PDFLoadError(f"Not a PDF file: {filename}")

    try:
        pdf = pymupdf.open(**open_kwargs)
    except Exception as exc:  # PyMuPDF raises several types for bad input
        raise PDFLoadError(f"Could not open {filename}: {exc}") from exc

    with pdf:
        if pdf.needs_pass:
            raise PDFLoadError(f"{filename} is password-protected")
        if pdf.page_count == 0:
            raise EmptyPDFError(f"{filename} has no pages")

        pages = []
        for index, pdf_page in enumerate(pdf):
            try:
                text = clean_text(pdf_page.get_text("text"))
            except Exception as exc:
                logger.warning("Skipping %s page %d: %s", filename, index + 1, exc)
                continue
            if text:
                pages.append(Page(filename=filename, page_number=index + 1, text=text))
        total_pages = pdf.page_count

    if not pages:
        raise EmptyPDFError(f"{filename} contains no extractable text")

    logger.info("Loaded %s: %d/%d pages with text", filename, len(pages), total_pages)
    return Document(filename=filename, total_pages=total_pages, pages=tuple(pages))
