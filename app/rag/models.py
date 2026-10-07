"""Internal document representation shared by the RAG pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    """Text extracted from a single PDF page.

    page_number is 1-based so it can be shown directly in citations.
    """

    filename: str
    page_number: int
    text: str


@dataclass(frozen=True)
class Document:
    """A loaded PDF: its source filename and the pages that contained text.

    total_pages counts every page in the file, including blank ones that were
    skipped, so len(pages) may be smaller than total_pages.
    """

    filename: str
    total_pages: int
    pages: tuple[Page, ...]


@dataclass(frozen=True)
class Chunk:
    """A piece of one page's text; the unit that is embedded and retrieved."""

    id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk with its cosine similarity to the question (higher = closer)."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    """A numbered source shown to the LLM and the user as [number]."""

    number: int
    filename: str
    page_number: int
    text: str
    score: float
    cited: bool


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    citations: tuple[Citation, ...]
    llm_used: bool
