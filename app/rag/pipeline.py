"""End-to-end RAG workflow.

Ingest:  PDF -> extract text -> split into chunks -> embed chunks -> store in vector DB
Ask:     question -> embed -> retrieve similar chunks -> LLM answer -> citations
"""

import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.rag.chunker import chunk_document
from app.rag.embeddings import Embedder
from app.rag.llm import NOT_FOUND, Generator
from app.rag.models import Answer, Citation
from app.rag.pdf_loader import load_pdf
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class IngestResult:
    filename: str
    total_pages: int
    pages_with_text: int
    chunks: int
    replaced: bool


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        generator: Generator,
        documents_dir: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.documents_dir = documents_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self._write_lock = threading.Lock()
        documents_dir.mkdir(parents=True, exist_ok=True)

    # ---- ingestion -------------------------------------------------------

    def ingest_pdf(self, data: bytes, filename: str) -> IngestResult:
        """Index a PDF. Uploading a file with an existing name replaces it.

        Raises PDFLoadError / EmptyPDFError for unusable files.
        """
        document = load_pdf(data, filename=filename)  # 1. extract text per page
        chunks = chunk_document(document, self.chunk_size, self.chunk_overlap)  # 2. split
        embeddings = self.embedder.embed([c.text for c in chunks])  # 3. vectors

        with self._write_lock:  # 4. store (replace any previous version)
            replaced = self.store.delete_document(document.filename) > 0
            self.store.add(
                chunks,
                embeddings,
                doc_meta={
                    "total_pages": document.total_pages,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )
            (self.documents_dir / document.filename).write_bytes(data)

        logger.info("Indexed %s: %d chunks", document.filename, len(chunks))
        return IngestResult(
            filename=document.filename,
            total_pages=document.total_pages,
            pages_with_text=len(document.pages),
            chunks=len(chunks),
            replaced=replaced,
        )

    def list_documents(self) -> list[dict]:
        return self.store.list_documents()

    def delete_document(self, filename: str) -> bool:
        filename = Path(filename).name  # never touch files outside documents_dir
        with self._write_lock:
            removed = self.store.delete_document(filename) > 0
            (self.documents_dir / filename).unlink(missing_ok=True)
        return removed

    def document_path(self, filename: str) -> Path | None:
        path = self.documents_dir / Path(filename).name
        return path if path.is_file() and self.store.has_document(path.name) else None

    # ---- question answering ---------------------------------------------

    def ask(self, question: str, top_k: int | None = None, filenames: list[str] | None = None) -> Answer:
        question = question.strip()
        query_vector = self.embedder.embed([question])[0]  # 1. question -> vector
        hits = self.store.search(query_vector, top_k or self.top_k, filenames)  # 2. retrieve

        if not hits:
            return Answer(question=question, answer=NOT_FOUND, citations=(), llm_used=False)

        text = self.generator.generate(question, hits)  # 3. LLM answer
        cited = {int(n) for n in _CITATION.findall(text)}  # 4. citations
        citations = tuple(
            Citation(
                number=i,
                filename=h.chunk.filename,
                page_number=h.chunk.page_number,
                text=h.chunk.text,
                score=h.score,
                cited=i in cited,
            )
            for i, h in enumerate(hits, start=1)
        )
        return Answer(question=question, answer=text, citations=citations, llm_used=self.generator.llm_used)
