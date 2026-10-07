"""REST API consumed by the frontend."""

from functools import lru_cache

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.rag.embeddings import LocalEmbedder
from app.rag.llm import ExtractiveGenerator, GenerationError, OpenAIGenerator
from app.rag.pdf_loader import EmptyPDFError, PDFLoadError
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import VectorStore

router = APIRouter(prefix="/api")


@lru_cache
def get_pipeline() -> RAGPipeline:
    llm = settings.llm
    generator = (
        OpenAIGenerator(llm.api_key, llm.model, llm.base_url, provider=llm.provider)
        if llm.provider != "none"
        else ExtractiveGenerator()
    )
    return RAGPipeline(
        embedder=LocalEmbedder(),
        store=VectorStore(settings.chroma_dir),
        generator=generator,
        documents_dir=settings.documents_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    documents: list[str] | None = None


@router.get("/status")
def status(pipeline: RAGPipeline = Depends(get_pipeline)) -> dict:
    return {
        "llm_enabled": pipeline.generator.llm_used,
        "model": getattr(pipeline.generator, "model", None),
        "provider": getattr(pipeline.generator, "provider", None),
        "documents": len(pipeline.list_documents()),
        "max_upload_mb": settings.max_upload_mb,
    }


@router.post("/documents", status_code=201)
def upload_document(file: UploadFile = File(...), pipeline: RAGPipeline = Depends(get_pipeline)) -> dict:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    data = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")

    try:
        result = pipeline.ingest_pdf(data, filename)
    except EmptyPDFError as exc:
        raise HTTPException(422, f"{exc}. Scanned (image-only) PDFs are not supported.") from exc
    except PDFLoadError as exc:
        raise HTTPException(400, str(exc)) from exc
    return result.__dict__


@router.get("/documents")
def list_documents(pipeline: RAGPipeline = Depends(get_pipeline)) -> list[dict]:
    return pipeline.list_documents()


@router.delete("/documents/{filename}")
def delete_document(filename: str, pipeline: RAGPipeline = Depends(get_pipeline)) -> dict:
    if not pipeline.delete_document(filename):
        raise HTTPException(404, "Document not found.")
    return {"deleted": filename}


@router.get("/documents/{filename}/file")
def get_document_file(filename: str, pipeline: RAGPipeline = Depends(get_pipeline)) -> FileResponse:
    path = pipeline.document_path(filename)
    if path is None:
        raise HTTPException(404, "Document not found.")
    return FileResponse(path, media_type="application/pdf", content_disposition_type="inline")


@router.post("/ask")
def ask(request: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> dict:
    if not request.question.strip():
        raise HTTPException(422, "Question must not be empty.")
    try:
        answer = pipeline.ask(request.question, request.top_k, request.documents)
    except GenerationError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {
        "question": answer.question,
        "answer": answer.answer,
        "llm_used": answer.llm_used,
        "citations": [c.__dict__ for c in answer.citations],
    }
