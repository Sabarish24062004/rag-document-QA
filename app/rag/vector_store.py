"""Persistent vector storage and similarity search with ChromaDB."""

from pathlib import Path

import chromadb

from app.rag.models import Chunk, SearchResult

COLLECTION = "documents"


class VectorStore:
    """Stores chunk vectors plus their filename/page metadata.

    Embeddings are always supplied by the caller (embedding_function=None), so
    the same Embedder is used for chunks and questions.
    """

    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            COLLECTION,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]], doc_meta: dict) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {**doc_meta, "filename": c.filename, "page_number": c.page_number, "chunk_index": c.chunk_index}
                for c in chunks
            ],
        )

    def delete_document(self, filename: str) -> int:
        """Remove all chunks of a document; returns how many were removed."""
        ids = self._collection.get(where={"filename": filename}, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def search(
        self, embedding: list[float], top_k: int, filenames: list[str] | None = None
    ) -> list[SearchResult]:
        if self._collection.count() == 0:
            return []
        where = None
        if filenames:
            where = {"filename": filenames[0]} if len(filenames) == 1 else {"filename": {"$in": filenames}}
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            chunk = Chunk(
                id=chunk_id,
                filename=meta["filename"],
                page_number=int(meta["page_number"]),
                chunk_index=int(meta["chunk_index"]),
                text=text,
            )
            hits.append(SearchResult(chunk=chunk, score=round(1.0 - distance, 4)))
        return hits

    def list_documents(self) -> list[dict]:
        """One summary row per stored document, newest first."""
        docs: dict[str, dict] = {}
        for meta in self._collection.get(include=["metadatas"])["metadatas"]:
            name = meta["filename"]
            row = docs.setdefault(
                name,
                {
                    "filename": name,
                    "total_pages": meta.get("total_pages", 0),
                    "uploaded_at": meta.get("uploaded_at", ""),
                    "chunks": 0,
                },
            )
            row["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["uploaded_at"], reverse=True)

    def has_document(self, filename: str) -> bool:
        return bool(self._collection.get(where={"filename": filename}, limit=1, include=[])["ids"])
