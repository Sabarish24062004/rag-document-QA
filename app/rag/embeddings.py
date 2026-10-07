"""Text -> vector conversion.

Uses the all-MiniLM-L6-v2 sentence-embedding model (ONNX, runs locally on CPU,
384 dimensions) bundled with ChromaDB, so no API key is needed. The model
(~80 MB) is downloaded to ~/.cache/chroma on first use.
"""

import threading
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    """Lazily loads the model on first call, so app startup stays fast."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            if self._model is None:
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

                self._model = DefaultEmbeddingFunction()
        return [vector.tolist() for vector in self._model(texts)]
