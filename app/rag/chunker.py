"""Split page text into overlapping chunks that keep their page number."""

from app.rag.models import Chunk, Document

# Preferred break points, from strongest to weakest.
_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into pieces of at most chunk_size characters.

    Cuts at the strongest separator found in the second half of each window,
    and starts the next piece `overlap` characters earlier (on a word boundary)
    so sentences spanning a cut stay retrievable.
    """
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("require chunk_size > 0 and 0 <= overlap < chunk_size")

    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in _SEPARATORS:
                cut = window.rfind(sep)
                if cut >= chunk_size // 2:
                    end = start + cut + len(sep)
                    break

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break

        next_start = max(end - overlap, start + 1)
        if overlap and not text[next_start - 1].isspace():
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = next_start
    return pieces


def chunk_document(document: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk each page separately so every chunk maps to exactly one page."""
    chunks = []
    for page in document.pages:
        for index, piece in enumerate(split_text(page.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{document.filename}::p{page.page_number}::c{index}",
                    filename=document.filename,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=piece,
                )
            )
    return chunks
