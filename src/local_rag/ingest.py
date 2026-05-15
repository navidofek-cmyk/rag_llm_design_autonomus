import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "64"))

SUPPORTED_SUFFIXES = {".pdf", ".rst", ".md", ".txt"}


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str


def _make_chunk_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _chunk_text(text: str, source: str, page: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    length = len(text)
    while start < length:
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    page=page,
                    chunk_id=_make_chunk_id(chunk_text),
                )
            )
        if end >= length:
            break
        start = end - CHUNK_OVERLAP
    return chunks


def load_pdf(path: Path) -> list[Chunk]:
    import fitz  # type: ignore[import-untyped]

    chunks: list[Chunk] = []
    doc = fitz.open(str(path))
    try:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                chunks.extend(_chunk_text(text, source=str(path), page=page_num))
    finally:
        doc.close()
    return chunks


def load_text(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return _chunk_text(text, source=str(path), page=1)


def load_file(path: Path) -> list[Chunk]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in {".rst", ".md", ".txt"}:
        return load_text(path)
    else:
        return []


def load_directory(directory: Path, glob: str = "**/*") -> list[Chunk]:
    chunks: list[Chunk] = []
    for file_path in directory.glob(glob):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
            chunks.extend(load_file(file_path))
    return chunks
