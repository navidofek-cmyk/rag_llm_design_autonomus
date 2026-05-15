# ingest.py — Document Loading and Chunking

**File:** `src/local_rag/ingest.py`

This module is the entry point for all data entering the RAG system. Its job is
to take raw files (PDFs, RST, Markdown, plain text) and convert them into small,
uniform pieces called *chunks* that can be embedded and stored.

---

## Why Chunking At All?

Large Language Models have a finite **context window** — the maximum amount of
text they can process in a single call. A full technical book might be hundreds
of thousands of characters, far more than most models can accept. Even if a
model could handle the full document, flooding it with irrelevant text hurts
answer quality.

Retrieval-Augmented Generation solves this by:

1. Splitting documents into small chunks at ingest time.
2. Embedding each chunk as a vector.
3. At query time, finding the top-K *most relevant* chunks (not the whole
   document) and passing only those to the LLM.

This means the LLM always receives a tightly focused prompt regardless of how
large the original corpus is.

---

## Fixed-Size vs Semantic Chunking

There are two main chunking strategies, and each has trade-offs:

| Strategy | How it works | Pros | Cons |
|---|---|---|---|
| **Fixed-size** | Split every N characters, step by (N - overlap) | Simple, fast, deterministic | Can cut mid-sentence |
| **Semantic** | Split on sentence/paragraph boundaries using NLP | Respects linguistic units | Slower, harder to implement uniformly |

This project uses **fixed-size chunking** (`_chunk_text`). For a learning
project the simplicity wins: no NLP dependency, no edge cases around unusual
sentence structures, and the overlap window is enough to keep most ideas intact
across chunk boundaries.

---

## Why 512 Characters and 64 Overlap?

These values were chosen to balance two competing pressures:

- **Too small** (e.g. 128 chars): chunks carry very little meaning, retrieval
  becomes noisy, and you need many more calls to cover a concept.
- **Too large** (e.g. 2048 chars): fewer chunks, but each is harder to match
  precisely to a query. The LLM also receives more irrelevant context.

512 characters is roughly 80–100 words — one substantial paragraph. This is
small enough for precise vector search but large enough to hold a complete idea.

The **64-character overlap** ensures that a sentence split across two chunk
boundaries still appears intact in at least one of them. Without overlap,
retrieval could silently miss a key sentence that happened to straddle the split
point.

Both values are configurable via environment variables:

```python
# src/local_rag/ingest.py
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "64"))
```

Set `CHUNK_SIZE=256 CHUNK_OVERLAP=32 uv run rag ingest ...` to experiment.

---

## The Chunk Dataclass

```python
@dataclass
class Chunk:
    text: str      # The actual text content of this chunk
    source: str    # Absolute path to the original file
    page: int      # Page number (1-based for PDFs, always 1 for text files)
    chunk_id: str  # MD5 hash of the text — used as a unique ID in Chroma
```

Each field serves a purpose:

- **`text`** — this is what gets embedded and later shown to the LLM as context.
- **`source`** — provenance: when the LLM cites a source, it uses this path.
  Visible in `uv run rag ask "..." --sources`.
- **`page`** — helps users find the exact page in a PDF. For plain text files,
  this is always `1` because there are no page boundaries.
- **`chunk_id`** — a deterministic identifier. Because it is the MD5 of the
  text, the same text always produces the same ID. Chroma uses IDs to deduplicate:
  if you ingest the same file twice, Chroma silently ignores chunks whose IDs
  already exist. This makes ingest idempotent.

---

## How chunk_id Works (MD5 Deduplication)

```python
def _make_chunk_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
```

MD5 produces a 32-character hex string from any input. Two identical strings
always produce the same hash. Two different strings almost never produce the
same hash (collisions are astronomically rare for text of this length).

This is not used for security — MD5 is cryptographically broken — but for
**content addressing**: a chunk's identity is its content. If you update a file,
the changed chunks get new IDs (new content = new hash) and are added fresh,
while unchanged chunks keep their old IDs and are skipped by Chroma.

---

## Code Walkthrough: `_chunk_text`

```python
def _chunk_text(text: str, source: str, page: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    length = len(text)
    while start < length:
        end = start + CHUNK_SIZE          # cut window end
        chunk_text = text[start:end]      # slice the text
        if chunk_text.strip():            # skip whitespace-only slices
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    page=page,
                    chunk_id=_make_chunk_id(chunk_text),
                )
            )
        if end >= length:                 # reached end of text — done
            break
        start = end - CHUNK_OVERLAP       # step back by overlap for next chunk
    return chunks
```

Step through a small example. Say `CHUNK_SIZE=10`, `CHUNK_OVERLAP=3`,
and `text = "Hello World Python Programming"` (30 chars):

| Iteration | `start` | `end` | `chunk_text` |
|---|---|---|---|
| 1 | 0 | 10 | `Hello Worl` |
| 2 | 7 | 17 | `orld Pytho` |
| 3 | 14 | 24 | `thon Progr` |
| 4 | 21 | 30 | `amming` |

Notice how "orld" appears in both chunk 1 and 2, and "thon" in chunk 2 and 3.
That is the overlap doing its job: the word "World" is never split irrecoverably.

The `if chunk_text.strip(): ` guard prevents empty chunks — common at the end of
PDF pages that have trailing whitespace.

---

## File Loaders

### `load_pdf(path)`

Uses PyMuPDF (`fitz`) to open the PDF and iterate page by page. Each page's
text is extracted with `page.get_text()` and chunked. The `page_num` (1-based)
is stored so users can locate the original source.

```python
def load_pdf(path: Path) -> list[Chunk]:
    import fitz
    doc = fitz.open(str(path))
    try:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                chunks.extend(_chunk_text(text, source=str(path), page=page_num))
    finally:
        doc.close()   # always close even if extraction fails midway
    return chunks
```

The `finally` block ensures the file handle is released even if a page raises
an exception.

### `load_text(path)`

Reads the entire file as UTF-8 text (using `errors="replace"` to handle
malformed bytes gracefully) and feeds it to `_chunk_text` with `page=1`.

### `load_file(path)`

Dispatch function: inspects the file extension and delegates to the right
loader. Unknown extensions return an empty list (no crash).

```python
SUPPORTED_SUFFIXES = {".pdf", ".rst", ".md", ".txt"}
```

### `load_directory(directory, glob="**/*")`

Recursively walks a directory using `Path.glob("**/*")`. The `**` pattern means
"any number of nested subdirectories". For each file found, it checks whether
the suffix is in `SUPPORTED_SUFFIXES` before calling `load_file`.

```python
def load_directory(directory: Path, glob: str = "**/*") -> list[Chunk]:
    chunks: list[Chunk] = []
    for file_path in directory.glob(glob):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
            chunks.extend(load_file(file_path))
    return chunks
```

The `glob` parameter is overridable — you could call
`load_directory(path, glob="*.pdf")` to load only top-level PDFs.

---

## Summary

| Concept | Implementation | Why |
|---|---|---|
| Chunking | Fixed-size sliding window | Simple, fast, deterministic |
| Chunk size | 512 chars | ~1 paragraph, good retrieval granularity |
| Overlap | 64 chars | Prevent information loss at boundaries |
| Chunk ID | MD5(text) | Idempotent re-ingest, deduplication in Chroma |
| PDF loading | PyMuPDF page-by-page | Preserves page numbers for citation |
| Directory traversal | `Path.glob("**/*")` | Handles arbitrarily nested file trees |
