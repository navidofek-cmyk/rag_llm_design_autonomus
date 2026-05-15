# Full Architecture

## Overview

This project is a complete RAG system with two phases:

1. **Ingest phase** — process documents once, build a searchable index.
2. **Query phase** — answer questions by retrieving relevant chunks and passing them to an LLM.

All configuration is driven by environment variables in `.env`, letting you toggle features like hybrid retrieval and reranking without changing any code.

---

## Ingest Phase: Document → Searchable Index

```
┌─────────────────────────────────────────────────────────────┐
│                        INGEST PHASE                         │
└─────────────────────────────────────────────────────────────┘

  Input: /path/to/documents/  (PDF, RST, MD, TXT)
         │
         ▼
  ┌─────────────┐
  │  ingest.py  │  load_file() / load_directory()
  │             │  - PDF: PyMuPDF (fitz) page-by-page
  │             │  - RST/MD/TXT: read as UTF-8
  │             │
  │  _chunk_text() splits into 512-char chunks
  │  with 64-char overlap (configurable via .env)
  │
  │  Each chunk becomes a Chunk dataclass:
  │    .text      = raw text (≤512 chars)
  │    .source    = absolute file path
  │    .page      = page number (1-based)
  │    .chunk_id  = MD5(text)  ← deduplication key
  └──────┬──────┘
         │  list[Chunk]
         ▼
  ┌─────────────┐
  │  embed.py   │  Embedder.embed(texts)
  │             │  SentenceTransformer("BAAI/bge-small-en")
  │             │  batch size = 512 chunks (pipeline default)
  │             │  normalize_embeddings=True
  │             │
  │  Output: np.ndarray shape (N, 384)
  └──────┬──────┘
         │  embeddings
         ▼
  ┌─────────────┐
  │  store.py   │  add_chunks(chunks, embeddings, chroma_dir, collection)
  │             │  - Chroma PersistentClient(path=chroma_dir)
  │             │  - get_or_create_collection(name)
  │             │  - Batched at 5000 chunks per .add() call
  │             │
  │  Stored per chunk: id (MD5), embedding (384 floats),
  │                    document (text), metadata (source, page)
  └──────┬──────┘
         │
         ▼
  data/chroma/          ← persisted on disk
  ├── chroma.sqlite3    ← texts, metadata
  └── <uuid>/
      └── *.bin         ← HNSW vector index

  Side effect: pipeline._bm25_cache.pop(collection)
               ← BM25 index for this collection is invalidated
                  (rebuilt lazily on next query)
```

### The 512-char Chunk With 64-char Overlap

The chunking is character-based, not word or sentence based. With a chunk size of 512 and overlap of 64:

```
Text: [............512 chars............][64 char overlap]
                         [............512 chars............]
```

The overlap ensures that a sentence or code expression at the boundary of one chunk also appears at the start of the next chunk. Without overlap, a critical sentence that straddles a boundary would be split in half and potentially irretrievable.

The MD5 `chunk_id` provides automatic deduplication: if you ingest the same file twice, Chroma sees that the IDs already exist and skips them.

---

## Query Phase: Question → Answer

### Dense-Only Mode (`USE_HYBRID=false`, `USE_RERANKER=false`)

```
┌─────────────────────────────────────────────────────────────┐
│                   QUERY PHASE (dense only)                  │
└─────────────────────────────────────────────────────────────┘

  Question: "How does Python's GIL work?"
         │
         ▼
  ┌─────────────┐
  │  embed.py   │  Embedder.embed_one(question)
  │             │  → np.ndarray shape (384,)
  └──────┬──────┘
         │  query_embedding
         ▼
  ┌─────────────┐
  │  store.py   │  query(embedding, chroma_dir, collection, top_k=5)
  │             │  Chroma HNSW approximate nearest-neighbor search
  │             │  Returns top_k chunks sorted by distance (asc)
  └──────┬──────┘
         │  list[dict] with keys: text, source, page, distance
         ▼
  ┌─────────────┐
  │   llm.py   │  OllamaClient.ask(question, chunks)
  │             │  Builds prompt:
  │             │    SYSTEM: "Answer only from provided context..."
  │             │    USER:   "Context:\n{chunk1}\n---\n{chunk2}\n...\n
  │             │             \nQuestion: {question}"
  │             │  POST /api/chat → Ollama → model response
  └──────┬──────┘
         │
         ▼
  answer (str) + chunks (list[dict] with source attribution)
```

### Hybrid Mode (`USE_HYBRID=true`)

The retrieval step is expanded:

```
         │  query_embedding
         ▼
  ┌──────────────────────────────────────────┐
  │              retrieve.py                 │
  │  Dense: store.query(embedding, fetch_k=20)│
  │                   +                       │
  │              bm25.py                      │
  │  Sparse: BM25Okapi.get_scores(query tokens)│
  │          top fetch_k=20 by BM25 score     │
  └────────────────┬─────────────────────────┘
                   │  dense[20] + sparse[20]
                   ▼
  ┌─────────────────────────────────────────┐
  │               hybrid.py                 │
  │  reciprocal_rank_fusion(dense, sparse)  │
  │  score = 1/(60 + rank), sum across lists│
  │  return fused[:top_k=5]                  │
  └────────────────┬────────────────────────┘
                   │  fused list[dict] with rrf_score
```

### Hybrid + Reranker Mode (`USE_HYBRID=true`, `USE_RERANKER=true`)

After fusion, an additional reranking step is inserted:

```
                   │  fused list[dict] — potentially many candidates
                   ▼
  ┌─────────────────────────────────────────┐
  │              reranker.py                │
  │  Reranker.rerank(question, chunks, top_k)│
  │  CrossEncoder("BAAI/bge-reranker-base") │
  │  pairs = [(question, chunk.text) ...]   │
  │  model.predict(all pairs at once)       │
  │  sort descending by rerank_score        │
  │  return top_k                           │
  └────────────────┬────────────────────────┘
                   │  reranked list[dict] with rerank_score
                   ▼
               llm.py → answer
```

### All Modes Side By Side

```
USE_HYBRID=false, USE_RERANKER=false:
  question → embed → dense(top_k) → LLM → answer

USE_HYBRID=true, USE_RERANKER=false:
  question → embed → dense(fetch_k) + BM25(fetch_k) → RRF → top_k → LLM → answer

USE_HYBRID=false, USE_RERANKER=true:
  question → embed → dense(top_k) → rerank → LLM → answer

USE_HYBRID=true, USE_RERANKER=true:
  question → embed → dense(fetch_k) + BM25(fetch_k) → RRF → rerank → LLM → answer
```

---

## The BM25 Lazy Cache

The pipeline maintains a `_bm25_cache` dictionary keyed by collection name:

```python
class RAGPipeline:
    def __init__(self):
        ...
        self._bm25_cache: dict[str, bm25.BM25Index] = {}

    def _get_bm25_index(self, collection: str) -> bm25.BM25Index:
        if collection not in self._bm25_cache:
            self._bm25_cache[collection] = bm25.build_index(self.chroma_dir, collection)
        return self._bm25_cache[collection]

    def ingest(self, path: str, collection: str = "default", ...) -> int:
        ...
        store.add_chunks(chunks, embeddings, self.chroma_dir, collection)
        self._bm25_cache.pop(collection, None)   # ← invalidate cache
        return len(chunks)
```

### Why Lazy?

Building a BM25 index requires fetching all documents from Chroma and computing term frequencies. For a large collection this takes a few seconds. If you run many queries against multiple collections but only actually use BM25 for a few of them (or if `USE_HYBRID=false`), building all indexes upfront would be wasteful.

Lazy initialization — "build on first use" — means you only pay the cost when you actually need it.

### Why Invalidate After Ingest?

The BM25 index is a static data structure built from a snapshot of the collection at build time. After you ingest new documents into a collection, the Chroma store has new chunks that are NOT in the BM25 index. If you query without rebuilding, those new chunks are invisible to BM25 (though they are visible to dense retrieval, since Chroma's HNSW index is updated immediately).

By deleting the cache entry after ingest (`self._bm25_cache.pop(collection, None)`), the next query against that collection triggers a fresh index build from the updated Chroma data. The cost is paid once per ingest cycle, which is acceptable.

---

## Pipeline Orchestration (`pipeline.py`)

`RAGPipeline` is the central object that owns all resources and wires everything together:

```python
class RAGPipeline:
    def __init__(self):
        self.chroma_dir = os.getenv("CHROMA_DIR", "./data/chroma")
        self.top_k      = int(os.getenv("TOP_K", "5"))
        self.fetch_k    = int(os.getenv("FETCH_K", "20"))
        self.use_hybrid  = os.getenv("USE_HYBRID", "false").lower() == "true"
        self.use_reranker = os.getenv("USE_RERANKER", "false").lower() == "true"
        self.embedder    = embed.Embedder()          # loads BAAI/bge-small-en
        self.llm_client  = llm.OllamaClient()
        self._bm25_cache: dict[str, bm25.BM25Index] = {}
        if self.use_reranker:
            self.reranker = reranker.Reranker()      # loads CrossEncoder
        else:
            self.reranker = None
```

Note that `Reranker()` (which downloads ~278MB) is only instantiated when `USE_RERANKER=true`. This keeps startup fast in the common case.

---

## API Layer (`api.py`)

The FastAPI application wraps the pipeline behind HTTP endpoints:

```
GET  /health                              → {"status": "ok"}
GET  /ask?q=...&collection=...&top_k=5   → AskResponse
GET  /collections                         → ["default", "peps", ...]
POST /ingest  {"path": "...", "collection": "..."} → {"chunks": N}
```

The pipeline is a singleton via `functools.lru_cache(maxsize=1)`:

```python
@functools.lru_cache(maxsize=1)
def get_pipeline():
    from local_rag.pipeline import RAGPipeline
    return RAGPipeline()
```

This ensures the embedding model and (optionally) the CrossEncoder are loaded once at the first request and reused for all subsequent requests. Loading a 130MB embedding model on every HTTP request would be catastrophically slow.

---

## CLI Commands (`cli.py`)

```
uv run rag --help

Commands:
  ingest     Ingest documents into the vector store
  ask        Ask a question against the RAG pipeline
  eval       Evaluate on a Q&A JSONL file
  inspect    Inspect a collection (count + sample chunks)
  batch-ask  Batch process questions from a JSONL file
  serve      Start the FastAPI server (uvicorn)
  ui         Launch the Gradio web UI
```

All commands create a fresh `RAGPipeline` instance, so they read `.env` at startup. The pipeline initialization includes loading the embedding model — expect 3–10 seconds for the first command while the model loads (subsequent commands in the same process reuse the loaded model).

---

## Data Flow Summary

```
                    ┌────────────────────────────────┐
                    │         USER / CLI / API         │
                    └───────────────┬────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                      │
        INGEST                   QUERY                  SWARM
              │                     │                      │
    ┌─────────▼──────┐     ┌────────▼───────┐    ┌────────▼───────┐
    │  ingest.py     │     │  embed.py      │    │  swarm.py      │
    │  (chunking)    │     │  (1 query vec) │    │  ThreadPool    │
    └────────┬───────┘     └────────┬───────┘    │  N agents      │
             │                      │            └────────┬───────┘
    ┌────────▼───────┐              │                     │
    │  embed.py      │     ┌────────▼───────┐    ┌────────▼───────┐
    │  (batch embed) │     │  retrieve.py   │    │  hybrid.py     │
    └────────┬───────┘     │  store.py      │    │  RRF merge     │
             │             └────────┬───────┘    └────────┬───────┘
    ┌────────▼───────┐              │                     │
    │  store.py      │     [if USE_HYBRID]                │
    │  Chroma write  │     ┌────────▼───────┐             │
    └────────────────┘     │  bm25.py       │             │
                           │  hybrid.py     │             │
                           └────────┬───────┘             │
                                    │                      │
                           [if USE_RERANKER]               │
                           ┌────────▼───────┐             │
                           │  reranker.py   │             │
                           └────────┬───────┘             │
                                    │◄────────────────────┘
                           ┌────────▼───────┐
                           │  llm.py        │
                           │  Ollama /chat  │
                           └────────┬───────┘
                                    │
                              answer + sources
```
