# pipeline.py — Orchestrating the Full RAG Flow

**File:** `src/local_rag/pipeline.py`

`RAGPipeline` is the central coordinator of the entire system. It wires together
ingest, embedding, storage, retrieval, BM25 indexing, hybrid fusion, reranking,
and LLM generation into two clean methods: `ingest()` and `query()`. Everything
else in the codebase — the CLI, the FastAPI server, the Gradio UI — creates a
single `RAGPipeline` instance and calls those two methods.

---

## Why a Pipeline Class?

The alternative to a class would be a chain of module-level function calls
scattered across CLI commands, API handlers, and UI callbacks. That creates three
problems:

1. **Duplication** — every entry point must re-read the same environment
   variables and create the same objects.
2. **Inconsistency** — a typo in one entry point means the CLI and API behave
   differently.
3. **Wasteful re-initialization** — embedding models take 1–3 seconds to load.
   Creating them once in `__init__` and reusing them is far cheaper.

A class solves all three: one place to read config, one place to create heavy
objects, shared across all call sites.

---

## dotenv Loading and Environment Variables

```python
from dotenv import load_dotenv
load_dotenv()
```

This line is called at module import time. It reads `.env` from the current
working directory and injects the values into `os.environ` — but only for keys
that are not already set. This means you can always override `.env` values with
real environment variables:

```bash
# .env says USE_HYBRID=false — override it just for this run:
USE_HYBRID=true uv run rag ask "What is PEP 8?" --collection peps
```

### What Each Variable Controls

| Variable | Default | Effect |
|---|---|---|
| `CHROMA_DIR` | `./data/chroma` | Where the vector database is stored on disk |
| `TOP_K` | `5` | How many chunks to return per query |
| `FETCH_K` | `20` | How many candidates to fetch before hybrid/rerank |
| `USE_HYBRID` | `false` | Enable BM25 + dense fusion (Reciprocal Rank Fusion) |
| `USE_RERANKER` | `false` | Enable CrossEncoder reranking of retrieved chunks |
| `EMBED_MODEL` | `BAAI/bge-small-en` | Sentence-transformers model name |
| `EMBED_DEVICE` | `cpu` | `cpu` or `cuda` |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is running |
| `LLM_MODEL` | `llama3.2` | Which model Ollama should use |
| `CHUNK_SIZE` | `512` | Characters per chunk (read by ingest.py) |
| `CHUNK_OVERLAP` | `64` | Overlap characters between chunks (read by ingest.py) |

---

## `__init__`: Initialization

```python
class RAGPipeline:
    def __init__(self):
        self.chroma_dir = os.getenv("CHROMA_DIR", "./data/chroma")
        self.top_k = int(os.getenv("TOP_K", "5"))
        self.fetch_k = int(os.getenv("FETCH_K", "20"))
        self.use_hybrid = os.getenv("USE_HYBRID", "false").lower() == "true"
        self.use_reranker = os.getenv("USE_RERANKER", "false").lower() == "true"
        self.embedder = embed.Embedder()       # loads model weights — takes ~2 seconds
        self.llm_client = llm.OllamaClient()
        self._bm25_cache: dict[str, bm25.BM25Index] = {}
        if self.use_reranker:
            self.reranker = reranker.Reranker()   # loads CrossEncoder — takes ~3 seconds
        else:
            self.reranker = None
```

The reranker is only instantiated when `USE_RERANKER=true`. Loading a
CrossEncoder when you are not using it wastes ~3 seconds and 300 MB of RAM.

---

## The `ingest()` Flow, Step by Step

```python
def ingest(self, path: str, collection: str = "default", embed_batch: int = 512) -> int:
```

**Step 1 — Load files into chunks**

```python
p = Path(path)
if p.is_dir():
    chunks = ingest.load_directory(p)
else:
    chunks = ingest.load_file(p)
```

`ingest.load_directory` walks the directory recursively. `ingest.load_file`
handles a single PDF, RST, MD, or TXT file. Both return `list[Chunk]`.

**Step 2 — Embed in batches of 512**

```python
all_embeddings = []
for i in range(0, len(chunks), embed_batch):
    batch = chunks[i : i + embed_batch]
    all_embeddings.append(self.embedder.embed([c.text for c in batch]))
embeddings = np.vstack(all_embeddings)
```

Why 512 chunks per batch rather than embedding everything at once?

The embedding model (`BAAI/bge-small-en`) allocates one large float32 array
for the entire batch. On machines without AVX-512 SIMD instructions — which
includes many cloud VMs and older CPUs — NumPy falls back to a software path
that can trigger a `SIGILL` (Illegal Instruction) crash when the array is too
large. Batching at 512 keeps the per-call array size bounded and the process
stable. The `np.vstack` at the end reassembles all batches into one matrix.

**Step 3 — Store in Chroma**

```python
store.add_chunks(chunks, embeddings, self.chroma_dir, collection)
```

Each chunk is stored with its embedding, text, source path, and page number.
Chroma deduplicates by `chunk_id` (MD5 hash), so running ingest twice is safe.

**Step 4 — Invalidate BM25 cache**

```python
self._bm25_cache.pop(collection, None)
```

After new chunks are added, the existing BM25 index for that collection is
stale. Deleting the cache entry forces a rebuild on the next query that needs it.
If the collection was not in the cache yet, `pop(..., None)` is a no-op.

**Return value:** the number of chunks ingested (useful for CLI feedback and API response).

---

## The `query()` Flow, Step by Step

```python
def query(
    self,
    question: str,
    collection: str = "default",
    top_k: int | None = None,
) -> tuple[str, list[dict]]:
```

The method returns a tuple `(answer, chunks)`. The `chunks` list is returned
alongside the answer so callers can display sources to the user.

**Step 1 — Resolve top_k**

```python
k = top_k if top_k is not None else self.top_k
```

Callers can override the default `TOP_K` per request (useful in the API's
`?top_k=10` query parameter).

**Step 2a — Dense-only retrieval (default, `USE_HYBRID=false`)**

```python
chunks = retrieve.retrieve(question, self.embedder, self.chroma_dir, collection, k)
```

Embeds the question, queries Chroma for the top-K nearest chunks by cosine
similarity, returns them ordered by distance.

**Step 2b — Hybrid retrieval (`USE_HYBRID=true`)**

```python
bm25_index = self._get_bm25_index(collection)
chunks = hybrid.hybrid_retrieve(
    question,
    self.embedder,
    bm25_index,
    self.chroma_dir,
    collection,
    k,
    self.fetch_k,
)
```

Retrieves `fetch_k` candidates from both dense (Chroma) and sparse (BM25),
then fuses them using Reciprocal Rank Fusion. Final list is truncated to `k`.
`fetch_k` (default 20) is always larger than `k` (default 5) to give RRF
enough candidates to work with.

**Step 3 — Optional reranking (`USE_RERANKER=true`)**

```python
if self.use_reranker and self.reranker is not None:
    chunks = self.reranker.rerank(question, chunks, k)
```

The CrossEncoder scores each (question, chunk) pair jointly — more accurate
than cosine similarity but much slower. It re-orders the already-retrieved
chunks and picks the top-K. Because reranking is expensive, it runs on the
small retrieved set (size k or fetch_k), not on the entire corpus.

**Step 4 — Generate answer**

```python
answer = self.llm_client.ask(question, chunks)
return answer, chunks
```

Chunks are formatted into a context string and sent to Ollama. The answer and
the raw chunks are both returned.

---

## BM25 Lazy Cache

```python
def _get_bm25_index(self, collection: str) -> bm25.BM25Index:
    if collection not in self._bm25_cache:
        self._bm25_cache[collection] = bm25.build_index(self.chroma_dir, collection)
    return self._bm25_cache[collection]
```

`bm25.build_index` fetches **all documents** from Chroma and builds a
BM25Okapi index in memory. For a large collection this takes a few seconds.
Caching avoids repeating that work on every query.

The cache is per-collection (a `dict[str, BM25Index]`), so you can query
multiple collections without interference.

After ingest, the relevant entry is deleted from `_bm25_cache`. The next query
triggers a fresh rebuild that includes the new documents. This is the
cache-invalidation pattern: rather than updating incrementally, rebuild on demand.

---

## Feature Flags: USE_HYBRID and USE_RERANKER

These two boolean flags let you progressively enable more powerful (but more
expensive) retrieval modes:

| Mode | Flags | Characteristics |
|---|---|---|
| Dense only | both `false` | Fast (~50 ms), good baseline |
| Hybrid | `USE_HYBRID=true` | Better recall for keyword-heavy queries |
| Hybrid + Rerank | both `true` | Highest quality, but slowest (~2–5 s extra) |

Start with both off when learning. Turn on `USE_HYBRID` when you notice the
dense search missing obvious keyword matches. Add `USE_RERANKER` when you want
the most accurate ranking and latency is acceptable.

---

## Flow Diagram

```
ingest(path, collection)
  └── load_directory() / load_file()
        └── _chunk_text()
              └── Chunk(text, source, page, chunk_id)
  └── embedder.embed(batch)          # batched at 512 chunks
        └── np.vstack(all_batches)
  └── store.add_chunks(chunks, embeddings)
  └── _bm25_cache.pop(collection)   # invalidate stale BM25

query(question, collection, top_k)
  └── [USE_HYBRID=false]
        └── retrieve.retrieve(question, embedder, ...)
  └── [USE_HYBRID=true]
        └── _get_bm25_index(collection)   # lazy cache
        └── hybrid.hybrid_retrieve(...)   # RRF fusion
  └── [USE_RERANKER=true]
        └── reranker.rerank(question, chunks, top_k)
  └── llm_client.ask(question, chunks)
  └── return (answer, chunks)
```
