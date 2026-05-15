# api.py — REST API with FastAPI

**File:** `src/local_rag/api.py`

This module exposes the RAG pipeline over HTTP so any program — a browser,
another service, a shell script — can query it without importing Python code.
FastAPI is used because it generates interactive documentation automatically,
validates request and response types via Pydantic, and is fast enough for local
use without any configuration.

---

## FastAPI Overview

FastAPI converts Python function signatures into HTTP endpoints. When the server
starts, it reads your type annotations and generates:

- **JSON Schema** for all request and response models
- **Swagger UI** at `http://localhost:8000/docs` (try endpoints interactively)
- **ReDoc** at `http://localhost:8000/redoc` (cleaner documentation view)

```python
app = FastAPI(title="Local RAG API")
```

Start the server with:

```bash
uv run rag serve
# or directly:
uvicorn local_rag.api:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs` in your browser to see the full API.

---

## Pipeline Singleton via `lru_cache`

```python
@functools.lru_cache(maxsize=1)
def get_pipeline():
    from local_rag.pipeline import RAGPipeline
    return RAGPipeline()
```

`lru_cache(maxsize=1)` turns `get_pipeline()` into a function that runs once
and caches the result forever. Every subsequent call returns the same
`RAGPipeline` instance.

Why this matters:

- `RAGPipeline.__init__` loads the embedding model (~2 seconds, ~50 MB).
- If `USE_RERANKER=true`, it also loads the CrossEncoder (~3 seconds, ~300 MB).
- Loading these on every HTTP request would make the API unusably slow.
- A singleton loads once at first request and is reused for the lifetime of
  the process.

The import is inside the function (rather than at module level) to avoid loading
the full pipeline when `api.py` is merely imported during testing.

---

## Pydantic Models

Pydantic models define the shape of JSON request and response bodies.
FastAPI validates all incoming and outgoing data against these models
automatically — bad input returns a clear 422 error with field-level details.

```python
class IngestRequest(BaseModel):
    path: str            # Filesystem path to file or directory
    collection: str = "default"   # Optional, defaults to "default"


class AskResponse(BaseModel):
    answer: str          # The LLM's answer text
    sources: list[dict]  # The retrieved chunks (text, source, page, distance)
    collection: str      # Which collection was queried
```

---

## Endpoints

### GET /health

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

Returns immediately without touching the pipeline or Chroma. Used by:

- **Load balancers** to check if the service is alive before routing traffic.
- **Monitoring** tools (`curl http://localhost:8000/health`) to verify the
  server started successfully.
- **Docker health checks** (`HEALTHCHECK CMD curl -f http://localhost:8000/health`).

Always fast (~1 ms), always `{"status": "ok"}`.

### GET /ask

```python
@app.get("/ask")
def ask(q: str, collection: str = "default", top_k: int = 5) -> AskResponse:
    pipeline = get_pipeline()
    answer, chunks = pipeline.query(q, collection, top_k)
    return AskResponse(answer=answer, sources=chunks, collection=collection)
```

This is the main endpoint. Parameters come from the query string:

- `q` — the question (required)
- `collection` — which Chroma collection to search (optional, default `"default"`)
- `top_k` — how many chunks to retrieve (optional, default `5`)

FastAPI automatically converts the `AskResponse` dataclass to JSON.

**curl example:**

```bash
curl "http://localhost:8000/ask?q=What+is+PEP+8&collection=peps&top_k=3"
```

**Response:**

```json
{
  "answer": "PEP 8 is the style guide for Python code...",
  "sources": [
    {
      "text": "PEP 8 -- Style Guide for Python Code...",
      "source": "/home/user/data/raw/peps/pep-0008.rst",
      "page": 1,
      "distance": 0.12
    }
  ],
  "collection": "peps"
}
```

### GET /collections

```python
@app.get("/collections")
def list_collections() -> list[str]:
    chroma_dir = os.getenv("CHROMA_DIR", "./data/chroma")
    client = chromadb.PersistentClient(path=chroma_dir)
    return [c.name for c in client.list_collections()]
```

Lists all collections that exist in the Chroma database. Useful for:

- Discovering what data has been ingested.
- Populating a UI dropdown of available collections.

This endpoint creates a fresh Chroma client rather than going through the
pipeline singleton, so it works even if the pipeline has not been initialized yet.

**curl example:**

```bash
curl http://localhost:8000/collections
# Response: ["default", "peps", "python-docs"]
```

### POST /ingest

```python
@app.post("/ingest")
def ingest_documents(req: IngestRequest) -> dict:
    pipeline = get_pipeline()
    n = pipeline.ingest(req.path, req.collection)
    return {"chunks": n}
```

Ingests a file or directory into the specified collection. The request body
must be JSON.

**curl example:**

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/home/user/data/raw/peps", "collection": "peps"}'
```

**Response:**

```json
{"chunks": 1847}
```

Note that `path` must be a filesystem path accessible to the *server process*,
not the client. If the server runs on a different machine, the path must be valid
on that machine.

---

## Testing All Endpoints with curl

**Health check:**
```bash
curl http://localhost:8000/health
```

**List collections:**
```bash
curl http://localhost:8000/collections
```

**Ask a question:**
```bash
curl "http://localhost:8000/ask?q=What+is+a+Python+decorator&collection=python-docs"
```

**Ask with more sources:**
```bash
curl "http://localhost:8000/ask?q=What+is+PEP+8&collection=peps&top_k=10"
```

**Ingest a directory:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "./data/raw/peps", "collection": "peps"}'
```

**Ingest a single file:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "./data/raw/peps/pep-0008.rst", "collection": "pep8-only"}'
```

---

## Interactive Documentation

Once the server is running, the full API is explorable without curl:

- `http://localhost:8000/docs` — Swagger UI with try-it-out buttons
- `http://localhost:8000/redoc` — ReDoc, better for reading
- `http://localhost:8000/openapi.json` — raw OpenAPI schema (useful for
  generating client code in other languages)

---

## Architecture Note: Why GET for /ask?

A question ("What is PEP 8?") is semantically a read operation with no
side effects, which makes GET appropriate. GET requests are also easier to
test from a browser address bar and easier to bookmark. The query string
handles short questions fine, though for very long questions (>2000 chars)
you would want to switch to POST with a JSON body.
