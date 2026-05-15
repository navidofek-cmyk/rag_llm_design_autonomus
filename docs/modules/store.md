# store.py — Vector Database with Chroma

**File:** `src/local_rag/store.py`

This module manages all reads and writes to Chroma, the vector database that
stores embeddings alongside their original text and metadata. Every chunk that
gets ingested passes through `add_chunks`. Every query result comes from
`query`. The rest of the system never talks to Chroma directly.

---

## What Is Chroma PersistentClient?

Chroma is an open-source vector database designed for embedding-based search.
It supports three deployment modes:

| Mode | How to use | Persistence |
|---|---|---|
| In-memory (ephemeral) | `chromadb.Client()` | Lost when process exits |
| Persistent (embedded) | `chromadb.PersistentClient(path=...)` | Saved to disk |
| Server mode | `chromadb.HttpClient(host=...)` | Separate Chroma server process |

This project uses **PersistentClient**, which embeds the database in the calling
process and saves data to a local directory. This is the right choice for a
development project because:

- No separate process to manage.
- Data survives restarts (unlike in-memory).
- Simple setup (just a directory path).

```python
client = chromadb.PersistentClient(path=chroma_dir)
```

`chroma_dir` defaults to `./data/chroma` (configured via `CHROMA_DIR` in
`.env`). Chroma creates this directory on first use and writes SQLite + binary
files inside it.

---

## get_or_create_collection (Idempotent)

```python
collection = client.get_or_create_collection(collection_name)
```

This single call either retrieves an existing collection by name or creates a
new empty one if it does not exist. It is **idempotent**: calling it ten times
with the same name has exactly the same effect as calling it once.

The alternative would be a try/except pattern:

```python
# What you would have to write without get_or_create_collection:
try:
    collection = client.get_collection(collection_name)
except Exception:
    collection = client.create_collection(collection_name)
```

`get_or_create_collection` handles this cleanly and is safe to call on every
ingest run.

---

## Why Batching at 5000

```python
BATCH_SIZE = 5000

for start in range(0, len(chunks), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch = chunks[start:end]
    collection.add(
        ids=[c.chunk_id for c in batch],
        embeddings=embeddings[start:end].tolist(),
        documents=[c.text for c in batch],
        metadatas=[{"source": c.source, "page": c.page} for c in batch],
    )
```

Chroma's `collection.add()` has an internal limit: it cannot accept more than
roughly 5000 items in a single call. Exceeding this limit raises a cryptic
error. The batching loop prevents this by feeding data in chunks of at most
5000 at a time.

For a corpus of 50,000 chunks, this means 10 sequential calls. Each call is
typically fast (~100 ms), so the total overhead is acceptable.

If you ingest the same chunks twice, Chroma silently ignores duplicates because
the `chunk_id` (MD5 hash of text) already exists as an ID. This makes re-ingest
safe.

---

## What IDs, Embeddings, Documents, and Metadatas Map To

```python
collection.add(
    ids=[c.chunk_id for c in batch],          # unique identifier per chunk
    embeddings=embeddings[start:end].tolist(),  # float32 vectors (384-dim for bge-small-en)
    documents=[c.text for c in batch],         # raw text, returned in query results
    metadatas=[{"source": c.source, "page": c.page} for c in batch],  # arbitrary JSON
)
```

| Parameter | Type | Purpose |
|---|---|---|
| `ids` | `list[str]` | Unique identifiers. Chroma rejects duplicate IDs. |
| `embeddings` | `list[list[float]]` | The actual vectors used for similarity search. |
| `documents` | `list[str]` | The text to return when a chunk is retrieved. |
| `metadatas` | `list[dict]` | Arbitrary JSON attached to each chunk. |

Note that `embeddings` must be converted from NumPy arrays to plain Python lists
(`.tolist()`) because the Chroma Python client does not accept NumPy arrays
directly.

`metadatas` accepts any JSON-serializable dict. This project stores only
`source` and `page`, but you could add anything: author, date, document title,
language, section heading.

---

## How `query()` Returns Results

```python
def query(
    embedding: np.ndarray,
    chroma_dir: str,
    collection_name: str,
    top_k: int = 5,
) -> list[dict]:
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(collection_name)
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
```

`query_embeddings` takes a list of query vectors (one per query). We always
send exactly one (`[embedding.tolist()]`), but the API supports batch queries.

`include=["documents", "metadatas", "distances"]` controls what Chroma returns:

- `"documents"` — the stored text for each result chunk
- `"metadatas"` — the `{"source": ..., "page": ...}` dicts
- `"distances"` — cosine distance to the query vector (lower = more similar)
- `"embeddings"` — the stored embedding vectors (we omit this to save bandwidth)

---

## The Nested List Unwrapping

```python
docs = results["documents"][0]
metas = results["metadatas"][0]
dists = results["distances"][0]
```

Because Chroma supports batch queries (multiple `query_embeddings` at once),
its response is always a list-of-lists. With a single query, the outer list
has exactly one element:

```
results["documents"] == [["chunk text 1", "chunk text 2", ...]]
results["documents"][0] == ["chunk text 1", "chunk text 2", ...]
```

Indexing `[0]` unwraps the outer list to get the results for our single query.
This is a common Chroma gotcha — forgetting the `[0]` gives you a list
containing a list instead of a flat list of strings.

After unwrapping, the three lists are zipped together into a list of dicts:

```python
return [
    {
        "text": doc,
        "source": meta["source"],
        "page": meta["page"],
        "distance": dist,
    }
    for doc, meta, dist in zip(docs, metas, dists)
]
```

This uniform dict format is what flows through `retrieve.py`, `hybrid.py`,
`reranker.py`, `llm.py`, and ultimately appears in API responses and UI output.

---

## Utility Functions

### `list_collections(chroma_dir)`

```python
def list_collections(chroma_dir: str) -> list[str]:
    client = chromadb.PersistentClient(path=chroma_dir)
    return [c.name for c in client.list_collections()]
```

Returns the names of all collections in the database. Used by `cli.py`'s
`inspect` command and `api.py`'s `/collections` endpoint.

### `count(chroma_dir, collection_name)`

```python
def count(chroma_dir: str, collection_name: str) -> int:
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(collection_name)
    return collection.count()
```

Returns the number of chunks stored in a collection. Useful for diagnostics:

```bash
uv run rag inspect peps
# Collection 'peps': 1847 chunks
```

---

## Summary

| Concept | Implementation | Why |
|---|---|---|
| Persistence | `PersistentClient(path=chroma_dir)` | Data survives process restarts |
| Idempotent collection creation | `get_or_create_collection` | Safe to call on every ingest |
| Batch size limit | `BATCH_SIZE = 5000` | Chroma's hard per-call limit |
| IDs | MD5 hash of text | Content-addressable, enables dedup |
| Embeddings | NumPy array converted to `.tolist()` | Chroma requires plain Python lists |
| Metadata | `{"source": path, "page": N}` | Provenance for citation |
| Result format | Flat `list[dict]` | Uniform interface consumed by all downstream modules |
| Nested list unwrap | `results["documents"][0]` | Chroma batch API always wraps in outer list |
