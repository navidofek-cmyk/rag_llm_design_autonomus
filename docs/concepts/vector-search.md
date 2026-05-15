# Vector Search and Chroma

## What a Vector Database Does

A vector database stores embeddings (numerical vectors) alongside their source documents and metadata, and provides one critical operation: **nearest-neighbor search** — "given this query vector, find the N most similar vectors I have stored."

This is a fundamentally different operation from a traditional SQL or keyword search. There is no `WHERE text LIKE '%python%'`. Instead, you ask: "which stored vectors point in approximately the same direction as my query vector?"

Think of it like this: instead of searching a library's card catalog by title, you describe a book's *subject matter* to a librarian who has read every book and can immediately point you to the shelf where the most thematically similar books live.

---

## How Chroma Works Internally

Chroma is an open-source, embedded vector database. "Embedded" here means it runs in the same process as your Python code — there is no separate server to start, no network connection to manage. It persists data to disk using SQLite and a custom index format.

### HNSW — Hierarchical Navigable Small World

Chroma uses the **HNSW** algorithm for its vector index. Understanding HNSW fully requires graph theory, but the intuition is straightforward:

Imagine you are trying to find the closest city to a given GPS coordinate on a world map. A naive approach scans every city on earth (millions). HNSW instead builds a multi-layer graph:

- **Top layer**: A handful of widely-spaced "landmark" nodes — one per continent. You enter here and immediately jump close to the right region.
- **Middle layers**: Progressively denser graphs — one per country, then per province.
- **Bottom layer**: Every single city. You do the final precise search here.

At query time, you start at the top, greedily move toward the closest node at each layer, then descend. You inspect only a tiny fraction of all nodes but consistently find the nearest neighbors. HNSW trades a small amount of accuracy (it is *approximate* nearest-neighbor, or ANN) for massive speed gains: searching 1 million vectors in milliseconds instead of seconds.

---

## Cosine Similarity vs L2 Distance

There are two common ways to measure "closeness" between vectors:

### L2 (Euclidean) Distance

```
L2(A, B) = sqrt( sum( (A_i - B_i)^2 ) )
```

This is straight-line distance in N-dimensional space. **Lower is closer.**

Problem: L2 is sensitive to vector magnitude. A long document that has a large embedding magnitude will appear far from a short query even if they are about the same topic.

### Cosine Similarity

```
cosine(A, B) = (A · B) / (|A| × |B|)
```

This measures the **angle** between two vectors, ignoring magnitude. Two vectors pointing in the same direction have cosine similarity = 1.0 (angle = 0°). Perpendicular vectors have cosine = 0.0. Opposite directions have cosine = -1.0.

For semantic search, cosine similarity is almost always the right choice because it captures *directional* agreement — two texts about the same topic will point in similar directions regardless of length.

> **Key insight:** When embeddings are L2-normalized (unit length), cosine similarity and dot product give identical results, and L2 distance is monotonically related to cosine similarity: `L2(A,B)^2 = 2 - 2*cosine(A,B)`. Chroma reports distances in L2-squared by default, so a distance of 0.0 means identical vectors, and a distance near 2.0 means opposite vectors.

---

## What "Distance" Means in Query Results

When `store.query()` returns results, each chunk has a `"distance"` field. Confusingly, Chroma returns squared L2 distances even when using cosine similarity as the similarity measure. The mapping to familiar cosine similarity is:

```
cosine_similarity = 1 - (distance / 2)

distance = 0.0   →  cosine = 1.0  (identical)
distance = 0.5   →  cosine = 0.75 (very similar)
distance = 1.0   →  cosine = 0.5  (moderately related)
distance = 2.0   →  cosine = 0.0  (unrelated)
```

In practice, good RAG retrieval returns distances in the 0.1–0.6 range. If all your distances are above 1.5, your query is semantically unrelated to your document collection.

---

## Why We Batch at 5,000 Chunks

From `src/local_rag/store.py`:

```python
BATCH_SIZE = 5000

def add_chunks(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    chroma_dir: str,
    collection_name: str,
) -> None:
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(collection_name)

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

Chroma's Python client passes data through gRPC or SQLite calls that have internal buffer limits. Sending all vectors in a single `collection.add()` call when you have tens of thousands of chunks can exceed those limits and raise cryptic errors. Batching at 5,000 keeps each call well within Chroma's comfort zone while still being efficient (5,000 is large enough that the overhead of multiple calls is negligible).

The `embeddings[start:end].tolist()` converts a NumPy slice to a plain Python list of lists, which is what Chroma's API expects.

---

## What Gets Stored in Each Record

Each chunk stored in Chroma has four components:

| Field | Type | Contents |
|-------|------|----------|
| `id` | string | MD5 hash of chunk text — guarantees deduplication |
| `embedding` | list[float] | 384-dimensional normalized vector |
| `document` | string | The raw text of the chunk |
| `metadata` | dict | `{"source": "/path/to/file.pdf", "page": 3}` |

The ID-based deduplication is a subtle but important feature. If you run `rag ingest` on the same file twice, Chroma sees that the IDs already exist and skips them rather than creating duplicates. This makes the ingest operation **idempotent** — safe to run multiple times.

---

## Querying: Full Example

From `src/local_rag/store.py`:

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
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
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

Note `query_embeddings=[embedding.tolist()]` — Chroma expects a **list of query vectors** (batch query), so we wrap the single embedding in an outer list. The result is then `results["documents"][0]` — the `[0]` extracts results for our first (and only) query vector.

---

## Collections: Logical Namespaces

Chroma organizes vectors into **collections** — think of them as separate tables in a database. Each collection has its own HNSW index, so querying one collection does not touch another. In this project, the default collections are:

- `default` — general purpose
- `python-docs` — official Python documentation
- `peps` — Python Enhancement Proposals
- `fastapi` — FastAPI documentation

This separation means you can ask "What does PEP 8 say about line length?" specifically against the `peps` collection without getting noise from unrelated documents. The swarm pattern (see [swarm.md](swarm.md)) takes this further by searching multiple collections in parallel.

---

## Persistent Storage Layout

```
data/chroma/
├── chroma.sqlite3          ← metadata, collection definitions, document texts
└── <uuid>/
    ├── data_level0.bin     ← HNSW graph bottom layer
    ├── header.bin          ← HNSW index metadata
    ├── length.bin
    └── link_lists.bin      ← HNSW graph upper layers
```

The `PersistentClient(path=chroma_dir)` call opens this directory and memory-maps the index files. Changes are written to disk automatically. The SQLite file stores the document text and metadata; the `.bin` files store the vector index itself.
