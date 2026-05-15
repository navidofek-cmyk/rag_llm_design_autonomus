# Architecture Decision Records

## ADR-001: Chroma as Vector Store

**Status:** Accepted

**Decision:** Use Chroma in embedded (PersistentClient) mode rather than a separate vector DB service.

**Why:** Zero infrastructure overhead — no Docker, no network, no auth. Chroma's embedded mode writes to disk and survives restarts. For local RAG with < 1M chunks, latency and throughput are more than sufficient.

**Trade-off:** Cannot scale horizontally or share the index across multiple processes simultaneously.

---

## ADR-002: BAAI/bge-small-en for Embeddings

**Status:** Accepted

**Decision:** Use `BAAI/bge-small-en` (384-dim) via sentence-transformers on CPU, not a larger model or API.

**Why:** bge-small-en punches above its weight on MTEB benchmarks. At 384 dimensions, Chroma queries are fast even on CPU. No GPU required, no API key, no egress cost. normalize_embeddings=True ensures cosine similarity works correctly.

**Trade-off:** Lower recall than bge-large or text-embedding-3-large. Mitigated by hybrid retrieval + reranking.

---

## ADR-003: Hybrid Retrieval with RRF

**Status:** Accepted

**Decision:** Combine dense vector search (Chroma) and sparse BM25 via Reciprocal Rank Fusion (k=60).

**Why:** Dense retrieval excels at semantic similarity; BM25 excels at exact keyword matches. RRF is parameter-free and robust — it doesn't require score normalization or tuning. The k=60 default is the standard from the original RRF paper.

**Trade-off:** BM25 index is built by fetching all documents from Chroma, which is memory-intensive for very large collections. Mitigated by lazy per-collection caching in RAGPipeline.

---

## ADR-004: Ollama /api/chat over /api/generate

**Status:** Accepted

**Decision:** Use Ollama's `/api/chat` endpoint with system + user message structure, not the older `/api/generate`.

**Why:** `/api/chat` supports proper system prompts and multi-turn conversations. The system prompt instructs the model to answer only from context and cite sources — this is critical for RAG faithfulness. `/api/generate` would require manual prompt templating that varies per model.

**Trade-off:** Requires Ollama ≥ 0.1.14 which supports `/api/chat`. All recent Ollama releases qualify.
