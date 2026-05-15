# Learning Log

## 2026-05-15 — Bootstrap Complete

### What was built
Complete RAG system from scratch using parallel agent waves:
- Wave A (parallel): ingest.py, embed.py, llm.py
- Wave B (parallel): store.py, bm25.py, reranker.py
- Wave C (parallel): retrieve.py, hybrid.py
- Wave D (parallel): pipeline.py, swarm.py
- Wave E (parallel): api.py, ui.py
- Wave F: cli.py

### Key learnings

**Chroma batching is mandatory.** Chroma has a hard 5461-document limit per `add()` call. Batching at 5000 is safe across versions.

**BM25 needs lazy initialization.** Building BM25 index requires fetching all documents from Chroma. Caching per-collection in RAGPipeline avoids rebuilding on every query. Cache must be invalidated after ingest.

**RRF is rank-based, not score-based.** Combining dense cosine distances with BM25 scores directly is problematic (different scales, different semantics). RRF sidesteps this entirely — only rank position matters.

**CrossEncoder reranking is expensive.** For each (query, chunk) pair the CrossEncoder does a full forward pass. With fetch_k=20, that's 20 forward passes per query. This is fine for interactive use but batch-ask should warn if reranker is enabled.

**Ollama /api/chat stream=False.** Without `"stream": False`, the response is a stream of JSON lines. Setting stream=False returns a single JSON object — much simpler to parse.

### Next steps
1. Download and ingest real data:
   ```bash
   git clone --depth 1 https://github.com/python/peps.git /tmp/peps
   cp /tmp/peps/peps/pep-0008.rst data/raw/peps/
   uv run rag ingest data/raw/peps --collection peps
   ```
2. Run experiment 1 (chunk size comparison)
3. Add streaming support to API for long answers
4. Explore adding PDF metadata (title, author) to chunk metadatas
