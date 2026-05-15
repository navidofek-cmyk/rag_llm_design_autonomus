# RAG Project — Learning Guide

## What Is RAG and Why Does It Exist?

Large Language Models (LLMs) are remarkably capable, but they suffer from two fundamental problems that make them unreliable for real-world knowledge tasks:

### The Hallucination Problem

An LLM learns by compressing patterns from its training corpus into billions of numerical weights. When you ask it a question, it generates the most statistically plausible continuation — it does not look anything up. If the answer is not well-represented in the training data, the model will confidently invent a plausible-sounding but wrong answer. This is called a **hallucination**.

For example, ask an LLM about the return value of a specific internal function in your company's codebase, and it will fabricate an answer rather than admit ignorance. It has no other option — it has never seen your codebase.

### The Knowledge Cutoff Problem

Training a large model is expensive and takes months. After training ends, the model's knowledge is frozen. Events, code changes, documentation updates, or any new information that happened after the cutoff date are invisible to it. A model trained through 2024 cannot tell you about a library released in 2025.

### The RAG Solution

**Retrieval-Augmented Generation** solves both problems by separating *retrieval* from *generation*:

1. **Index phase** (done once): Take your documents, split them into chunks, convert each chunk to a numerical vector (embedding), and store everything in a vector database.
2. **Query phase** (done per question): Embed the user's question, find the most similar chunks in the database, and hand those chunks to the LLM as context.

The LLM is now told: "Answer only from what I am showing you. If it is not here, say so." The model no longer needs to remember facts — it reads them from the retrieved chunks in real time. Hallucinations are dramatically reduced, and the knowledge base can be updated by re-indexing documents at any time.

---

## Why Not Just Use a Bigger Context Window?

Modern LLMs support context windows of 128k, 200k, or even 1M tokens. A reasonable question is: why not just paste your entire document collection into the prompt instead of building a retrieval system?

Several reasons make this impractical:

**Cost and latency.** Every token in the context window costs money and takes time to process. A 1M token context costs roughly 100x more per query than a 1k token context. If you have millions of documents, it is physically impossible to fit them all in anyway.

**Attention degradation.** Research shows that LLMs are much better at using information near the beginning and end of a long context than information buried in the middle (the "lost in the middle" problem). Retrieval pre-selects only the relevant chunks, so the model sees a short, focused context.

**Stale data.** If you paste in static documents, you still cannot update them without rebuilding the prompt. A RAG system lets you add and remove documents from the index without changing your prompt or model.

**Retrieval is a feature.** Finding the most relevant 5 chunks out of 100,000 is itself a meaningful task. The retrieval step adds a layer of precision that pure context-stuffing cannot provide.

---

## What You Will Learn From This Project

This project is a complete, production-grade RAG system built step by step. By reading the code and these docs you will understand:

| Module | Concept |
|--------|---------|
| `ingest.py` | Document loading, text chunking, deduplication via MD5 |
| `embed.py` | Text embeddings, sentence transformers, vector normalization |
| `store.py` | Vector databases, HNSW indexing, cosine similarity search |
| `retrieve.py` | Dense retrieval end-to-end |
| `bm25.py` | Sparse (keyword) retrieval, TF-IDF, BM25 scoring |
| `hybrid.py` | Reciprocal Rank Fusion, combining retrieval signals |
| `reranker.py` | Cross-encoder reranking, two-stage retrieval |
| `swarm.py` | Parallel multi-collection search, agent patterns |
| `llm.py` | Talking to a local LLM, grounding prompts |
| `pipeline.py` | Orchestrating it all with configurable feature flags |
| `api.py` | Exposing the pipeline as a REST API |
| `cli.py` | Command-line interface patterns with Typer |

---

## Architecture at a Glance

### Ingest (Build the Index)

```
Documents (PDF, RST, MD, TXT)
         |
         v
    [ingest.py]
    Split into 512-char chunks (64-char overlap)
    Assign MD5 chunk_id for deduplication
         |
         v
    [embed.py]
    SentenceTransformer: text → 384-dim float vector
         |
         v
    [store.py]
    Chroma PersistentClient: save vector + metadata to disk
         |
         v
    data/chroma/  (persisted on disk)
```

### Query (Answer a Question)

```
User question: "How does Python's GIL work?"
         |
         v
    [embed.py]  embed_one(question) → 384-dim query vector
         |
    ┌────┴──────────────────────┐
    │ Dense path                │ Sparse path (if USE_HYBRID)
    │ [store.py] HNSW search    │ [bm25.py] BM25Okapi scoring
    └────────────┬──────────────┘
                 │
         [hybrid.py]  Reciprocal Rank Fusion  (if USE_HYBRID)
                 |
         [reranker.py]  CrossEncoder rerank  (if USE_RERANKER)
                 |
                 v
         Top-K chunks with source metadata
                 |
         [llm.py]  Build prompt: system + "Context: {chunks}" + question
                 |
         Ollama /api/chat  →  answer string
                 |
                 v
         Answer + source citations
```

### Multi-Collection Swarm

```
Question
    |
    ├──► agent-0 searches "python-docs" collection
    ├──► agent-1 searches "peps" collection      } ThreadPoolExecutor
    ├──► agent-2 searches "fastapi" collection
    |
    └──► merge results via RRF → top-K → LLM → answer
```

---

## Concept Deep Dives

- [Embeddings](concepts/embeddings.md) — How text becomes numbers that capture meaning
- [Vector Search](concepts/vector-search.md) — How Chroma finds similar vectors
- [BM25](concepts/bm25.md) — Keyword retrieval, and why it still matters
- [Hybrid + RRF](concepts/hybrid-rrf.md) — Combining dense and sparse retrieval
- [Reranking](concepts/reranking.md) — The CrossEncoder precision boost
- [Swarm Search](concepts/swarm.md) — Parallel multi-collection agents
- [Full Architecture](architecture.md) — End-to-end pipeline walk-through

---

## Quick Start

```bash
# Ingest a folder of documents
uv run rag ingest data/raw/peps --collection peps

# Ask a question
uv run rag ask "What is PEP 8?" --collection peps --sources

# Start the REST API
uv run rag serve

# Launch the web UI
uv run rag ui

# Evaluate on a Q&A set
uv run rag eval tests/eval/qa_set.jsonl --collection peps
```
