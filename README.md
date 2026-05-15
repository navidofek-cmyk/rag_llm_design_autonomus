# Local RAG System

A production-ready Retrieval-Augmented Generation system with hybrid retrieval, cross-encoder reranking, and swarm search across multiple collections.

## Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `BAAI/bge-small-en` (CPU, sentence-transformers) |
| Vector DB | Chroma (embedded, persistent) |
| BM25 | rank-bm25 |
| Reranker | `BAAI/bge-reranker-base` (CrossEncoder) |
| LLM | Ollama `/api/chat` |
| API | FastAPI + uvicorn |
| UI | Gradio |
| CLI | Typer |

## Quick Start

```bash
# Install
uv sync
uv pip install -e .

# Configure
cp .env.example .env
# Edit .env: set OLLAMA_HOST, LLM_MODEL

# Ingest documents
uv run rag ingest data/raw/peps --collection peps

# Ask a question
uv run rag ask "What is PEP 8?" --collection peps

# Start API server
uv run rag serve

# Launch UI
uv run rag ui
```

## Architecture

```
src/local_rag/
  ingest.py     — PDF/RST/MD/TXT loader, fixed-size chunking (512 chars, 64 overlap)
  embed.py      — SentenceTransformer wrapper (BAAI/bge-small-en)
  store.py      — Chroma PersistentClient, batch upsert at 5000 chunks
  retrieve.py   — Dense vector retrieval
  bm25.py       — BM25Okapi sparse retrieval with lazy cache
  hybrid.py     — Reciprocal Rank Fusion (RRF) of dense + BM25
  reranker.py   — CrossEncoder reranker (BAAI/bge-reranker-base)
  swarm.py      — Parallel search across multiple collections, merged via RRF
  llm.py        — Ollama /api/chat client with structured prompting
  pipeline.py   — Orchestration: ingest → embed → store → retrieve → rerank → LLM
  api.py        — FastAPI REST endpoints
  ui.py         — Gradio Blocks interface
  cli.py        — Typer CLI (7 commands)
```

## CLI Commands

```
rag ingest <path>              Ingest files or directory
rag ask <question>             Ask a question
rag eval <qa_file>             Evaluate on Q&A JSONL set
rag inspect <collection>       Show collection stats and sample chunks
rag batch-ask <in> <out>       Batch process questions
rag serve                      Start FastAPI on :8000
rag ui                         Launch Gradio UI on 0.0.0.0:7860
```

## Configuration

All settings via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `LLM_MODEL` | `qwen3-coder:480b-cloud` | Model to use |
| `EMBED_MODEL` | `BAAI/bge-small-en` | Embedding model |
| `CHROMA_DIR` | `./data/chroma` | Chroma persistence path |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | Chunks to retrieve |
| `FETCH_K` | `20` | Chunks fetched before reranking |
| `USE_HYBRID` | `false` | Enable hybrid retrieval |
| `USE_RERANKER` | `false` | Enable CrossEncoder reranking |

## Testing

```bash
uv run pytest tests/unit -q    # 77 tests
uv run ruff check src/
uv run mypy src/local_rag
```
