# Quickstart Guide

This guide takes you from a fresh checkout to a working RAG system in about
10 minutes (plus model download time). Follow the steps in order.

---

## Prerequisites

- Python 3.12 (check with `python --version`)
- `uv` package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- [Ollama](https://ollama.com/) installed and in your PATH

---

## Step 1: Install Python Dependencies

```bash
cd /path/to/rag_llm_design_autonomus

# Install all dependencies (including dev tools: pytest, ruff, mypy)
uv sync

# Install the project itself as an editable package
# (makes "uv run rag" work and allows "from local_rag import ..." in tests)
uv pip install -e .
```

Verify the install worked:

```bash
uv run rag --help
```

You should see 7 commands listed: `ingest`, `ask`, `eval`, `inspect`,
`batch-ask`, `serve`, `ui`.

---

## Step 2: Configure .env

```bash
# Copy the example config
cp .env.example .env
```

Open `.env` and verify (or edit) these two settings:

```dotenv
# Where Ollama is listening — default is fine if running locally
OLLAMA_HOST=http://localhost:11434

# The model to use for answers — must be pulled with "ollama pull"
LLM_MODEL=llama3.2
```

All other defaults are fine for getting started:

```dotenv
EMBED_MODEL=BAAI/bge-small-en   # embedding model (auto-downloaded)
EMBED_DEVICE=cpu                  # change to "cuda" if you have a GPU
CHROMA_DIR=./data/chroma          # where vector database is stored
CHUNK_SIZE=512                    # characters per chunk
CHUNK_OVERLAP=64                  # overlap between adjacent chunks
TOP_K=5                           # chunks returned per query
USE_HYBRID=false                  # enable BM25 + dense fusion
USE_RERANKER=false                # enable CrossEncoder reranking
```

---

## Step 3: Install Ollama and Pull a Model

If Ollama is not yet installed:

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama server and pull a model:

```bash
# Start Ollama (runs in background)
ollama serve &

# Pull a model — llama3.2 is a good starting point (~2 GB)
ollama pull llama3.2

# Verify it works
ollama list
```

If you have more disk space and want a more capable model:

```bash
ollama pull qwen2.5:7b      # 7B params, better reasoning (~4.5 GB)
ollama pull mistral:7b      # alternative 7B model
```

Set `LLM_MODEL` in `.env` to match the model you pulled.

---

## Step 4: Download Sample Data (Python PEPs)

PEPs (Python Enhancement Proposals) are RST files — perfect for testing because
they are publicly available and cover well-known topics.

```bash
# Clone the PEPs repository (shallow clone for speed)
git clone --depth 1 https://github.com/python/peps.git /tmp/peps

# Copy some interesting PEPs into the project's data directory
mkdir -p data/raw/peps
cp /tmp/peps/peps/pep-0008.rst data/raw/peps/   # Style Guide
cp /tmp/peps/peps/pep-0020.rst data/raw/peps/   # The Zen of Python
cp /tmp/peps/peps/pep-0257.rst data/raw/peps/   # Docstring Conventions
cp /tmp/peps/peps/pep-0484.rst data/raw/peps/   # Type Hints
cp /tmp/peps/peps/pep-0526.rst data/raw/peps/   # Variable Annotations
```

Or copy all PEPs if you want a full corpus (1000+ files, may take a few minutes
to ingest):

```bash
cp /tmp/peps/peps/*.rst data/raw/peps/
```

---

## Step 5: Ingest Documents

```bash
uv run rag ingest data/raw/peps --collection peps
```

Expected output:

```
Ingested 847 chunks into collection 'peps'
```

The first run downloads the `BAAI/bge-small-en` embedding model (~130 MB) from
Hugging Face and caches it in `~/.cache/huggingface/`. Subsequent runs skip
the download.

Verify the ingest:

```bash
uv run rag inspect peps
# Collection 'peps': 847 chunks
# [1] data/raw/peps/pep-0008.rst p.1: PEP 8 – Style Guide for Python Code...
```

---

## Step 6: Ask Your First Question

```bash
uv run rag ask "What is PEP 8?" --collection peps
```

Expected output (answer will vary by model):

```
PEP 8 is the style guide for Python code. It provides coding conventions
for the Python standard library, covering topics like indentation (4 spaces),
line length (79 characters), naming conventions, and more.
```

Show which chunks the answer came from:

```bash
uv run rag ask "What is PEP 8?" --collection peps --sources
```

```
PEP 8 is the style guide for Python code...

--- Sources ---
[1] data/raw/peps/pep-0008.rst p.1
[2] data/raw/peps/pep-0008.rst p.1
[3] data/raw/peps/pep-0257.rst p.1
```

Ask another question:

```bash
uv run rag ask "What does the Zen of Python say about readability?" --collection peps --sources
```

---

## Step 7: Start the REST API

```bash
uv run rag serve
# Server starts at http://0.0.0.0:8000
```

In another terminal, test it:

```bash
# Health check
curl http://localhost:8000/health
# {"status":"ok"}

# Ask a question
curl "http://localhost:8000/ask?q=What+is+PEP+8&collection=peps"

# List collections
curl http://localhost:8000/collections
# ["peps"]

# Ingest more documents via API
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "data/raw/peps/pep-0020.rst", "collection": "peps"}'
```

Open `http://localhost:8000/docs` to explore the API interactively in your
browser (no curl needed).

Stop the server with `Ctrl+C`.

---

## Step 8: Launch the Gradio UI

```bash
uv run rag ui
```

Open `http://localhost:7860` in your browser. You will see:

- A text box for your question
- A dropdown to select the collection (`peps`, `default`, etc.)
- A dropdown to select the Ollama model
- A slider for Top K (how many chunks to retrieve)
- A checkbox to show source chunks
- Answer and Sources output boxes

The UI is served on `0.0.0.0` so it is accessible from other devices on your
local network at `http://<your-ip>:7860`.

---

## Step 9: Enable Hybrid Retrieval

Dense (embedding) search is great for semantic similarity, but poor at exact
keyword matching. Hybrid search combines dense + BM25 (keyword) retrieval using
Reciprocal Rank Fusion.

```bash
# Run a single query with hybrid enabled (without changing .env)
USE_HYBRID=true uv run rag ask "PEP 8 naming convention" --collection peps

# Or enable it permanently in .env:
# USE_HYBRID=true
```

Hybrid retrieval is especially noticeable for queries that contain specific
identifiers, version numbers, or rare technical terms that embeddings may not
capture well.

To understand when hybrid helps, compare results:

```bash
# Dense only (default)
uv run rag ask "snake_case" --collection peps --sources

# Hybrid
USE_HYBRID=true uv run rag ask "snake_case" --collection peps --sources
```

---

## Step 10: Enable the Reranker

The CrossEncoder reranker scores each (question, chunk) pair jointly — more
accurate than cosine similarity but takes 1–5 extra seconds per query.

```bash
# Enable reranker for one query
USE_RERANKER=true uv run rag ask "What is PEP 8?" --collection peps --sources

# Enable both hybrid and reranker (highest quality, slowest)
USE_HYBRID=true USE_RERANKER=true uv run rag ask "What is PEP 8?" --collection peps
```

The reranker downloads `BAAI/bge-reranker-base` (~280 MB) from Hugging Face on
first use.

The reranker re-orders the retrieved chunks. You may see different source
rankings compared to dense-only or hybrid retrieval, especially when the top
chunk from embedding search is topically related but not the most directly
relevant to the specific question.

---

## Common Issues

**"Connection refused" when asking a question:**
Ollama is not running. Start it with `ollama serve`.

**"Collection not found" error:**
You have not ingested documents yet, or used a different collection name.
Run `uv run rag inspect <collection>` to check.

**Slow first query:**
The embedding model is being downloaded or the Chroma index is being built.
Subsequent queries are much faster.

**Out of memory during ingest:**
Reduce the batch size: `embed_batch` in `pipeline.ingest()` controls how many
chunks are embedded at once. The default of 512 is conservative; on a machine
with very little RAM you can reduce it further.

**Empty answer / "I don't know based on the provided context":**
The retrieved chunks do not contain the answer. Try:
1. `--sources` to see what was retrieved
2. A more specific question
3. `USE_HYBRID=true` to improve recall
4. Ingesting more documents
