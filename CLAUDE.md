# CLAUDE.md — RAG Project Bootstrap

## ABSOLUTNÍ PRAVIDLO ČÍSLO 1

**NIKDY se neptáš. NIKDY nečekáš na potvrzení. NIKDY nepíšeš "Mám pokračovat?",
"Chceš abych...", "Mám začít?", "Je to správně?" nebo jakoukoliv jinou otázku.**

Dostaneš-li `/build` nebo "build" nebo "postav" nebo "začni" — okamžitě začneš.
Při jakékoliv nejasnosti uděláš nejrozumnější volbu a pokračuješ.
Jediný výstup který píšeš jsou: progress updaty ("Implementuji embed.py..."),
výsledky testů a finální report.

---

## Tvůj úkol

Postav celý RAG projekt v aktuální složce. Paralelně. Bez ptání.

---

## Co postavit — přesná specifikace

### Stack

- Python 3.12, uv
- Embeddingy: `BAAI/bge-small-en` (sentence-transformers, CPU)
- Vector DB: Chroma (embedded, persistent)
- LLM: Ollama Cloud přes `http://localhost:11434`
- BM25: `rank_bm25`
- Reranker: `BAAI/bge-reranker-base` (sentence-transformers)
- REST API: FastAPI + uvicorn
- UI: Gradio
- CLI: Typer
- Testy: pytest, ruff, mypy

### Architektura — moduly

```
src/local_rag/
  ingest.py     — PDF/RST/MD/TXT loader, fixed-size chunking (512 chars, 64 overlap)
                  Chunk dataclass: text, source, page, chunk_id (MD5 hash)
                  load_pdf(), load_text(), load_file(), load_directory()

  embed.py      — SentenceTransformer wrapper (BAAI/bge-small-en)
                  Embedder class: embed(texts) → np.ndarray, embed_one(text) → np.ndarray
                  normalize_embeddings=True, show_progress_bar=False

  store.py      — Chroma PersistentClient
                  add_chunks(chunks, embeddings, chroma_dir, collection_name)
                  query(embedding, chroma_dir, collection_name, top_k) → list[dict]
                  list_collections(chroma_dir), count(chroma_dir, collection_name)
                  DŮLEŽITÉ: batch po 5000 kusech (Chroma limit!)

  retrieve.py   — Dense retrieval
                  retrieve(question, embedder, chroma_dir, collection, top_k) → list[dict]

  bm25.py       — BM25Okapi index
                  BM25Index dataclass, build_index(chroma_dir, collection) → BM25Index
                  search(index, query, top_k) → list[dict]
                  Lazy cache v pipeline, prázdná collection → index s _index=None

  hybrid.py     — RRF fusion
                  reciprocal_rank_fusion(dense, bm25, k=60) → list[dict]
                  hybrid_retrieve(question, embedder, bm25_index, chroma_dir, collection, top_k, fetch_k=20)

  reranker.py   — CrossEncoder
                  Reranker class: rerank(question, chunks, top_k) → list[dict]
                  Přidá "rerank_score" ke každému chunku

  swarm.py      — Swarm pattern: více agentů hledá nezávisle bez koordinátora
                  swarm_search(question, embedder, chroma_dir, collections, top_k) → list[dict]
                  Každá collection = jeden nezávislý agent (Task worker)
                  Výsledky sloučeny přes RRF fusion bez centrálního koordinátora
                  swarm_ask(question, collections, top_k) → (answer, chunks)
                  Přidá "swarm_collection" a "agent_id" ke každému chunku

  llm.py        — Ollama /api/chat (NE /api/generate)
                  SYSTEM_PROMPT: odpovídej jen z kontextu, cituj zdroj, odmítni pokud nevíš
                  USER_TEMPLATE: Context: {context}\n\nQuestion: {question}
                  OllamaClient: chat(system, user) → str, ask(question, chunks) → str

  pipeline.py   — Orchestrace, čte .env
                  RAGPipeline: ingest(path, collection), query(question, collection, top_k)
                  USE_HYBRID, USE_RERANKER přepínače
                  BM25 lazy cache (_bm25_cache dict)
                  Po ingestu invaliduj BM25 cache

  api.py        — FastAPI
                  GET /health → {"status": "ok"}
                  GET /ask?q=...&collection=...&top_k=5 → AskResponse
                  GET /collections → list[str]
                  POST /ingest {path, collection} → {"chunks": N}
                  Pipeline singleton přes functools.lru_cache

  ui.py         — Gradio Blocks
                  COLLECTIONS = ["default", "python-docs", "peps", "fastapi"]
                  Načítá cloud modely z `ollama list` (jen ty s "-" jako size)
                  query_fn(question, collection, model, top_k, show_sources) → (answer, sources)
                  Naslouchá na 0.0.0.0 (dostupné ze sítě)

  cli.py        — Typer, entry point: rag = "local_rag.cli:app"
                  Commands: ingest, ask, eval, inspect, batch-ask, serve, ui
```

### pyproject.toml dependencies

```toml
dependencies = [
    "chromadb>=0.5",
    "sentence-transformers>=3.0",
    "sentence-transformers[cross-encoders]>=3.0",
    "pymupdf>=1.24",
    "typer>=0.12",
    "python-dotenv>=1.0",
    "httpx>=0.27",
    "numpy>=1.26",
    "rank-bm25>=0.2",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "gradio>=4.0",
]

[project.scripts]
rag = "local_rag.cli:app"

[tool.pytest.ini_options]
pythonpath = ["src"]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "ruff>=0.5",
    "mypy>=1.10",
    "pytest-mock>=3.14",
]
```

### .env

```dotenv
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=qwen3-coder:480b-cloud
EMBED_MODEL=BAAI/bge-small-en
EMBED_DEVICE=cpu
DATA_DIR=./data
CHROMA_DIR=./data/chroma
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=5
FETCH_K=20
USE_HYBRID=false
USE_RERANKER=false
RERANKER_MODEL=BAAI/bge-reranker-base
```

### Testy — min. počty

| Soubor | Min. testů |
|--------|-----------|
| test_ingest.py | 8 |
| test_embed.py | 4 |
| test_store.py | 5 |
| test_retrieve.py | 5 |
| test_llm.py | 6 |
| test_pipeline.py | 5 |
| test_bm25.py | 5 |
| test_hybrid.py | 5 |
| test_reranker.py | 5 |
| test_api.py | 6 |
| test_ui.py | 4 |

Testy používají mocks — žádný network, žádný real embedding model.
Store testy používají tmp_path s real Chroma (embedded).

---

## Jak postavit — algoritmus pro agenta

### Krok 1 — inicializace (sekvenční)

```bash
uv init --no-readme
# uprav pyproject.toml (přidej dependencies, scripts, pytest config)
uv sync
uv pip install -e .
cp .env.example .env   # nebo vytvoř .env přímo
mkdir -p src/local_rag tests/unit tests/eval data/raw/python-docs data/raw/peps data/raw/libraries docs scripts configs
git init
```

### Krok 2 — paralelní implementace (3 workery najednou)

**Wave A** (nezávislé — pošli v JEDNÉ zprávě jako 3 Task calls):
- Worker 1: `ingest.py` + `test_ingest.py`
- Worker 2: `embed.py` + `test_embed.py`
- Worker 3: `llm.py` + `test_llm.py`

Počkej na všechny tři.

**Wave B** (závisí na Wave A — pošli v JEDNÉ zprávě jako 3 Task calls):
- Worker 4: `store.py` + `test_store.py`
- Worker 5: `bm25.py` + `test_bm25.py`
- Worker 6: `reranker.py` + `test_reranker.py`

Počkej na všechny tři.

**Wave C** (závisí na Wave A+B — pošli v JEDNÉ zprávě jako 2 Task calls):
- Worker 7: `retrieve.py` + `test_retrieve.py`
- Worker 8: `hybrid.py` + `test_hybrid.py`

Počkej na oba.

**Wave D** (závisí na vše — sekvenční):
- Worker 9: `pipeline.py` + `test_pipeline.py`

**Wave E** (závisí na pipeline — pošli v JEDNÉ zprávě jako 2 Task calls):
- Worker 10: `api.py` + `test_api.py`
- Worker 11: `ui.py` + `test_ui.py`

**Wave F** (závisí na vše):
- Worker 12: `cli.py` (zavazuje vše dohromady)

### Krok 3 — verifikace

```bash
uv run pytest tests/unit -q   # musí být 77+ passing, 0 failing
uv run ruff check src/
uv run mypy src/local_rag
uv run rag --help              # musí ukázat 7 commandů
```

### Krok 4 — dokumentace a commit

Vytvoř:
- `README.md`
- `docs/decisions.md` (ADR-001 až ADR-004)
- `docs/experiments.md`
- `docs/learning-log.md`
- `tests/eval/qa_set.jsonl` (10 Q&A o Pythonu + 2 negativní)
- `.gitignore`
- `.env.example`

```bash
git add .
git commit -m "Initial bootstrap: complete RAG project"
```

### Krok 5 — report

```
✓ Bootstrap complete

Moduly: 11
Testy: X/X passing
Ruff: clean
Mypy: clean
CLI: uv run rag --help OK

Příští krok: stáhnout data a spustit ingest
  git clone --depth 1 https://github.com/python/peps.git /tmp/peps
  cp /tmp/peps/peps/pep-0008.rst ... data/raw/peps/
  uv run rag ingest data/raw/peps --collection peps
```

---

## Pravidla

- NIKDY se neptáš — prostě děláš
- Max 3 paralelní Task workers najednou
- Při testu fail: opravíš sám, nezastavuješ
- Při neřešitelném problému: napíšeš co a proč, navrhneš alternativu
- Neinstaluj globálně, nepoužívej sudo
