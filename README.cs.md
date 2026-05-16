# Lokální RAG systém

Produkčně připravený systém pro Retrieval-Augmented Generation s hybridním vyhledáváním, CrossEncoder rerankingem a swarm vyhledáváním přes více kolekcí.

> 📖 **Anglická verze:** [README.md](README.md) &nbsp;|&nbsp; 🌐 **Learning Guide:** [navidofek-cmyk.github.io/rag_llm_design_autonomus](https://navidofek-cmyk.github.io/rag_llm_design_autonomus/)

## Stack

| Komponenta | Technologie |
|------------|-------------|
| Embeddingy | `BAAI/bge-small-en` (CPU, sentence-transformers) |
| Vektorová DB | Chroma (embedded, persistentní) |
| BM25 | rank-bm25 |
| Reranker | `BAAI/bge-reranker-base` (CrossEncoder) |
| LLM | Ollama `/api/chat` |
| API | FastAPI + uvicorn |
| UI | Gradio |
| CLI | Typer |

## Rychlý start

```bash
# Instalace
uv sync
uv pip install -e .

# Konfigurace
cp .env.example .env
# Uprav .env: nastav OLLAMA_HOST, LLM_MODEL

# Ingest dokumentů
uv run rag ingest data/raw/peps --collection peps

# Dotaz
uv run rag ask "Co je PEP 8?" --collection peps

# Spuštění API serveru
uv run rag serve

# Spuštění UI
uv run rag ui
```

## Architektura

```
src/local_rag/
  ingest.py     — Načítání PDF/RST/MD/TXT, chunking (512 znaků, 64 overlap)
  embed.py      — Wrapper pro SentenceTransformer (BAAI/bge-small-en)
  store.py      — Chroma PersistentClient, batch upsert po 5000 chuncích
  retrieve.py   — Hustá vektorová retrieval
  bm25.py       — BM25Okapi sparse retrieval s lazy cache
  hybrid.py     — Reciprocal Rank Fusion (RRF) dense + BM25
  reranker.py   — CrossEncoder reranker (BAAI/bge-reranker-base)
  swarm.py      — Paralelní vyhledávání přes více kolekcí, spojení přes RRF
  llm.py        — Ollama /api/chat klient se strukturovaným promptingem
  pipeline.py   — Orchestrace: ingest → embed → store → retrieve → rerank → LLM
  api.py        — FastAPI REST endpointy
  ui.py         — Gradio Blocks rozhraní
  cli.py        — Typer CLI (7 příkazů)
```

## CLI příkazy

```
rag ingest <cesta>             Ingestuje soubory nebo adresář
rag ask <otázka>               Položí otázku
rag eval <qa_soubor>           Vyhodnotí na Q&A JSONL sadě
rag inspect <kolekce>          Zobrazí statistiky kolekce a ukázky chunků
rag batch-ask <vstup> <výstup> Dávkové zpracování otázek
rag serve                      Spustí FastAPI na :8000
rag ui                         Spustí Gradio UI na 0.0.0.0:7860
```

## Konfigurace

Vše přes `.env`:

| Proměnná | Výchozí | Popis |
|----------|---------|-------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `LLM_MODEL` | `qwen3-coder:480b-cloud` | Model |
| `EMBED_MODEL` | `BAAI/bge-small-en` | Embedding model |
| `CHROMA_DIR` | `./data/chroma` | Chroma persistence cesta |
| `CHUNK_SIZE` | `512` | Znaků na chunk |
| `CHUNK_OVERLAP` | `64` | Překryv mezi chunky |
| `TOP_K` | `5` | Počet retrievovaných chunků |
| `FETCH_K` | `20` | Chunků načtených před rerankingem |
| `USE_HYBRID` | `false` | Zapnout hybridní retrieval |
| `USE_RERANKER` | `false` | Zapnout CrossEncoder reranking |

## Testování

```bash
uv run pytest tests/unit -q    # 77 testů
uv run ruff check src/
uv run mypy src/local_rag
```

## Dokumentace

Kompletní learning guide (koncepty, architektura, moduly): viz záložka **GitHub Pages** tohoto repozitáře.
