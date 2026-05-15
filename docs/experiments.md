# Experiments Log

## Experiment 1: Chunk Size Impact

**Hypothesis:** Smaller chunks (256 chars) improve retrieval precision at the cost of context.

**Setup:**
- Ingest same corpus with chunk_size=256 vs chunk_size=512
- Evaluate on qa_set.jsonl (10 questions)
- Metric: answer correctness (manual review)

**To run:**
```bash
CHUNK_SIZE=256 uv run rag ingest data/raw/peps --collection peps-256
CHUNK_SIZE=512 uv run rag ingest data/raw/peps --collection peps-512
uv run rag eval tests/eval/qa_set.jsonl --collection peps-256
uv run rag eval tests/eval/qa_set.jsonl --collection peps-512
```

---

## Experiment 2: Hybrid vs Dense Retrieval

**Hypothesis:** Hybrid retrieval (dense + BM25 via RRF) outperforms dense-only on keyword-heavy queries.

**Setup:**
- USE_HYBRID=false vs USE_HYBRID=true
- Fixed qa_set.jsonl evaluation set
- Metric: source recall (does the correct document appear in top-5?)

**To run:**
```bash
USE_HYBRID=false uv run rag eval tests/eval/qa_set.jsonl --collection peps
USE_HYBRID=true uv run rag eval tests/eval/qa_set.jsonl --collection peps
```

---

## Experiment 3: Reranker Impact

**Hypothesis:** CrossEncoder reranking improves answer quality by selecting the most relevant chunks.

**Setup:**
- USE_RERANKER=false vs USE_RERANKER=true
- Same qa_set.jsonl
- Metric: answer faithfulness (manual)

**To run:**
```bash
USE_RERANKER=false uv run rag eval tests/eval/qa_set.jsonl
USE_RERANKER=true uv run rag eval tests/eval/qa_set.jsonl
```

---

## Experiment 4: Swarm Search Across Collections

**Hypothesis:** Searching peps + python-docs + fastapi simultaneously via swarm returns more complete answers than single-collection search.

**To run:**
```python
from local_rag.swarm import swarm_ask
answer, chunks = swarm_ask(
    "How does FastAPI handle dependency injection?",
    collections=["peps", "python-docs", "fastapi"],
    top_k=5
)
```
