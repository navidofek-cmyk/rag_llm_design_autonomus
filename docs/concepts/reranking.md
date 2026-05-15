# Reranking with CrossEncoders

## The Core Problem: Fast but Approximate

The dense retrieval path uses a **BiEncoder**: documents are embedded independently of the query. You compute all document embeddings once at ingest time and store them. At query time, you embed only the question and compare it against the stored embeddings. This is extremely fast — one embedding call plus an HNSW index lookup.

But there is a fundamental limitation: **the document never "sees" the query, and the query never "sees" the document**. Each is converted to a vector in isolation. The vectors are then compared geometrically.

This independence is both the strength (scalability) and the weakness (approximation) of BiEncoders. Two texts that use very similar words but in very different contexts can end up with similar embeddings. Two texts that are highly relevant to each other but phrased differently can end up with a moderate cosine similarity score.

---

## BiEncoder vs CrossEncoder: The Architectural Difference

### BiEncoder (used in `embed.py`)

```
Query: "How does Python's GIL work?"
    |
    v
Encoder → [0.12, -0.45, 0.78, ...]   (384-dim vector)

Document: "The Global Interpreter Lock prevents..."
    |
    v
Encoder → [0.11, -0.43, 0.80, ...]   (384-dim vector)

Similarity = dot product of the two vectors
```

The query and document are processed **separately** through the same encoder. Their representations never interact until the final dot product. This is the BiEncoder.

### CrossEncoder (used in `reranker.py`)

```
[Query: "How does Python's GIL work?"] [SEP] [Document: "The Global Interpreter Lock..."]
                        |
                        v
                    Encoder
              (reads BOTH together)
                        |
                        v
                  Relevance score: 0.94
```

The query and document are **concatenated** and fed through the transformer together, separated by a `[SEP]` token. Every attention head in every layer can attend from any query token to any document token and vice versa. The model directly learns to predict: "Is this document relevant to this query?"

This cross-attention is what makes CrossEncoders so much more accurate — the model can reason about the relationship between query and document, not just their independent representations.

---

## Why CrossEncoder Is Slow but Accurate

**Speed comparison:**

| Method | Document embedding | Query embedding | Similarity | Total |
|--------|-------------------|-----------------|------------|-------|
| BiEncoder | Precomputed once | ~5ms | ~0.1ms | ~5ms per query |
| CrossEncoder | N/A | N/A | ~50ms per pair | ~50ms × N pairs |

If you have 100 candidate documents and use a CrossEncoder to score all of them, that is 100 separate forward passes through the transformer. For a top-5 query, that is 20x more compute than just returning the BiEncoder results.

**Accuracy comparison:**

On standard information retrieval benchmarks (MS MARCO, BEIR), CrossEncoders consistently score 5–15 percentage points higher than BiEncoders on ranking quality metrics like NDCG and MAP. The cross-attention that makes them slow is also what makes them better.

---

## The Two-Stage Retrieval Pattern

The solution to the speed/accuracy tradeoff is to use both methods in sequence:

```
Stage 1 — Recall (fast, approximate):
  BiEncoder + HNSW → top 20 candidates in ~10ms

Stage 2 — Precision (slow, accurate):
  CrossEncoder → score all 20 candidates → top 5 results in ~200ms
```

Stage 1 ensures you do not miss relevant documents (**recall**: find everything that might be relevant). Stage 2 ensures the final results are in the right order (**precision**: put the most relevant results first). The CrossEncoder only sees 20 documents, not millions, so the cost is manageable.

This pattern — retrieve many candidates, then rerank a small set with a more expensive model — is used everywhere in production information retrieval systems. Google uses it. Bing uses it. Almost every modern search system uses it.

> **Key insight:** The two-stage pattern is about managing a fundamental tradeoff. You use a fast approximate method for the first pass (to avoid scanning millions of documents with the expensive model) and a slow accurate method for the second pass (to get the best possible ranking among a small candidate set). The first stage optimizes for recall; the second optimizes for precision.

---

## Our `Reranker` Class

From `src/local_rag/reranker.py`:

```python
import os
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        if model_name is None:
            model_name = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        if not chunks:
            return []

        pairs = [(question, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        sorted_chunks = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return sorted_chunks[:top_k]
```

**`pairs = [(question, chunk["text"]) for chunk in chunks]`** — this creates one `(query, document)` pair for each candidate chunk. The CrossEncoder will process each pair independently.

**`self.model.predict(pairs)`** — this is a single batch call to the CrossEncoder. All pairs are processed in one go, which is much more efficient than calling `predict()` once per pair. The result is a NumPy array of raw logit scores, one per pair.

**`chunk["rerank_score"] = float(score)`** — the score is attached to the chunk dict in-place before sorting. This is destructive (it modifies the input chunks), but the pipeline always creates fresh chunk dicts from Chroma, so this is safe.

**Sorting descending** — higher CrossEncoder score = more relevant. Unlike Chroma distances (lower = better), CrossEncoder scores are similarity scores (higher = better).

---

## What `rerank_score` Means

The raw CrossEncoder score is a logit — a number on an uncalibrated scale, typically ranging from about -10 to +10. You can think of it as log-odds of relevance:

```
rerank_score = +5.2  →  very confident: this document IS relevant
rerank_score =  0.0  →  uncertain: could go either way
rerank_score = -3.1  →  confident: this document is NOT relevant
```

The absolute value matters less than the relative ordering. You will commonly see:

- Top result: 4.5 to 8.0
- Middle results: 0.5 to 3.0
- Irrelevant results: -5.0 to 0.0

If all your rerank scores are negative, your document collection probably does not contain a good answer to the question — a useful signal for the "I don't know" case.

---

## Why `BAAI/bge-reranker-base`?

The BGE reranker family is trained on large-scale relevance datasets specifically for the reranking task. `bge-reranker-base` (278M parameters) strikes a good balance:

| Property | Value |
|----------|-------|
| Architecture | BERT-based CrossEncoder |
| Parameters | ~278M |
| BEIR benchmark | Top performance in its size class |
| License | MIT |
| Inference speed | ~20ms per pair on CPU |

There is also `bge-reranker-large` (560M params) for higher accuracy at the cost of 2x more compute. For a learning project on CPU, `bge-reranker-base` is the right starting point.

---

## Enabling the Reranker

In `.env`:

```dotenv
USE_RERANKER=true
RERANKER_MODEL=BAAI/bge-reranker-base
```

The pipeline initializes the `Reranker` at startup only when `USE_RERANKER=true`, so there is no cost when the reranker is disabled. When enabled, the reranker runs after retrieval (and after hybrid fusion if `USE_HYBRID=true`):

```
retrieve (top_k=20) → [optional: BM25 + RRF] → rerank (top_k=5) → LLM
```

The number of candidates passed to the reranker is controlled by `FETCH_K`. In hybrid mode, `FETCH_K` chunks are fetched from each retrieval method, fused, and then the top fused results are reranked. In dense-only mode, `TOP_K` chunks are retrieved and reranked down to `TOP_K`.
