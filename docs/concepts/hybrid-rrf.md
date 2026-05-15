# Hybrid Retrieval and Reciprocal Rank Fusion

## Why Neither Method Alone Is Best

Dense retrieval (vector search) and sparse retrieval (BM25) have complementary failure modes:

| Situation | Dense | BM25 |
|-----------|-------|------|
| "What is a closure?" | Excellent — semantic match | OK — word may appear |
| "Python 3.12 exception groups" | Weak — version numbers poorly embedded | Excellent — exact keywords |
| "How do I write a list comprehension?" | Excellent — understands intent | Good — keywords present |
| `"TypeError: unsupported operand"` | Weak — unusual token sequence | Excellent — verbatim match |
| Synonym ("automobile" vs "car") | Excellent | Fails completely |
| Paraphrase of same concept | Excellent | Often fails |

The pattern is clear: dense retrieval wins on semantic understanding; sparse retrieval wins on exact keyword matching. The natural conclusion is to run both and combine the results.

But how do you combine two ranked lists whose scores are on completely different scales? BM25 produces scores like `4.72` or `12.3`; dense distances are in `[0, 2]`. You cannot simply add them.

---

## Reciprocal Rank Fusion (RRF)

**RRF** is the answer. Introduced by Cormack, Clarke & Buettcher (2009), it ignores scores entirely and works only with **ranks** — the position of each result in each ranked list.

The formula for the RRF score of a document is:

```
RRF_score(document) = sum over all lists of: 1 / (k + rank_in_list)
```

Where:
- `rank_in_list` is the document's 1-based position in that retrieval list (1 = best)
- `k` is a smoothing constant (default: 60)
- The sum runs over every retrieval method that returned this document

A document that appears in **both** lists gets contributions from both terms. A document that appears in only one list gets a contribution from one term. The RRF score is then used to sort the final merged list.

---

## Concrete Example

Suppose dense retrieval returns:
```
Rank 1: chunk A (about Python closures)
Rank 2: chunk B (about Python scoping)
Rank 3: chunk C (about LEGB rule)
```

And BM25 returns:
```
Rank 1: chunk C (exact keyword match)
Rank 2: chunk D (another keyword match)
Rank 3: chunk A
```

With k=60, the RRF scores are:

```
chunk A: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
chunk B: 1/(60+2)             = 0.01613
chunk C: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
chunk D: 1/(60+2)             = 0.01613
```

Final ranking:
```
Rank 1: chunk A  (0.03226) — high in dense, also present in BM25
Rank 1: chunk C  (0.03226) — high in BM25, also present in dense
Rank 3: chunk B  (0.01613)
Rank 3: chunk D  (0.01613)
```

Chunks that appear in both lists are rewarded. Chunks present in only one list score lower. The system naturally favors **consensus results** across retrieval methods.

---

## Why k=60 Specifically?

The k constant controls the **dampening effect**. Consider what happens as rank increases:

```
k=60:
  rank=1:   1/61  = 0.01639
  rank=2:   1/62  = 0.01613
  rank=5:   1/65  = 0.01538
  rank=10:  1/70  = 0.01429
  rank=60:  1/120 = 0.00833
  rank=100: 1/160 = 0.00625
```

With k=60, even a result at rank 100 gets a non-trivial score (0.00625). If k were 0, rank 1 scores 1.0 and rank 100 scores 0.01 — a 100x difference. The large k value **flattens** the rank-score curve, meaning being ranked 5th vs ranked 60th matters much less than being in both lists vs only one list.

This makes RRF **robust**: a document in both lists at moderate ranks beats a document in only one list at rank 1. The original paper found k=60 to give consistently strong performance across many different dataset combinations, and it has become the de-facto standard.

> **Key insight:** RRF is score-agnostic and rank-based. It does not matter that BM25 returns scores of 12.4 and dense returns distances of 0.3 — RRF only looks at positions. This makes it universally applicable whenever you have two or more ranked lists to combine.

---

## Why We Fetch `fetch_k=20` Then Return `top_k=5`

The hybrid retrieval function signature:

```python
def hybrid_retrieve(
    question: str,
    embedder: "Embedder",
    bm25_index: "BM25Index",
    chroma_dir: str,
    collection: str,
    top_k: int = 5,
    fetch_k: int = 20,
) -> list[dict]:
    dense = retrieve.retrieve(question, embedder, chroma_dir, collection, fetch_k)
    bm25_results = bm25.search(bm25_index, question, fetch_k)
    fused = reciprocal_rank_fusion(dense, bm25_results)
    return fused[:top_k]
```

Why fetch 20 from each method when you only want 5 results?

Consider a simple example: you fetch `top_k=5` from both dense and BM25. Dense returns [A, B, C, D, E]. BM25 returns [F, G, H, I, J]. There is **zero overlap**. RRF has nothing to boost — every result appears in only one list, so there is no "consensus bonus." You end up with a merged list of 10 results and take the top 5 — which could be almost arbitrary.

By fetching `fetch_k=20` from each, you dramatically increase the chance that good documents appear in **both** lists, allowing RRF to identify and boost the true consensus results. You then trim to `top_k=5` after fusion.

The `fetch_k` / `top_k` ratio of 4x (20/5) is a common practical choice. You can experiment with this in `.env`:

```dotenv
FETCH_K=20
TOP_K=5
```

---

## The Full RRF Implementation

From `src/local_rag/hybrid.py`:

```python
def reciprocal_rank_fusion(
    dense: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    for rank, chunk in enumerate(dense, start=1):
        text = chunk["text"]
        scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank)
        if text not in chunks:
            chunks[text] = dict(chunk)

    for rank, chunk in enumerate(bm25_results, start=1):
        text = chunk["text"]
        scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank)
        if text not in chunks:
            chunks[text] = dict(chunk)

    results = []
    for text, score in scores.items():
        item = dict(chunks[text])
        item["rrf_score"] = score
        results.append(item)

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results
```

Several implementation details worth noting:

**Text as the key.** Documents are deduplicated by their raw text content (`text = chunk["text"]`). If the same chunk appears in both the dense and BM25 results, it accumulates scores from both contributions instead of appearing twice.

**First-seen metadata wins.** `if text not in chunks: chunks[text] = dict(chunk)` ensures we keep the metadata (source, page, distance) from the first time we see a chunk. This is arbitrary but deterministic.

**`rrf_score` added to output.** Each result gets an `rrf_score` field so you can see how the fusion scored it. Useful for debugging.

**`enumerate(dense, start=1)`.** Ranks are 1-based, matching the original RRF paper's formula.

---

## Enabling Hybrid Retrieval

In `.env`, set:

```dotenv
USE_HYBRID=true
FETCH_K=20
```

The pipeline will automatically build (and cache) a BM25 index on first query, then run both retrieval methods and fuse the results. See [architecture.md](../architecture.md) for how this fits into the full pipeline.
