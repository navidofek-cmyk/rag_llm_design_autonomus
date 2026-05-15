# Swarm Search — Parallel Multi-Collection Agents

## The Problem: Multiple Knowledge Sources

A single-collection RAG system works well when all your documents belong to a single coherent domain. But suppose you have:

- `python-docs` — official Python language documentation
- `peps` — Python Enhancement Proposals
- `fastapi` — FastAPI web framework documentation

A question like "How do I use async context managers with FastAPI?" needs context from multiple collections. If you search only `fastapi`, you miss the Python language semantics. If you search only `peps`, you miss the practical API usage. The answer lives across all three.

The naive solution is to search them sequentially: query collection 1, query collection 2, query collection 3, merge. This works but is slow — each query blocks on the previous one.

---

## The Swarm Pattern

The swarm pattern solves this with parallelism: **every collection is searched simultaneously, and no agent coordinates with any other**. There is no central dispatcher deciding which collection to search first, no negotiation, no shared state during the search phase. Each agent gets the same question and independently finds the best answer it can from its own collection.

This is the key architectural insight that distinguishes a swarm from a routed search:

- **Routed search**: A coordinator decides "this question is about PEPs, send it to the PEP agent." Requires an expensive routing step; fails if the router makes a wrong decision.
- **Swarm search**: Every agent searches in parallel; the merge step decides what was relevant. No routing errors; no bottleneck.

The name "swarm" comes from the observation that individual agents act independently (like ants or bees) and emergent order appears only at the merge step.

---

## Implementation: ThreadPoolExecutor

From `src/local_rag/swarm.py`:

```python
import concurrent.futures
from local_rag import retrieve, hybrid, llm
from local_rag.embed import Embedder


def _agent_search(args: tuple) -> list[dict]:
    """Single agent: searches one collection independently."""
    question, embedder, chroma_dir, collection, top_k, agent_id = args
    chunks = retrieve.retrieve(question, embedder, chroma_dir, collection, top_k)
    for chunk in chunks:
        chunk["swarm_collection"] = collection
        chunk["agent_id"] = agent_id
    return chunks


def swarm_search(
    question: str,
    embedder: Embedder,
    chroma_dir: str,
    collections: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Search across multiple collections in parallel, merge with RRF."""
    if not collections:
        return []

    args_list = [
        (question, embedder, chroma_dir, col, top_k * 2, f"agent-{i}")
        for i, col in enumerate(collections)
    ]

    all_results: list[list[dict]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(collections)) as executor:
        futures = [executor.submit(_agent_search, args) for args in args_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                all_results.append(future.result())
            except Exception:
                all_results.append([])

    if not all_results:
        return []

    # Merge via RRF: treat each collection's results as a "retrieval list"
    merged = all_results[0]
    for other in all_results[1:]:
        merged = hybrid.reciprocal_rank_fusion(merged, other)

    return merged[:top_k]
```

### Why `ThreadPoolExecutor` and Not `ProcessPoolExecutor`?

Python's **Global Interpreter Lock (GIL)** prevents multiple Python threads from executing CPU-bound Python code simultaneously. For CPU-intensive work (like running the embedding model), you need processes (`ProcessPoolExecutor`), not threads.

But Chroma queries are **I/O-bound**: they mostly involve reading from disk (SQLite + memory-mapped HNSW index) and the actual Python computation is minimal. I/O operations release the GIL, so threads truly run in parallel for I/O-bound work. `ThreadPoolExecutor` is lighter-weight (no process startup overhead, no memory copying) and entirely appropriate here.

If the embedding model were running *inside* each agent, you would need processes. In this implementation, the `embedder` object is shared across threads (thread-safe for inference) and the actual search is I/O-dominated — threads are the right choice.

### `top_k * 2` in Each Agent

Each agent fetches `top_k * 2` results from its collection. If the final merged answer needs 5 results, each agent fetches 10. This gives RRF more material to work with when merging — the same logic as `fetch_k > top_k` in hybrid retrieval.

### Fault Tolerance

```python
try:
    all_results.append(future.result())
except Exception:
    all_results.append([])
```

If a collection does not exist or Chroma throws an error for one agent, the exception is caught and an empty list is contributed. The other agents' results are unaffected. The system degrades gracefully rather than crashing.

---

## Merging: RRF Applied Iteratively

The merge strategy applies RRF repeatedly across agent results:

```python
merged = all_results[0]
for other in all_results[1:]:
    merged = hybrid.reciprocal_rank_fusion(merged, other)
```

With three collections [A, B, C], this computes:
1. `merged = RRF(A_results, B_results)` — fuse first two agents
2. `merged = RRF(merged, C_results)` — fuse result with third agent

RRF is applied pairwise and is order-dependent (different orders can give slightly different scores), but in practice this is robust. The same "consensus bonus" principle applies: a chunk that appears in the top-5 of two or three collections will score much higher than one that only appears in one collection's results. Cross-collection consensus is a strong signal of relevance.

---

## The `swarm_collection` and `agent_id` Metadata

Each chunk returned by the swarm carries two extra fields:

```python
chunk["swarm_collection"] = collection   # e.g., "python-docs"
chunk["agent_id"] = agent_id             # e.g., "agent-0"
```

These are useful for:

- **Debugging**: You can inspect which collections contributed to the final answer.
- **Attribution**: The UI can show "Source: python-docs (agent-0)" alongside each citation.
- **Evaluation**: You can measure which collections are most often the source of top-ranked results.

---

## High-Level `swarm_ask`

For convenience there is a `swarm_ask` function that handles the full pipeline:

```python
def swarm_ask(
    question: str,
    collections: list[str],
    top_k: int = 5,
    chroma_dir: str | None = None,
) -> tuple[str, list[dict]]:
    """High-level: search across collections, ask LLM."""
    import os
    if chroma_dir is None:
        chroma_dir = os.getenv("CHROMA_DIR", "./data/chroma")

    embedder = Embedder()
    llm_client = llm.OllamaClient()

    chunks = swarm_search(question, embedder, chroma_dir, collections, top_k)
    answer = llm_client.ask(question, chunks)
    return answer, chunks
```

This handles initialization, runs the swarm search, and hands the results to the LLM — the full pipeline in one call.

---

## When to Use Swarm vs Single Collection

| Use single collection when... | Use swarm when... |
|-------------------------------|-------------------|
| All your documents are on one topic | Documents span multiple domains |
| Query latency is critical | Accuracy matters more than latency |
| Collections are unrelated (would add noise) | Collections are complementary (all relevant) |
| You have a large single collection | You have several medium-sized collections |

Practical guidance: if you are not sure, try single collection first. The swarm pattern adds parallelism overhead and can introduce noise if collections are unrelated. But for a question like "How does FastAPI implement Python's async protocol?" — where the answer genuinely requires both FastAPI docs and Python language specs — the swarm will give a noticeably better answer.

---

## Comparison to Other Agent Patterns

| Pattern | Coordinator? | Parallelism | Use case |
|---------|-------------|-------------|----------|
| Swarm | No | Yes (ThreadPoolExecutor) | Many independent knowledge sources |
| Router | Yes (picks one) | No | Clearly separated domains |
| Chain-of-thought | Yes (sequential) | No | Multi-step reasoning |
| Map-reduce | Yes (aggregates) | Yes | Summarizing many documents |

The swarm pattern is most closely related to ensemble methods in machine learning: run many independent estimators and aggregate. The RRF merge is the "ensemble vote" that combines independent rankings into a final answer.
