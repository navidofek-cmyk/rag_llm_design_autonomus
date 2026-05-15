from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag import bm25, retrieve

if TYPE_CHECKING:
    from local_rag.bm25 import BM25Index
    from local_rag.embed import Embedder


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
