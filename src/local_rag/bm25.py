from dataclasses import dataclass
from typing import Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class BM25Index:
    _index: Any | None
    _chunks: list[dict]


def build_index(chroma_dir: str, collection_name: str) -> BM25Index:
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(collection_name)
    results = collection.get(include=["documents", "metadatas"])

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    if not documents:
        return BM25Index(_index=None, _chunks=[])

    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)

    chunks = [
        {
            "text": doc,
            "source": meta.get("source", "") if meta else "",
            "page": meta.get("page", 0) if meta else 0,
        }
        for doc, meta in zip(documents, metadatas)
    ]

    return BM25Index(_index=bm25, _chunks=chunks)


def search(index: BM25Index, query: str, top_k: int = 5) -> list[dict]:
    if index._index is None:
        return []

    tokenized_query = query.lower().split()
    scores = index._index.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = dict(index._chunks[idx])
        chunk["bm25_score"] = float(scores[idx])
        results.append(chunk)

    return results
