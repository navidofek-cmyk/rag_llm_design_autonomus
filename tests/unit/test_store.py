import numpy as np
import pytest

from local_rag.ingest import Chunk
from local_rag.store import add_chunks, count, list_collections, query


def make_chunk(i: int) -> Chunk:
    return Chunk(
        text=f"chunk text {i}",
        source=f"doc_{i}.txt",
        page=i,
        chunk_id=f"id_{i}",
    )


def make_embedding(i: int) -> list[float]:
    return [float(i) * 0.1, float(i) * 0.2, float(i) * 0.3]


def test_add_and_count(tmp_path):
    chunks = [make_chunk(i) for i in range(3)]
    embeddings = np.array([make_embedding(i) for i in range(3)])
    chroma_dir = str(tmp_path)
    add_chunks(chunks, embeddings, chroma_dir, "col")
    assert count(chroma_dir, "col") == 3


def test_add_and_query(tmp_path):
    chunks = [make_chunk(i) for i in range(3)]
    embeddings = np.array([make_embedding(i) for i in range(3)])
    chroma_dir = str(tmp_path)
    add_chunks(chunks, embeddings, chroma_dir, "col")
    q_embedding = np.array(make_embedding(1))
    results = query(q_embedding, chroma_dir, "col", top_k=2)
    assert len(results) == 2


def test_query_returns_correct_fields(tmp_path):
    chunks = [make_chunk(i) for i in range(3)]
    embeddings = np.array([make_embedding(i) for i in range(3)])
    chroma_dir = str(tmp_path)
    add_chunks(chunks, embeddings, chroma_dir, "col")
    q_embedding = np.array(make_embedding(0))
    results = query(q_embedding, chroma_dir, "col", top_k=1)
    assert len(results) == 1
    r = results[0]
    assert "text" in r
    assert "source" in r
    assert "page" in r
    assert "distance" in r
    assert isinstance(r["text"], str)
    assert isinstance(r["source"], str)
    assert isinstance(r["page"], int)
    assert isinstance(r["distance"], float)


def test_list_collections(tmp_path):
    chroma_dir = str(tmp_path)
    chunks_a = [make_chunk(0)]
    embeddings_a = np.array([make_embedding(0)])
    add_chunks(chunks_a, embeddings_a, chroma_dir, "alpha")

    chunks_b = [make_chunk(1)]
    embeddings_b = np.array([make_embedding(1)])
    add_chunks(chunks_b, embeddings_b, chroma_dir, "beta")

    cols = list_collections(chroma_dir)
    assert set(cols) == {"alpha", "beta"}


def test_add_batching(tmp_path):
    # Add 6 chunks — verify they are all stored even across batches
    # (batch size is 5000 by default, so this tests the logic runs at all)
    n = 6
    chunks = [make_chunk(i) for i in range(n)]
    embeddings = np.array([make_embedding(i) for i in range(n)])
    chroma_dir = str(tmp_path)
    add_chunks(chunks, embeddings, chroma_dir, "col")
    assert count(chroma_dir, "col") == n


def test_add_duplicate_ids_handled(tmp_path):
    chunk = make_chunk(0)
    embedding = np.array([make_embedding(0)])
    chroma_dir = str(tmp_path)
    add_chunks([chunk], embedding, chroma_dir, "col")
    # Adding same chunk_id again should not raise (Chroma upserts)
    add_chunks([chunk], embedding, chroma_dir, "col")
    assert count(chroma_dir, "col") == 1
