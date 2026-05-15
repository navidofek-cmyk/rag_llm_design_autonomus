from unittest.mock import MagicMock, patch

import pytest
from rank_bm25 import BM25Okapi

from local_rag.bm25 import BM25Index, build_index, search


def make_mock_client(documents, metadatas=None):
    if metadatas is None:
        metadatas = [{"source": f"src{i}", "page": i} for i in range(len(documents))]
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "documents": documents,
        "metadatas": metadatas,
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    return mock_client


def test_build_index_empty_collection():
    mock_client = make_mock_client(documents=[], metadatas=[])
    with patch("local_rag.bm25.chromadb.PersistentClient", return_value=mock_client):
        index = build_index("/fake/chroma", "test_collection")
    assert index._index is None
    assert index._chunks == []


def test_build_index_with_documents():
    docs = ["hello world", "foo bar baz", "python rag test"]
    mock_client = make_mock_client(documents=docs)
    with patch("local_rag.bm25.chromadb.PersistentClient", return_value=mock_client):
        index = build_index("/fake/chroma", "test_collection")
    assert index._index is not None
    assert isinstance(index._index, BM25Okapi)
    assert len(index._chunks) == 3
    assert index._chunks[0]["text"] == "hello world"
    assert index._chunks[1]["text"] == "foo bar baz"
    assert index._chunks[2]["text"] == "python rag test"


def test_build_index_chunks_have_source_and_page():
    docs = ["doc one", "doc two"]
    metas = [{"source": "file1.txt", "page": 1}, {"source": "file2.txt", "page": 2}]
    mock_client = make_mock_client(documents=docs, metadatas=metas)
    with patch("local_rag.bm25.chromadb.PersistentClient", return_value=mock_client):
        index = build_index("/fake/chroma", "col")
    assert index._chunks[0]["source"] == "file1.txt"
    assert index._chunks[0]["page"] == 1
    assert index._chunks[1]["source"] == "file2.txt"
    assert index._chunks[1]["page"] == 2


def test_search_empty_index():
    index = BM25Index(_index=None, _chunks=[])
    result = search(index, "any query")
    assert result == []


def test_search_returns_results():
    docs = ["python is great", "java is verbose", "rag retrieval augmented generation"]
    tokenized = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    chunks = [{"text": doc, "source": "src", "page": 0} for doc in docs]
    index = BM25Index(_index=bm25, _chunks=chunks)

    results = search(index, "python", top_k=3)
    assert len(results) > 0
    texts = [r["text"] for r in results]
    assert "python is great" in texts


def test_search_top_k_limit():
    docs = [f"document number {i}" for i in range(10)]
    tokenized = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    chunks = [{"text": doc, "source": "", "page": 0} for doc in docs]
    index = BM25Index(_index=bm25, _chunks=chunks)

    results = search(index, "document", top_k=3)
    assert len(results) == 3


def test_search_adds_score():
    docs = ["alpha beta gamma", "delta epsilon zeta", "theta iota kappa"]
    tokenized = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    chunks = [{"text": doc, "source": "", "page": 0} for doc in docs]
    index = BM25Index(_index=bm25, _chunks=chunks)

    results = search(index, "alpha", top_k=3)
    for r in results:
        assert "bm25_score" in r
        assert isinstance(r["bm25_score"], float)


def test_search_does_not_mutate_chunks():
    docs = ["hello world", "foo bar"]
    tokenized = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    chunks = [{"text": doc, "source": "", "page": 0} for doc in docs]
    index = BM25Index(_index=bm25, _chunks=chunks)

    search(index, "hello", top_k=2)

    for chunk in index._chunks:
        assert "bm25_score" not in chunk
