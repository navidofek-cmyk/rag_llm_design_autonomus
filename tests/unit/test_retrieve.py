from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from local_rag.retrieve import retrieve


FAKE_EMBEDDING = np.array([0.1, 0.2, 0.3], dtype=np.float32)

FAKE_RESULTS = [
    {"text": "chunk one", "source": "doc.pdf", "page": 1, "distance": 0.1},
    {"text": "chunk two", "source": "doc.pdf", "page": 2, "distance": 0.2},
]


def make_embedder(embedding: np.ndarray = FAKE_EMBEDDING) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_one.return_value = embedding
    return embedder


@patch("local_rag.retrieve.store.query", return_value=FAKE_RESULTS)
def test_retrieve_embeds_question(mock_query):
    embedder = make_embedder()
    retrieve("what is python?", embedder, "/tmp/chroma", "default")
    embedder.embed_one.assert_called_once_with("what is python?")


@patch("local_rag.retrieve.store.query", return_value=FAKE_RESULTS)
def test_retrieve_queries_store(mock_query):
    embedder = make_embedder()
    retrieve("what is python?", embedder, "/tmp/chroma", "default", top_k=5)
    mock_query.assert_called_once_with(FAKE_EMBEDDING, "/tmp/chroma", "default", 5)


@patch("local_rag.retrieve.store.query", return_value=FAKE_RESULTS)
def test_retrieve_returns_results(mock_query):
    embedder = make_embedder()
    result = retrieve("what is python?", embedder, "/tmp/chroma", "default")
    assert result == FAKE_RESULTS


@patch("local_rag.retrieve.store.query", return_value=FAKE_RESULTS)
def test_retrieve_passes_top_k(mock_query):
    embedder = make_embedder()
    retrieve("what is python?", embedder, "/tmp/chroma", "default", top_k=10)
    _, _, _, passed_top_k = mock_query.call_args.args
    assert passed_top_k == 10


@patch("local_rag.retrieve.store.query", return_value=FAKE_RESULTS)
def test_retrieve_passes_collection(mock_query):
    embedder = make_embedder()
    retrieve("what is python?", embedder, "/tmp/chroma", "python-docs", top_k=5)
    _, _, passed_collection, _ = mock_query.call_args.args
    assert passed_collection == "python-docs"


@patch("local_rag.retrieve.store.query", return_value=FAKE_RESULTS)
def test_retrieve_passes_chroma_dir(mock_query):
    embedder = make_embedder()
    retrieve("what is python?", embedder, "/data/chroma", "default", top_k=5)
    _, passed_chroma_dir, _, _ = mock_query.call_args.args
    assert passed_chroma_dir == "/data/chroma"
