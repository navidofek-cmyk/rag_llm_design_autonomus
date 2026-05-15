import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from local_rag.ingest import Chunk
from local_rag.bm25 import BM25Index


def make_chunks(n: int = 3) -> list[Chunk]:
    return [
        Chunk(text=f"text {i}", source="file.txt", page=0, chunk_id=f"id{i}")
        for i in range(n)
    ]


def make_embeddings(n: int = 3) -> np.ndarray:
    return np.zeros((n, 384), dtype=np.float32)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for key in ("USE_HYBRID", "USE_RERANKER", "CHROMA_DIR", "TOP_K", "FETCH_K"):
        monkeypatch.delenv(key, raising=False)


def test_pipeline_init(monkeypatch):
    mock_embedder = MagicMock()
    mock_llm = MagicMock()

    with (
        patch("local_rag.pipeline.embed.Embedder", return_value=mock_embedder),
        patch("local_rag.pipeline.llm.OllamaClient", return_value=mock_llm),
    ):
        from local_rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()

    assert pipeline.embedder is mock_embedder
    assert pipeline.llm_client is mock_llm
    assert pipeline.use_hybrid is False
    assert pipeline.use_reranker is False
    assert pipeline.reranker is None
    assert isinstance(pipeline._bm25_cache, dict)


def test_ingest_directory(tmp_path, monkeypatch):
    chunks = make_chunks(4)
    embeddings = make_embeddings(4)

    with (
        patch("local_rag.pipeline.embed.Embedder"),
        patch("local_rag.pipeline.llm.OllamaClient"),
    ):
        from local_rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()

    pipeline.embedder.embed = MagicMock(return_value=embeddings)

    with (
        patch("local_rag.pipeline.ingest.load_directory", return_value=chunks) as mock_load_dir,
        patch("local_rag.pipeline.store.add_chunks") as mock_add,
    ):
        count = pipeline.ingest(str(tmp_path), collection="docs")

    assert count == 4
    mock_load_dir.assert_called_once()
    mock_add.assert_called_once()
    pipeline.embedder.embed.assert_called_once_with([c.text for c in chunks])


def test_ingest_file(tmp_path, monkeypatch):
    fake_file = tmp_path / "doc.txt"
    fake_file.write_text("hello world")
    chunks = make_chunks(2)
    embeddings = make_embeddings(2)

    with (
        patch("local_rag.pipeline.embed.Embedder"),
        patch("local_rag.pipeline.llm.OllamaClient"),
    ):
        from local_rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()

    pipeline.embedder.embed = MagicMock(return_value=embeddings)

    with (
        patch("local_rag.pipeline.ingest.load_file", return_value=chunks) as mock_load_file,
        patch("local_rag.pipeline.store.add_chunks") as mock_add,
    ):
        count = pipeline.ingest(str(fake_file), collection="docs")

    assert count == 2
    mock_load_file.assert_called_once()
    mock_add.assert_called_once()


def test_query_dense_retrieval(monkeypatch):
    chunks = make_chunks(3)
    expected_answer = "This is the answer."

    with (
        patch("local_rag.pipeline.embed.Embedder"),
        patch("local_rag.pipeline.llm.OllamaClient"),
    ):
        from local_rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()

    pipeline.llm_client.ask = MagicMock(return_value=expected_answer)

    with patch("local_rag.pipeline.retrieve.retrieve", return_value=chunks) as mock_retrieve:
        answer, result_chunks = pipeline.query("What is Python?", collection="default", top_k=3)

    assert answer == expected_answer
    assert result_chunks == chunks
    mock_retrieve.assert_called_once_with(
        "What is Python?", pipeline.embedder, pipeline.chroma_dir, "default", 3
    )
    pipeline.llm_client.ask.assert_called_once_with("What is Python?", chunks)


def test_query_hybrid_retrieval(monkeypatch):
    monkeypatch.setenv("USE_HYBRID", "true")
    chunks = make_chunks(3)
    fake_index = BM25Index(_index=None, _chunks=[])

    with (
        patch("local_rag.pipeline.embed.Embedder"),
        patch("local_rag.pipeline.llm.OllamaClient"),
    ):
        from local_rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()

    assert pipeline.use_hybrid is True
    pipeline.llm_client.ask = MagicMock(return_value="answer")

    with (
        patch("local_rag.pipeline.bm25.build_index", return_value=fake_index),
        patch("local_rag.pipeline.hybrid.hybrid_retrieve", return_value=chunks) as mock_hybrid,
        patch("local_rag.pipeline.retrieve.retrieve") as mock_dense,
    ):
        answer, result_chunks = pipeline.query("What is RAG?", collection="default", top_k=3)

    mock_hybrid.assert_called_once()
    mock_dense.assert_not_called()
    assert result_chunks == chunks


def test_bm25_cache_invalidated_after_ingest(tmp_path):
    chunks = make_chunks(2)
    embeddings = make_embeddings(2)
    fake_index = BM25Index(_index=None, _chunks=[])

    with (
        patch("local_rag.pipeline.embed.Embedder"),
        patch("local_rag.pipeline.llm.OllamaClient"),
    ):
        from local_rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()

    pipeline._bm25_cache["mycol"] = fake_index
    assert "mycol" in pipeline._bm25_cache

    pipeline.embedder.embed = MagicMock(return_value=embeddings)

    with (
        patch("local_rag.pipeline.ingest.load_file", return_value=chunks),
        patch("local_rag.pipeline.store.add_chunks"),
    ):
        fake_file = tmp_path / "doc.txt"
        fake_file.write_text("content")
        pipeline.ingest(str(fake_file), collection="mycol")

    assert "mycol" not in pipeline._bm25_cache


def test_query_with_reranker(monkeypatch):
    monkeypatch.setenv("USE_RERANKER", "true")
    chunks = make_chunks(5)
    reranked = make_chunks(3)
    expected_answer = "Reranked answer."

    mock_reranker_instance = MagicMock()
    mock_reranker_instance.rerank = MagicMock(return_value=reranked)

    with (
        patch("local_rag.pipeline.embed.Embedder"),
        patch("local_rag.pipeline.llm.OllamaClient"),
        patch("local_rag.pipeline.reranker.Reranker", return_value=mock_reranker_instance),
    ):
        from local_rag.pipeline import RAGPipeline
        pipeline = RAGPipeline()

    assert pipeline.use_reranker is True
    assert pipeline.reranker is mock_reranker_instance
    pipeline.llm_client.ask = MagicMock(return_value=expected_answer)

    with patch("local_rag.pipeline.retrieve.retrieve", return_value=chunks):
        answer, result_chunks = pipeline.query("How does reranking work?", top_k=3)

    mock_reranker_instance.rerank.assert_called_once_with("How does reranking work?", chunks, 3)
    assert result_chunks == reranked
    pipeline.llm_client.ask.assert_called_once_with("How does reranking work?", reranked)
    assert answer == expected_answer
