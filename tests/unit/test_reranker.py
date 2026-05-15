import pytest
from unittest.mock import MagicMock


def test_reranker_init(mocker):
    mock_ce = mocker.patch("local_rag.reranker.CrossEncoder")
    from local_rag.reranker import Reranker

    reranker = Reranker()

    mock_ce.assert_called_once_with("BAAI/bge-reranker-base")
    assert reranker.model_name == "BAAI/bge-reranker-base"
    assert reranker.model is mock_ce.return_value


def test_reranker_env_model(mocker, monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL", "my-custom/reranker")
    mock_ce = mocker.patch("local_rag.reranker.CrossEncoder")
    from local_rag.reranker import Reranker

    reranker = Reranker()

    mock_ce.assert_called_once_with("my-custom/reranker")
    assert reranker.model_name == "my-custom/reranker"


def test_rerank_empty_chunks(mocker):
    mocker.patch("local_rag.reranker.CrossEncoder")
    from local_rag.reranker import Reranker

    reranker = Reranker()
    result = reranker.rerank("some question", [], top_k=5)

    assert result == []


def test_rerank_adds_score(mocker):
    mock_ce = mocker.patch("local_rag.reranker.CrossEncoder")
    mock_instance = MagicMock()
    mock_ce.return_value = mock_instance
    mock_instance.predict.return_value = [0.9, 0.1]

    from local_rag.reranker import Reranker

    reranker = Reranker()
    chunks = [{"text": "first chunk"}, {"text": "second chunk"}]
    result = reranker.rerank("question", chunks, top_k=5)

    scores = {c["text"]: c["rerank_score"] for c in result}
    assert scores["first chunk"] == pytest.approx(0.9)
    assert scores["second chunk"] == pytest.approx(0.1)


def test_rerank_sorts_descending(mocker):
    mock_ce = mocker.patch("local_rag.reranker.CrossEncoder")
    mock_instance = MagicMock()
    mock_ce.return_value = mock_instance
    mock_instance.predict.return_value = [0.2, 0.9, 0.5]

    from local_rag.reranker import Reranker

    reranker = Reranker()
    chunks = [
        {"text": "low score"},
        {"text": "high score"},
        {"text": "mid score"},
    ]
    result = reranker.rerank("question", chunks, top_k=5)

    assert result[0]["text"] == "high score"
    assert result[1]["text"] == "mid score"
    assert result[2]["text"] == "low score"


def test_rerank_top_k(mocker):
    mock_ce = mocker.patch("local_rag.reranker.CrossEncoder")
    mock_instance = MagicMock()
    mock_ce.return_value = mock_instance
    mock_instance.predict.return_value = [0.1, 0.9, 0.5, 0.8, 0.3]

    from local_rag.reranker import Reranker

    reranker = Reranker()
    chunks = [{"text": f"chunk {i}"} for i in range(5)]
    result = reranker.rerank("question", chunks, top_k=2)

    assert len(result) == 2
    assert result[0]["rerank_score"] == pytest.approx(0.9)
    assert result[1]["rerank_score"] == pytest.approx(0.8)
