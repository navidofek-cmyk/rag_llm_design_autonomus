import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client(mocker):
    mock_pipeline = MagicMock()
    mocker.patch("local_rag.api.get_pipeline", return_value=mock_pipeline)
    from local_rag.api import app
    return TestClient(app), mock_pipeline


def test_health(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_success(client):
    test_client, mock_pipeline = client
    mock_pipeline.query.return_value = (
        "Paris is the capital.",
        [{"text": "Paris is the capital of France.", "source": "geo.txt", "chunk_id": "abc123"}],
    )
    response = test_client.get("/ask?q=What+is+the+capital+of+France")
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Paris is the capital."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "geo.txt"
    assert data["collection"] == "default"


def test_ask_passes_top_k(client):
    test_client, mock_pipeline = client
    mock_pipeline.query.return_value = ("answer", [])
    response = test_client.get("/ask?q=test&top_k=3")
    assert response.status_code == 200
    mock_pipeline.query.assert_called_once_with("test", "default", 3)


def test_collections_returns_list(client, mocker):
    test_client, _ = client
    mock_col_a = MagicMock()
    mock_col_a.name = "default"
    mock_col_b = MagicMock()
    mock_col_b.name = "python-docs"
    mock_chroma_client = MagicMock()
    mock_chroma_client.list_collections.return_value = [mock_col_a, mock_col_b]
    mocker.patch("local_rag.api.chromadb.PersistentClient", return_value=mock_chroma_client)
    response = test_client.get("/collections")
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert "default" in result
    assert "python-docs" in result
    assert len(result) == 2


def test_ingest_success(client):
    test_client, mock_pipeline = client
    mock_pipeline.ingest.return_value = 42
    response = test_client.post(
        "/ingest",
        json={"path": "/data/raw/docs", "collection": "my-docs"},
    )
    assert response.status_code == 200
    assert response.json() == {"chunks": 42}
    mock_pipeline.ingest.assert_called_once_with("/data/raw/docs", "my-docs")


def test_ask_collection_param(client):
    test_client, mock_pipeline = client
    mock_pipeline.query.return_value = ("some answer", [{"text": "chunk", "source": "x.txt"}])
    response = test_client.get("/ask?q=hello&collection=python-docs&top_k=10")
    assert response.status_code == 200
    data = response.json()
    assert data["collection"] == "python-docs"
    mock_pipeline.query.assert_called_once_with("hello", "python-docs", 10)


def test_ingest_default_collection(client):
    test_client, mock_pipeline = client
    mock_pipeline.ingest.return_value = 10
    response = test_client.post("/ingest", json={"path": "/some/path"})
    assert response.status_code == 200
    assert response.json() == {"chunks": 10}
    mock_pipeline.ingest.assert_called_once_with("/some/path", "default")


def test_ask_returns_answer_field(client):
    test_client, mock_pipeline = client
    mock_pipeline.query.return_value = ("42 is the answer.", [])
    response = test_client.get("/ask?q=meaning")
    assert response.status_code == 200
    assert "answer" in response.json()
    assert response.json()["answer"] == "42 is the answer."


def test_collections_empty(client, mocker):
    test_client, _ = client
    mock_chroma_client = MagicMock()
    mock_chroma_client.list_collections.return_value = []
    mocker.patch("local_rag.api.chromadb.PersistentClient", return_value=mock_chroma_client)
    response = test_client.get("/collections")
    assert response.status_code == 200
    assert response.json() == []
