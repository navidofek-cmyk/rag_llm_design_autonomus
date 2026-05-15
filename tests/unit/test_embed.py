import numpy as np
import pytest

from local_rag.embed import Embedder


def test_embedder_init(mocker):
    mock_st = mocker.patch("local_rag.embed.SentenceTransformer")
    embedder = Embedder(model_name="BAAI/bge-small-en", device="cpu")
    mock_st.assert_called_once_with("BAAI/bge-small-en", device="cpu")
    assert embedder.model_name == "BAAI/bge-small-en"


def test_embed_returns_ndarray(mocker):
    mock_st = mocker.patch("local_rag.embed.SentenceTransformer")
    mock_st.return_value.encode.return_value = np.array([[0.1, 0.2]])
    embedder = Embedder()
    result = embedder.embed(["hello world"])
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 2)


def test_embed_one_returns_1d(mocker):
    mock_st = mocker.patch("local_rag.embed.SentenceTransformer")
    mock_st.return_value.encode.return_value = np.array([[0.1, 0.2]])
    embedder = Embedder()
    result = embedder.embed_one("hello world")
    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.shape == (2,)


def test_embed_uses_normalize(mocker):
    mock_st = mocker.patch("local_rag.embed.SentenceTransformer")
    mock_st.return_value.encode.return_value = np.array([[0.1, 0.2]])
    embedder = Embedder()
    embedder.embed(["test"])
    call_kwargs = mock_st.return_value.encode.call_args
    assert call_kwargs.kwargs.get("normalize_embeddings") is True


def test_embed_uses_no_progress(mocker):
    mock_st = mocker.patch("local_rag.embed.SentenceTransformer")
    mock_st.return_value.encode.return_value = np.array([[0.1, 0.2]])
    embedder = Embedder()
    embedder.embed(["test"])
    call_kwargs = mock_st.return_value.encode.call_args
    assert call_kwargs.kwargs.get("show_progress_bar") is False
