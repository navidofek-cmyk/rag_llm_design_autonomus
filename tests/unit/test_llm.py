import os
from unittest.mock import MagicMock

import pytest

from local_rag.llm import SYSTEM_PROMPT, USER_TEMPLATE, OllamaClient


def test_ollama_client_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = OllamaClient()
    assert client.host == "http://localhost:11434"
    assert client.model == "llama3.2"


def test_ollama_client_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://myhost:9999")
    monkeypatch.setenv("LLM_MODEL", "mistral")
    client = OllamaClient()
    assert client.host == "http://myhost:9999"
    assert client.model == "mistral"


def test_chat_calls_correct_endpoint(mocker: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "hello"}}
    mock_post = mocker.patch("httpx.post", return_value=mock_response)

    client = OllamaClient(host="http://testhost:1234", model="testmodel")
    client.chat("sys", "usr")

    mock_post.assert_called_once()
    called_url = mock_post.call_args[0][0]
    assert called_url == "http://testhost:1234/api/chat"


def test_chat_sends_correct_payload(mocker: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "answer"}}
    mock_post = mocker.patch("httpx.post", return_value=mock_response)

    client = OllamaClient(host="http://localhost:11434", model="llama3.2")
    client.chat("system text", "user text")

    _, kwargs = mock_post.call_args
    payload = kwargs["json"]
    assert payload["model"] == "llama3.2"
    assert payload["stream"] is False
    assert payload["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]


def test_chat_returns_content(mocker: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "expected answer"}}
    mocker.patch("httpx.post", return_value=mock_response)

    client = OllamaClient(host="http://localhost:11434", model="llama3.2")
    result = client.chat("sys", "usr")

    assert result == "expected answer"


def test_ask_formats_context(mocker: MagicMock) -> None:
    client = OllamaClient(host="http://localhost:11434", model="llama3.2")
    mock_chat = mocker.patch.object(client, "chat", return_value="mocked answer")

    chunks = [
        {"text": "First chunk text", "source": "doc1.pdf"},
        {"text": "Second chunk text", "source": "doc2.pdf"},
    ]
    result = client.ask("What is Python?", chunks)

    assert result == "mocked answer"
    mock_chat.assert_called_once()
    call_args = mock_chat.call_args
    system_arg = call_args[0][0]
    user_arg = call_args[0][1]
    assert system_arg == SYSTEM_PROMPT
    assert "First chunk text" in user_arg
    assert "Second chunk text" in user_arg
    assert "---" in user_arg
    assert "What is Python?" in user_arg


def test_ask_with_empty_chunks(mocker: MagicMock) -> None:
    client = OllamaClient(host="http://localhost:11434", model="llama3.2")
    mock_chat = mocker.patch.object(client, "chat", return_value="I don't know based on the provided context.")

    result = client.ask("What is Python?", [])

    assert result == "I don't know based on the provided context."
    mock_chat.assert_called_once()
    call_args = mock_chat.call_args
    user_arg = call_args[0][1]
    expected_user = USER_TEMPLATE.format(context="", question="What is Python?")
    assert user_arg == expected_user
