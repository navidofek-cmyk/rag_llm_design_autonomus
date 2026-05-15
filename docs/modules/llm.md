# llm.py — Talking to the Language Model

**File:** `src/local_rag/llm.py`

This module is the bridge between the retrieved chunks and the final answer.
It takes a question and a list of context chunks, formats them into a prompt,
and sends the prompt to a locally running Ollama instance.

---

## Why Ollama?

The most obvious alternative is a cloud API (OpenAI, Anthropic, Google).
Cloud APIs work well but have trade-offs that matter for a learning project:

| Concern | Cloud API | Ollama (local) |
|---|---|---|
| Cost | Per-token billing | Free after hardware |
| Privacy | Documents leave your machine | Stays local |
| Internet | Required | Not required |
| Latency | ~500 ms–2 s | Depends on hardware |
| Model choice | Fixed catalog | Any GGUF/GGML model |

For a RAG learning system where you are ingesting potentially sensitive documents
and running hundreds of experimental queries, local inference removes both the
cost concern and the privacy concern. Ollama wraps models in a simple HTTP API
so the code stays clean.

---

## /api/chat vs /api/generate

Ollama exposes two generation endpoints:

- **`/api/generate`** — takes a single `prompt` string. You have to manually
  embed system instructions and conversation history into the prompt text.
- **`/api/chat`** — takes a structured `messages` array with roles (`system`,
  `user`, `assistant`). The model handles system prompt injection correctly.

This project uses **`/api/chat`** because it properly separates the system
prompt (how the model should behave) from the user message (the actual query).
With `/api/generate` you would have to manually prepend the system instructions
to the prompt string, and different models have different expected formats for
doing so (`[INST]`, `<|system|>`, etc.). The `/api/chat` endpoint abstracts
that away.

```python
url = f"{self.host}/api/chat"
payload = {
    "model": self.model,
    "messages": [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ],
    "stream": False,
}
```

---

## The SYSTEM_PROMPT Design

```python
SYSTEM_PROMPT: str = (
    "Answer only from the provided context. "
    "Cite the source when possible. "
    "If the answer is not in the context, say 'I don't know based on the provided context.'"
)
```

Three directives, each serving a specific purpose:

**"Answer only from the provided context."**
This is the core RAG constraint. Without it, the LLM would use its training
knowledge to answer questions, defeating the purpose of retrieval. A model
trained on Python documentation already knows what a decorator is — but you want
it to answer from *your* ingested corpus, which may contain project-specific
overrides or newer information.

**"Cite the source when possible."**
Encourages the model to mention the file path or document name it is drawing
from. This makes answers verifiable: the user can open the source file and
check.

**"If the answer is not in the context, say 'I don't know based on the provided context.'"**
LLMs have a strong tendency to confabulate — to produce plausible-sounding but
invented answers. This instruction gives the model an explicit exit: when the
chunks do not contain the answer, it should say so rather than hallucinate.
This is the most important safety property of a RAG system.

---

## Why stream=False?

Ollama supports streaming responses (the model sends tokens as it generates
them). Streaming is useful for interactive UIs where you want the user to see
the answer appearing word by word.

For this project, `"stream": False` was chosen because:

1. **Simpler parsing.** The response is a single JSON object. With streaming
   you receive many newline-delimited JSON objects and must reassemble them.
2. **Simpler tests.** A mock can return a single dict rather than a generator.
3. **Compatibility.** The Gradio and FastAPI interfaces do not currently use
   streaming, so the simplicity wins.

```python
response = httpx.post(url, json=payload, timeout=120)
return response.json()["message"]["content"]
```

The path `["message"]["content"]` matches the `/api/chat` response structure.
For `/api/generate` it would be `["response"]` — another reason to prefer chat.

---

## The USER_TEMPLATE Format

```python
USER_TEMPLATE: str = "Context:\n{context}\n\nQuestion: {question}"
```

The context is placed *before* the question. This ordering is important:
many transformer models attend more strongly to text that appears earlier in
the prompt (known as primacy bias). Placing the evidence before the question
helps the model ground its answer in the retrieved content.

The `---` separator between chunks (added in `ask()`) creates a visible
boundary so the model can distinguish where one chunk ends and the next begins.

---

## How `ask()` Formats Chunks into Context

```python
def ask(self, question: str, chunks: list[dict]) -> str:
    context = "\n---\n".join(chunk["text"] for chunk in chunks)
    user = USER_TEMPLATE.format(context=context, question=question)
    return self.chat(SYSTEM_PROMPT, user)
```

With 5 chunks of 512 characters each, the context is roughly 2560 characters
plus 4 separators. A full prompt including the system message and template
wording is typically under 3000 characters — well within the context window of
any modern Ollama model.

Example of what `user` looks like when sent to the model:

```
Context:
PEP 8 is the style guide for Python code...
---
Indentation: Use 4 spaces per indentation level...
---
Naming conventions: function names should be lowercase...

Question: What does PEP 8 say about indentation?
```

---

## What Happens If Ollama Is Not Running?

`httpx.post` will raise `httpx.ConnectError` if nothing is listening at
`OLLAMA_HOST`. This exception propagates up through `ask()`, `chat()`, and
out of `pipeline.query()`, ultimately returning a 500 error from the FastAPI
server or a traceback in the CLI.

There is no retry logic in this module — that is intentional simplicity for a
learning project. In a production system you would add:

```python
# Example of what you might add for resilience:
response = httpx.post(url, json=payload, timeout=120)
response.raise_for_status()   # converts 4xx/5xx to exceptions
```

To check whether Ollama is running before querying:

```bash
curl http://localhost:11434/api/tags   # returns list of available models
```

To start Ollama (once installed):

```bash
ollama serve &
ollama pull llama3.2
```

---

## OllamaClient: Full Class Reference

```python
class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        # Reads OLLAMA_HOST and LLM_MODEL from environment, with fallback defaults.
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.environ.get("LLM_MODEL", "llama3.2")

    def chat(self, system: str, user: str) -> str:
        # Low-level: send arbitrary system + user messages, return content string.
        ...

    def ask(self, question: str, chunks: list[dict]) -> str:
        # High-level: format chunks into context, call chat(), return answer.
        ...
```

The constructor accepts optional `host` and `model` arguments so tests can
inject values without manipulating environment variables:

```python
# In tests:
client = OllamaClient(host="http://mock-server", model="test-model")
```
