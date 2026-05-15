import os

import httpx

SYSTEM_PROMPT: str = (
    "Answer only from the provided context. "
    "Cite the source when possible. "
    "If the answer is not in the context, say 'I don't know based on the provided context.'"
)

USER_TEMPLATE: str = "Context:\n{context}\n\nQuestion: {question}"


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.environ.get("LLM_MODEL", "llama3.2")

    def chat(self, system: str, user: str) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        response = httpx.post(url, json=payload, timeout=120)
        return response.json()["message"]["content"]

    def ask(self, question: str, chunks: list[dict]) -> str:
        context = "\n---\n".join(chunk["text"] for chunk in chunks)
        user = USER_TEMPLATE.format(context=context, question=question)
        return self.chat(SYSTEM_PROMPT, user)
