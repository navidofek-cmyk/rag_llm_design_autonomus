import os

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        if model_name is None:
            model_name = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        if not chunks:
            return []

        pairs = [(question, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        sorted_chunks = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return sorted_chunks[:top_k]
