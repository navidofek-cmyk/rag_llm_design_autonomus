from local_rag import store
from local_rag.embed import Embedder


def retrieve(
    question: str,
    embedder: Embedder,
    chroma_dir: str,
    collection: str,
    top_k: int = 5,
) -> list[dict]:
    embedding = embedder.embed_one(question)
    return store.query(embedding, chroma_dir, collection, top_k)
