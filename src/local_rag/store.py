import chromadb
import numpy as np

from local_rag.ingest import Chunk

BATCH_SIZE = 5000


def add_chunks(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    chroma_dir: str,
    collection_name: str,
) -> None:
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(collection_name)

    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch = chunks[start:end]
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings[start:end].tolist(),
            documents=[c.text for c in batch],
            metadatas=[{"source": c.source, "page": c.page} for c in batch],
        )


def query(
    embedding: np.ndarray,
    chroma_dir: str,
    collection_name: str,
    top_k: int = 5,
) -> list[dict]:
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(collection_name)
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [
        {
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "distance": dist,
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def list_collections(chroma_dir: str) -> list[str]:
    client = chromadb.PersistentClient(path=chroma_dir)
    return [c.name for c in client.list_collections()]


def count(chroma_dir: str, collection_name: str) -> int:
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(collection_name)
    return collection.count()
