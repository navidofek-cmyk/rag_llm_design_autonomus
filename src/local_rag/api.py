import functools
import os
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb

app = FastAPI(title="Local RAG API")


class IngestRequest(BaseModel):
    path: str
    collection: str = "default"


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    collection: str


@functools.lru_cache(maxsize=1)
def get_pipeline():
    from local_rag.pipeline import RAGPipeline
    return RAGPipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ask")
def ask(q: str, collection: str = "default", top_k: int = 5) -> AskResponse:
    pipeline = get_pipeline()
    answer, chunks = pipeline.query(q, collection, top_k)
    return AskResponse(answer=answer, sources=chunks, collection=collection)


@app.get("/collections")
def list_collections() -> list[str]:
    chroma_dir = os.getenv("CHROMA_DIR", "./data/chroma")
    client = chromadb.PersistentClient(path=chroma_dir)
    return [c.name for c in client.list_collections()]


@app.post("/ingest")
def ingest_documents(req: IngestRequest) -> dict:
    pipeline = get_pipeline()
    n = pipeline.ingest(req.path, req.collection)
    return {"chunks": n}
