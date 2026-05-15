import os
from pathlib import Path
from dotenv import load_dotenv
from local_rag import ingest, embed, store, retrieve, bm25, hybrid, reranker, llm

load_dotenv()


class RAGPipeline:
    def __init__(self):
        self.chroma_dir = os.getenv("CHROMA_DIR", "./data/chroma")
        self.top_k = int(os.getenv("TOP_K", "5"))
        self.fetch_k = int(os.getenv("FETCH_K", "20"))
        self.use_hybrid = os.getenv("USE_HYBRID", "false").lower() == "true"
        self.use_reranker = os.getenv("USE_RERANKER", "false").lower() == "true"
        self.embedder = embed.Embedder()
        self.llm_client = llm.OllamaClient()
        self._bm25_cache: dict[str, bm25.BM25Index] = {}
        if self.use_reranker:
            self.reranker = reranker.Reranker()
        else:
            self.reranker = None

    def ingest(self, path: str, collection: str = "default") -> int:
        p = Path(path)
        if p.is_dir():
            chunks = ingest.load_directory(p)
        else:
            chunks = ingest.load_file(p)
        if not chunks:
            return 0
        embeddings = self.embedder.embed([c.text for c in chunks])
        store.add_chunks(chunks, embeddings, self.chroma_dir, collection)
        self._bm25_cache.pop(collection, None)
        return len(chunks)

    def _get_bm25_index(self, collection: str) -> bm25.BM25Index:
        if collection not in self._bm25_cache:
            self._bm25_cache[collection] = bm25.build_index(self.chroma_dir, collection)
        return self._bm25_cache[collection]

    def query(
        self,
        question: str,
        collection: str = "default",
        top_k: int | None = None,
    ) -> tuple[str, list[dict]]:
        k = top_k if top_k is not None else self.top_k
        if self.use_hybrid:
            bm25_index = self._get_bm25_index(collection)
            chunks = hybrid.hybrid_retrieve(
                question,
                self.embedder,
                bm25_index,
                self.chroma_dir,
                collection,
                k,
                self.fetch_k,
            )
        else:
            chunks = retrieve.retrieve(question, self.embedder, self.chroma_dir, collection, k)
        if self.use_reranker and self.reranker is not None:
            chunks = self.reranker.rerank(question, chunks, k)
        answer = self.llm_client.ask(question, chunks)
        return answer, chunks
