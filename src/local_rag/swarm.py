import concurrent.futures
from local_rag import retrieve, hybrid, llm
from local_rag.embed import Embedder


def _agent_search(args: tuple) -> list[dict]:
    """Single agent: searches one collection independently."""
    question, embedder, chroma_dir, collection, top_k, agent_id = args
    chunks = retrieve.retrieve(question, embedder, chroma_dir, collection, top_k)
    for chunk in chunks:
        chunk["swarm_collection"] = collection
        chunk["agent_id"] = agent_id
    return chunks


def swarm_search(
    question: str,
    embedder: Embedder,
    chroma_dir: str,
    collections: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Search across multiple collections in parallel, merge with RRF."""
    if not collections:
        return []

    args_list = [
        (question, embedder, chroma_dir, col, top_k * 2, f"agent-{i}")
        for i, col in enumerate(collections)
    ]

    all_results: list[list[dict]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(collections)) as executor:
        futures = [executor.submit(_agent_search, args) for args in args_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                all_results.append(future.result())
            except Exception:
                all_results.append([])

    # Merge via RRF: treat each collection's results as a "retrieval list"
    if not all_results:
        return []

    # Flatten and deduplicate via RRF fusion across all agent results
    merged = all_results[0]
    for other in all_results[1:]:
        merged = hybrid.reciprocal_rank_fusion(merged, other)

    return merged[:top_k]


def swarm_ask(
    question: str,
    collections: list[str],
    top_k: int = 5,
    chroma_dir: str | None = None,
) -> tuple[str, list[dict]]:
    """High-level: search across collections, ask LLM."""
    import os
    if chroma_dir is None:
        chroma_dir = os.getenv("CHROMA_DIR", "./data/chroma")

    embedder = Embedder()
    llm_client = llm.OllamaClient()

    chunks = swarm_search(question, embedder, chroma_dir, collections, top_k)
    answer = llm_client.ask(question, chunks)
    return answer, chunks
