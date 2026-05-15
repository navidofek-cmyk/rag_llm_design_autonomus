from unittest.mock import MagicMock, patch

import pytest

from local_rag.hybrid import hybrid_retrieve, reciprocal_rank_fusion


def make_chunk(text: str, source: str = "src", page: int = 0) -> dict:
    return {"text": text, "source": source, "page": page}


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion tests
# ---------------------------------------------------------------------------


def test_rrf_empty_lists():
    result = reciprocal_rank_fusion([], [])
    assert result == []


def test_rrf_single_list():
    dense = [make_chunk("alpha"), make_chunk("beta")]
    result = reciprocal_rank_fusion(dense, [])
    assert len(result) == 2
    # rank 1 → 1/(60+1) ≈ 0.016393...
    top = result[0]
    assert top["text"] == "alpha"
    assert pytest.approx(top["rrf_score"], rel=1e-6) == 1.0 / 61
    second = result[1]
    assert pytest.approx(second["rrf_score"], rel=1e-6) == 1.0 / 62


def test_rrf_merges_duplicates():
    shared = make_chunk("shared")
    dense = [shared]
    bm25_results = [shared]
    result = reciprocal_rank_fusion(dense, bm25_results)
    # only one entry for "shared"
    texts = [r["text"] for r in result]
    assert texts.count("shared") == 1
    # combined score = 1/61 + 1/61
    expected = 1.0 / 61 + 1.0 / 61
    assert pytest.approx(result[0]["rrf_score"], rel=1e-6) == expected


def test_rrf_sorted_descending():
    dense = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
    bm25_results = [make_chunk("c"), make_chunk("b")]
    result = reciprocal_rank_fusion(dense, bm25_results)
    scores = [r["rrf_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_adds_rrf_score_key():
    dense = [make_chunk("x"), make_chunk("y")]
    bm25_results = [make_chunk("z")]
    result = reciprocal_rank_fusion(dense, bm25_results)
    for item in result:
        assert "rrf_score" in item


# ---------------------------------------------------------------------------
# hybrid_retrieve tests
# ---------------------------------------------------------------------------


def test_hybrid_retrieve_calls_both():
    embedder = MagicMock()
    bm25_index = MagicMock()

    dense_chunks = [make_chunk("dense1"), make_chunk("dense2")]
    bm25_chunks = [make_chunk("bm1")]

    with patch("local_rag.hybrid.retrieve.retrieve", return_value=dense_chunks) as mock_dense, \
         patch("local_rag.hybrid.bm25.search", return_value=bm25_chunks) as mock_bm25:
        result = hybrid_retrieve(
            question="what is python",
            embedder=embedder,
            bm25_index=bm25_index,
            chroma_dir="/tmp/chroma",
            collection="default",
            top_k=5,
            fetch_k=20,
        )

    mock_dense.assert_called_once_with(
        "what is python", embedder, "/tmp/chroma", "default", 20
    )
    mock_bm25.assert_called_once_with(bm25_index, "what is python", 20)
    assert isinstance(result, list)


def test_hybrid_retrieve_returns_top_k():
    embedder = MagicMock()
    bm25_index = MagicMock()

    dense_chunks = [make_chunk(f"d{i}") for i in range(10)]
    bm25_chunks = [make_chunk(f"b{i}") for i in range(10)]

    with patch("local_rag.hybrid.retrieve.retrieve", return_value=dense_chunks), \
         patch("local_rag.hybrid.bm25.search", return_value=bm25_chunks):
        result = hybrid_retrieve(
            question="query",
            embedder=embedder,
            bm25_index=bm25_index,
            chroma_dir="/tmp/chroma",
            collection="col",
            top_k=3,
            fetch_k=10,
        )

    assert len(result) <= 3
