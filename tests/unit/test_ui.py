import types
from unittest.mock import MagicMock, patch


def test_get_cloud_models_success():
    mock_result = MagicMock()
    mock_result.stdout = (
        "NAME                    ID              SIZE    MODIFIED\n"
        "qwen3-coder:480b-cloud  abc123          -       2 days ago\n"
    )
    with patch("local_rag.ui.subprocess.run", return_value=mock_result):
        import local_rag.ui as ui
        models = ui._get_cloud_models()
    assert "qwen3-coder:480b-cloud" in models


def test_get_cloud_models_failure():
    with patch("local_rag.ui.subprocess.run", side_effect=Exception("no ollama")):
        import local_rag.ui as ui
        models = ui._get_cloud_models()
    assert models == ["qwen3-coder:480b-cloud"]


def test_query_fn_returns_answer():
    import local_rag.ui as ui

    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = ("The answer is 42.", [])

    with patch.object(ui, "_pipeline", mock_pipeline):
        answer, sources = ui.query_fn(
            question="What is the answer?",
            collection="default",
            model="qwen3-coder:480b-cloud",
            top_k=5,
            show_sources=False,
        )

    assert answer == "The answer is 42."
    mock_pipeline.query.assert_called_once_with("What is the answer?", "default", 5)


def test_query_fn_show_sources():
    import local_rag.ui as ui

    chunks = [
        {"source": "doc.pdf", "page": 1, "text": "Python is a programming language."},
        {"source": "guide.md", "page": 3, "text": "FastAPI is a web framework."},
    ]
    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = ("Some answer.", chunks)

    with patch.object(ui, "_pipeline", mock_pipeline):
        answer, sources = ui.query_fn(
            question="Tell me about Python",
            collection="python-docs",
            model="qwen3-coder:480b-cloud",
            top_k=3,
            show_sources=True,
        )

    assert answer == "Some answer."
    assert "doc.pdf" in sources
    assert "guide.md" in sources
    assert "[1]" in sources
    assert "[2]" in sources


def test_query_fn_hide_sources():
    import local_rag.ui as ui

    chunks = [
        {"source": "doc.pdf", "page": 1, "text": "Python is a programming language."},
    ]
    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = ("Answer without sources.", chunks)

    with patch.object(ui, "_pipeline", mock_pipeline):
        answer, sources = ui.query_fn(
            question="What is Python?",
            collection="default",
            model="qwen3-coder:480b-cloud",
            top_k=5,
            show_sources=False,
        )

    assert answer == "Answer without sources."
    assert sources == ""


def test_create_ui_returns_blocks():
    import local_rag.ui as ui
    import gradio as gr

    mock_result = MagicMock()
    mock_result.stdout = (
        "NAME                    ID              SIZE    MODIFIED\n"
        "qwen3-coder:480b-cloud  abc123          -       2 days ago\n"
    )
    with patch("local_rag.ui.subprocess.run", return_value=mock_result):
        demo = ui.create_ui()

    assert isinstance(demo, gr.Blocks)
