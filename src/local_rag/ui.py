import os
import subprocess
import chromadb
import gradio as gr
from local_rag.pipeline import RAGPipeline

COLLECTIONS_FALLBACK = ["python-docs", "peps", "fastapi", "default"]


def _get_cloud_models() -> list[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        models = []
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if len(parts) >= 3 and "-" in parts[2]:
                models.append(parts[0])
        return models if models else ["qwen3-coder:480b-cloud"]
    except Exception:
        return ["qwen3-coder:480b-cloud"]


def _get_collections() -> list[str]:
    try:
        chroma_dir = os.getenv("CHROMA_DIR", "./data/chroma")
        client = chromadb.PersistentClient(path=chroma_dir)
        names = sorted(c.name for c in client.list_collections())
        return names if names else COLLECTIONS_FALLBACK
    except Exception:
        return COLLECTIONS_FALLBACK


_pipeline = RAGPipeline()


def query_fn(question: str, collection: str, model: str, top_k: int, show_sources: bool) -> tuple[str, str]:
    """Query pipeline and format output."""
    import os
    os.environ["LLM_MODEL"] = model
    answer, chunks = _pipeline.query(question, collection, int(top_k))
    if show_sources:
        sources_text = "\n\n".join(
            f"[{i+1}] {c.get('source','?')} p.{c.get('page','?')}: {c.get('text','')[:200]}..."
            for i, c in enumerate(chunks)
        )
    else:
        sources_text = ""
    return answer, sources_text


def create_ui() -> gr.Blocks:
    models = _get_cloud_models()
    collections = _get_collections()
    default_collection = collections[0] if collections else "peps"
    with gr.Blocks(title="Local RAG") as demo:
        gr.Markdown("# Local RAG System")
        with gr.Row():
            with gr.Column():
                question = gr.Textbox(label="Question", placeholder="Ask something...")
                collection = gr.Dropdown(choices=collections, value=default_collection, label="Collection")
                model = gr.Dropdown(choices=models, value=models[0], label="Model")
                top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top K")
                show_sources = gr.Checkbox(value=True, label="Show Sources")
                submit = gr.Button("Ask")
            with gr.Column():
                answer = gr.Textbox(label="Answer", lines=10)
                sources = gr.Textbox(label="Sources", lines=10)
        submit.click(
            query_fn,
            inputs=[question, collection, model, top_k, show_sources],
            outputs=[answer, sources],
        )
    return demo


def main():
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    main()
