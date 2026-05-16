import json
from pathlib import Path
import typer

app = typer.Typer(help="Local RAG CLI")

def _get_pipeline():
    from local_rag.pipeline import RAGPipeline
    return RAGPipeline()

@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to file or directory to ingest"),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection name"),
):
    """Ingest documents into the vector store."""
    pipeline = _get_pipeline()
    n = pipeline.ingest(path, collection)
    typer.echo(f"Ingested {n} chunks into collection '{collection}'")

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection name"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve"),
    show_sources: bool = typer.Option(False, "--sources", "-s", help="Show source chunks"),
):
    """Ask a question against the RAG pipeline."""
    pipeline = _get_pipeline()
    answer, chunks = pipeline.query(question, collection, top_k)
    typer.echo(answer)
    if show_sources:
        typer.echo("\n--- Sources ---")
        for i, chunk in enumerate(chunks, 1):
            typer.echo(f"[{i}] {chunk.get('source','?')} p.{chunk.get('page','?')}")

@app.command()
def eval(
    qa_file: str = typer.Argument(..., help="Path to JSONL file with Q&A pairs"),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection name"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve"),
):
    """Evaluate the RAG pipeline on a Q&A set."""
    pipeline = _get_pipeline()
    results = []
    qa_path = Path(qa_file)
    with qa_path.open() as f:
        for line in f:
            item = json.loads(line.strip())
            question = item["question"]
            expected = item.get("answer", "")
            answer, chunks = pipeline.query(question, collection, top_k)
            results.append({"question": question, "answer": answer, "expected": expected, "sources": len(chunks)})
    typer.echo(f"Evaluated {len(results)} questions")
    for r in results:
        typer.echo(f"Q: {r['question'][:60]}... | Sources: {r['sources']}")

@app.command()
def inspect(
    collection: str = typer.Argument(..., help="Collection name to inspect"),
):
    """Inspect a collection (count, sample chunks)."""
    import os
    import chromadb
    chroma_dir = os.getenv("CHROMA_DIR", "./data/chroma")
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        col = client.get_collection(collection)
        count = col.count()
        typer.echo(f"Collection '{collection}': {count} chunks")
        if count > 0:
            sample = col.get(limit=3, include=["documents", "metadatas"])
            for i, (doc, meta) in enumerate(zip(sample["documents"] or [], sample["metadatas"] or []), 1):
                typer.echo(f"  [{i}] {meta.get('source','?')} p.{meta.get('page','?')}: {doc[:100]}...")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

@app.command(name="batch-ask")
def batch_ask(
    input_file: str = typer.Argument(..., help="JSONL file with 'question' fields"),
    output_file: str = typer.Argument(..., help="Output JSONL file"),
    collection: str = typer.Option("default", "--collection", "-c", help="Collection name"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve"),
):
    """Batch process questions from a JSONL file."""
    pipeline = _get_pipeline()
    results = []
    with Path(input_file).open() as f:
        questions = [json.loads(line)["question"] for line in f if line.strip()]
    for q in questions:
        answer, chunks = pipeline.query(q, collection, top_k)
        results.append({"question": q, "answer": answer, "sources": len(chunks)})
    with Path(output_file).open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    typer.echo(f"Processed {len(results)} questions → {output_file}")

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run("local_rag.api:app", host=host, port=port, reload=reload)

@app.command()
def ui():
    """Launch the Gradio UI."""
    from local_rag.ui import main
    main()

if __name__ == "__main__":
    app()
