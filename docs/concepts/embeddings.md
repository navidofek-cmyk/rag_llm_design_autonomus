# Text Embeddings

## What Is an Embedding?

An embedding is a list of numbers (a vector) that represents the *meaning* of a piece of text. The key idea is deceptively simple:

> **Similar meaning → nearby vectors. Different meaning → distant vectors.**

A 384-dimensional embedding like `BAAI/bge-small-en` produces a list of 384 floating-point numbers for any input text. That list is a coordinate in 384-dimensional space. Two sentences that mean roughly the same thing will have coordinates that are close together; two sentences about completely different topics will have coordinates far apart.

This allows a computer to reason about meaning using nothing but arithmetic — no grammar rules, no dictionaries, no hand-crafted logic.

---

## Intuition: A Library Arranged by Meaning

Imagine a library where books are physically placed on shelves not alphabetically but by *semantic closeness*. Books about Python decorators sit next to books about Python closures. Books about closures sit near books about functional programming. Books about functional programming are in the same wing as books about lambda calculus, but they are far from the shelves holding cookbooks.

A 2-dimensional version might look like this:

```
          ^ abstract / theoretical
          |
   [lambda calculus]
          |
   [functional prog]
          |
   [Python closures] [Python decorators]
          |                |
   [Python generators]-----+
          |
          +------[Python web frameworks]----[FastAPI]----[HTTP]
          |
          |                                        [cookbooks]
          +-----------------------------------------> concrete / practical
```

An embedding model learns to place text into a high-dimensional version of this space by training on hundreds of millions of sentence pairs.

---

## How Sentence-Transformers Work (BiEncoder)

The `sentence-transformers` library uses a **BiEncoder** architecture built on top of a transformer (such as BERT or a variant). Here is what happens when you call `embed("How does Python's GIL work?")`:

### Step 1 — Tokenization

The text is split into subword tokens:
```
["How", "does", "Python", "'s", "G", "##IL", "work", "?"]
```

### Step 2 — Transformer Encoding

The token sequence is passed through the transformer. Each transformer layer updates every token's representation by looking at all the other tokens in the sequence (the attention mechanism). After all layers, each token has a rich contextual representation.

### Step 3 — Pooling

All the individual token vectors are collapsed into a single vector for the whole sentence. The most common method is **mean pooling**: average all the token vectors together. The result is a single 384-dimensional vector representing the entire input.

### Step 4 — The BiEncoder Insight

The "Bi" in BiEncoder means that the **document** and the **query** are encoded *separately* and *independently*. You encode all your documents once and store the results. At query time you encode only the question and compare it against the pre-stored document vectors. This is what makes it fast enough for real-time search.

---

## Why `normalize_embeddings=True`?

### Raw vs Normalized Vectors

Without normalization, two vectors might be similar in *direction* but differ dramatically in *magnitude* (length). A very long document might produce a vector with a large magnitude; a short sentence produces one with a small magnitude. Raw dot products would then be dominated by magnitude rather than directional similarity.

Normalization scales every vector to **unit length** (magnitude = 1), placing it on the surface of a unit sphere:

```
Raw vector:        [1.2,  0.8, -0.4]   magnitude ≈ 1.48
Normalized vector: [0.81, 0.54, -0.27] magnitude = 1.0
```

### Why This Matters: Cosine Similarity Becomes Dot Product

**Cosine similarity** between two vectors A and B is defined as:

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

When both vectors are unit-length, `|A| = 1` and `|B| = 1`, so:

```
cosine_similarity(A, B) = A · B
```

The cosine similarity *is* the dot product. This is important because:

- Dot products are the fastest operation in linear algebra (GPU and CPU are highly optimized for them).
- Chroma's HNSW index works with dot products under the hood.
- You can compare any two embeddings with a single dot product without worrying about magnitude normalization at query time.

> **Key insight:** `normalize_embeddings=True` is not just a nicety — it is what makes cosine similarity and dot product equivalent, enabling efficient approximate nearest-neighbor search.

---

## Why `BAAI/bge-small-en`?

BGE stands for **Beijing Academy of AI General Embeddings**. The `bge-small-en` variant is a strong choice for a learning project because:

| Property | Value |
|----------|-------|
| Embedding dimensions | 384 |
| Model size on disk | ~130 MB |
| Parameters | ~33M |
| MTEB leaderboard score | Competitive with models 3x its size |
| Device requirement | CPU only, no GPU needed |
| License | MIT |

The MTEB (Massive Text Embedding Benchmark) measures embedding quality across 56 tasks including retrieval, classification, and clustering. BGE-small-en consistently scores in the top tier for its size class, which means you get nearly the quality of much larger models with a fraction of the compute.

For comparison:
- `text-embedding-3-large` (OpenAI): 3072 dims, API call required, costs money
- `all-mpnet-base-v2`: 768 dims, good quality, 2x larger than bge-small
- `BAAI/bge-small-en`: 384 dims, excellent quality/size tradeoff, runs on any laptop

---

## Our `Embedder` Class

Here is the actual implementation from `src/local_rag/embed.py`:

```python
import os
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
    ) -> None:
        self.model_name = os.environ.get("EMBED_MODEL", model_name)
        device = os.environ.get("EMBED_DEVICE", device)
        self.model = SentenceTransformer(self.model_name, device=device)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def embed_one(self, text: str) -> np.ndarray:
        result = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return result[0]
```

**`embed(texts)`** takes a list of strings and returns a 2D NumPy array of shape `(len(texts), 384)`. Use this for batch embedding documents.

**`embed_one(text)`** is a convenience wrapper for embedding a single query. It calls `encode` with a single-element list (because the model always expects a list) and extracts `result[0]` to return a 1D array of shape `(384,)`.

### Example usage

```python
embedder = Embedder()

# Embed a batch of document chunks
docs = ["Python uses reference counting.", "The GIL prevents data races."]
vectors = embedder.embed(docs)
print(vectors.shape)  # (2, 384)
print(np.linalg.norm(vectors[0]))  # 1.0  ← unit length confirmed

# Embed a query at search time
query_vec = embedder.embed_one("What is the Global Interpreter Lock?")
print(query_vec.shape)  # (384,)

# Cosine similarity between query and first document
similarity = np.dot(query_vec, vectors[1])
print(f"Similarity: {similarity:.4f}")  # e.g. 0.7823
```

### Model caching

The first time you create an `Embedder`, the model is downloaded from HuggingFace Hub and cached in `~/.cache/huggingface/hub/`. Subsequent instantiations load from cache in seconds. The `EMBED_MODEL` and `EMBED_DEVICE` environment variables (set in `.env`) let you swap the model or move to GPU without code changes.

---

## Common Misconceptions

**"Embeddings are just keywords."**
No. Embeddings capture semantic meaning. The sentences "the car broke down" and "the vehicle stopped working" will have high cosine similarity even though they share no keywords.

**"More dimensions is always better."**
Not necessarily. BGE-small at 384 dimensions outperforms many 768-dim models on MTEB because the training procedure matters more than the dimension count. More dimensions do give more expressive capacity in theory, but they also cost more RAM, disk, and compute.

**"I need a GPU to use embeddings."**
For inference on reasonable batch sizes (hundreds of chunks), a modern CPU is fast enough. GPU matters primarily at training time. This project runs entirely on CPU.
