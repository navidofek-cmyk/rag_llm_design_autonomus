# What is RAG?

**Retrieval-Augmented Generation (RAG)** combines a retrieval system with a language model. Instead of asking the LLM to answer from memory, you first find relevant documents, then give them to the LLM as context.

## The problem RAG solves

Large language models have two fundamental limitations:

**1. Knowledge cutoff** — The model was trained on data up to a fixed date. It knows nothing about events, documents, or code written after that.

**2. Hallucination** — When the model doesn't know something, it often makes up a plausible-sounding answer instead of saying "I don't know." This is dangerous in production.

Both problems come from the same root cause: the LLM is forced to answer from *memory* (its weights), not from *evidence*.

## The RAG solution

```
User question
     │
     ▼
┌─────────────┐
│  Retrieval  │  ← Find the 5 most relevant chunks from your documents
└─────────────┘
     │
     ▼
┌─────────────┐
│  Augment    │  ← Build a prompt: "Here are the relevant docs. Now answer:"
└─────────────┘
     │
     ▼
┌─────────────┐
│ Generation  │  ← LLM answers based on the provided context, not memory
└─────────────┘
     │
     ▼
   Answer
```

The LLM's job becomes much simpler: **read the context and extract the answer**. It no longer needs to "remember" facts.

## Why not just put everything in the context window?

Modern LLMs support very long contexts (128k, even 1M tokens). Why not just load all your documents into the prompt?

| | RAG | Long context |
|---|---|---|
| Cost | Cheap — only relevant chunks sent | Expensive — full corpus every query |
| Speed | Fast — small prompt | Slow — processing millions of tokens |
| Accuracy | High — focused context | Degrades — models lose focus in long contexts ("lost in the middle" problem) |
| Scale | Unlimited — index grows independently | Hard limit — context window is fixed |

RAG wins on cost, speed, and accuracy for large document collections.

## How this project implements RAG

```
Documents → ingest.py → embed.py → store.py (Chroma)
                                        │
Question → embed.py → retrieve.py ──────┘ → bm25.py
                           │
                     hybrid.py (RRF fusion)
                           │
                    reranker.py (CrossEncoder)
                           │
                       llm.py (Ollama)
                           │
                         Answer
```

Each step is a separate, testable module. The `pipeline.py` orchestrates them all.

!!! tip "Key insight"
    RAG doesn't make the LLM smarter — it gives it better information to work with. Garbage in, garbage out still applies: retrieval quality is the bottleneck.
