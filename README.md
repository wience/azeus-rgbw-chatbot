# RGBW Chatbot

A RAG-based chatbot built during my Software Developer Internship at Azeus Systems (June – July 2024). Combines structured (SQL) and unstructured (vector) document retrieval to answer questions over an enterprise document corpus.

> **Note:** This repository is portfolio documentation of work completed during my internship. The full production code lives in Azeus's private infrastructure. The architecture, decisions, and representative code shown here are based on what I built.

## Problem

The internal team needed a chatbot that could answer questions across two different document types:

1. **Structured records** stored in Supabase Postgres (IDs, names, fields with exact-match semantics)
2. **Unstructured documents** (PDFs, internal notes) where semantic similarity beats exact-match

A single retriever wouldn't work — SQL beats embeddings on structured data; embeddings beat keyword search on unstructured text. The chatbot needed both, with the LLM agent deciding which tool to call per query.

## Architecture

```
User query
    ↓
LangChain agent (GROQ Llama 3 70B)
    ↓
Tool routing
    ├── Supabase SQLDocStore  → structured queries
    └── ChromaDB              → semantic similarity (top-k)
        ↓
    retrieved context
    ↓
LLM synthesis grounded in retrieved chunks
    ↓
Response
```

## Tech stack

- **Orchestration:** LangChain (chains + agents + tool routing)
- **LLM:** GROQ-hosted Llama 3 70B
- **Structured store:** Supabase (Postgres) via `SQLDocStore` adapter
- **Vector store:** ChromaDB (persisted locally)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Language:** Python 3.11

## Key decisions

**Why GROQ over OpenAI?** Latency. GROQ's LPU inference delivered roughly 10x faster token throughput than OpenAI's GPT-3.5 at the time, at lower cost — important for chat where perceived speed is the experience.

**Why split SQLDocStore + Chroma?** A pure vector store ranked structured fields poorly (exact ID lookups returned semantically-similar-but-wrong records). The split let the agent pick the right tool per query: SQL for "give me record #4521", vector search for "find documents about X".

**Why all-MiniLM-L6-v2 embeddings?** ~80MB model, fast inference on CPU, good-enough quality for English business docs. We didn't need OpenAI ada-002 cost at the prototype stage.

**Why an agent, not a pipeline?** Some queries needed both stores (e.g., "summarize the documents linked to project ID 4521"). A hard-coded pipeline would have forced one retrieval; the agent could chain SQL → vector → synthesis dynamically.

## Outcomes

- Demoed end-to-end to the internal team within the 6-week internship window
- Established the RAG + dual-retriever pattern the team continued building on after I left
- Documented the tool-routing approach for the team's later production work

## Representative code

See [`examples/rag_chain.py`](examples/rag_chain.py) for the LangChain agent + dual-tool wiring pattern.

## About me

Built by **Wince Dela Fuente**.

- Portfolio: [wience.tech](https://wience.tech)
- LinkedIn: [wince-dela-fuente](https://linkedin.com/in/wince-dela-fuente-61b3a8293/)
- GitHub: [@wience](https://github.com/wience)
