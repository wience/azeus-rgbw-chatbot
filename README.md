# RGBW Chatbot — Production Multi-RAG System

A production retrieval-augmented chatbot I built during my Software Developer Internship at Azeus Systems (June – July 2024). Answers domain-specific queries over the team's **Redmine** issue and discussion threads using a 5-stage multi-query retrieval pipeline orchestrated in LangGraph.

> **Note:** This repository is portfolio documentation of work completed during my internship. The full production code lives in Azeus Systems's private infrastructure. The architecture, retrieval pipeline, and representative code shown here are based on what I designed and built.

## Scope and credits

The full system had two parts. I want to be precise about which I owned:

- **Upstream — data ingestion (teammate-owned).** A teammate handled parsing Redmine issue threads and producing cleaned training/embedding-ready data. I did not write that code.
- **Downstream — RAG pipeline (mine).** Everything covered in this README: 5-stage retrieval design, LangGraph orchestration, multi-query expansion, parent/child chunking strategy, double-rerank, custom SQLDocStore extension for embedding storage, LangSmith tracing, and the Streamlit GUI.

## Problem

The internal team needed a chatbot that could autonomously answer domain-specific questions over the team's Redmine corpus — issue threads, comment chains, and project discussions. Redmine threads are long, multi-author, partially-structured (status, tracker, priority fields surround free-form discussion), and rich with the institutional knowledge the team wanted on tap.

Off-the-shelf "stuff-the-text-into-Chroma" RAG produced poor answers because:

1. Chunks lost their **context** (a comment from a "Refunds" tracker reads identically to a comment from a "Pricing" tracker once embedded)
2. Naive top-k retrieval missed relevant chunks when the user's phrasing didn't lexically match the source
3. Small chunks gave precise retrieval but lacked context for synthesis; large chunks gave context but blew the relevance ranking

The system needed multi-stage retrieval, multi-query expansion, hierarchical (parent/child) chunking, reranking, and stateful orchestration.

## Architecture

```
User question
    ↓
LangGraph orchestrator (state machine)
    ↓
┌─ Stage 1: Multi-query expansion ─────────────────────────────┐
│  Fast model (GPT-3.5 / Haiku) generates 1 paraphrase         │
│  → 2 queries total: original + paraphrase                    │
└──────────────────────────────────────────────────────────────┘
    ↓
┌─ Stage 2: Child-chunk retrieval (per query, x2) ─────────────┐
│  Chroma similarity search → top 200 small child chunks       │
│  (200–250 chars each, with up to 100-char context header)    │
└──────────────────────────────────────────────────────────────┘
    ↓
┌─ Stage 3: First rerank ──────────────────────────────────────┐
│  Rerank 200 child chunks per query                           │
│  Drop < 0.1 relevance score → keep up to 150 per query       │
└──────────────────────────────────────────────────────────────┘
    ↓
┌─ Stage 4: Parent expansion ──────────────────────────────────┐
│  Map child chunks → parent chunks (up to 2500 chars)         │
│  Up to 20 parents per query → ~40 parents across 2 queries   │
└──────────────────────────────────────────────────────────────┘
    ↓
┌─ Stage 5: Second rerank + LLM synthesis ─────────────────────┐
│  Rerank 40 parent docs against ORIGINAL question             │
│  Keep top 30 with score ≥ 0.1                                │
│  Send to GPT-4-turbo or Claude Sonnet for final answer       │
└──────────────────────────────────────────────────────────────┘
    ↓
Grounded response (3K–12K tokens of context, ~1–2¢ per query)
```

## Tech stack

- **Source corpus:** Redmine (issue + comment threads from the team's project tracker)
- **Orchestration:** LangGraph (5-stage state machine)
- **Retrieval framework:** LangChain (multi-query, parent-document, reranking)
- **Vector store:** ChromaDB (child-chunk similarity search)
- **Embedding storage:** Postgres-backed `SQLDocStore` via a **custom LangChain library extension** I authored — persists vector embeddings + chunk metadata in one transactional store rather than a separate Chroma directory, simplifying backups and tenant isolation
- **Main LLM:** GPT-4-turbo or Claude 3 Sonnet (for final synthesis)
- **Fast LLM:** GPT-3.5-turbo or Claude 3 Haiku (for multi-query expansion — cheaper, faster, quality doesn't matter much for paraphrasing)
- **Observability:** LangSmith (trace inspection, agent performance tracking, evaluation)
- **GUI:** Streamlit (internal tool for non-engineers to run data ingestion + test RAG queries)
- **Language:** Python 3.11

## Key design decisions

### 1. Contextual headers on every chunk (the single biggest retrieval-quality win)

Every child and parent chunk carries a 50–100 char context header prepended before embedding. For Redmine threads, the header is the natural hierarchy:

`{project} / {tracker} / {issue_title}` — e.g., `"Customer Portal / Bug / Login fails after password reset"`

Without headers, "we rolled it back after the staging test" embeds identically whether it's from a login-bug thread or an unrelated billing thread. With headers, retrieval distinguishes them and the LLM gets cleaner signal.

### 2. Markdown-aware splitting (not naive char-count chunking)

Redmine descriptions and comments are Markdown-formatted. `MarkdownTextSplitter` cuts on heading and list boundaries first, then on character limits within sections. Result: chunks rarely cross a semantic boundary mid-thought.

### 3. Parent/child chunk hierarchy

- **Child:** 200–250 chars + ~100 char header. Small enough for precise similarity matching.
- **Parent:** up to 2500 chars + same header. Big enough for LLM to synthesize an answer with surrounding context.

Retrieval happens at the child level (precision); the LLM sees the parent level (context).

### 4. Multi-query expansion

A single user query has one phrasing bias. We generate a second paraphrase with a cheap model and retrieve for both. This recovers chunks that the original phrasing missed.

### 5. Two-stage reranking

- First rerank: after child retrieval, against the same query that fetched them — drops irrelevant noise before parent expansion
- Second rerank: after parent expansion, against the **original** question only — multi-query paraphrases helped retrieve, but we don't want them confusing the final synthesis ranking

### 6. Cost-aware model tiering

- Fast/cheap model for query expansion (no quality bar — just need a different phrasing)
- Premium model for synthesis (quality is the user-visible output)

Typical query: 3K–12K tokens of context to the synthesis model. With Claude Sonnet that's about **1–2¢ per answer**. Acceptable for internal use.

### 7. Custom SQLDocStore extension

I extended LangChain's `BaseStore` interface to persist vector embeddings into a `documents` table on Postgres alongside the structured chunk metadata, instead of in a separate Chroma persist directory. This let the team:

- Run a single backup pipeline
- Query embeddings + structured fields in one SQL statement
- Reuse standard Postgres row-level controls for access scoping

### 8. LangSmith for evaluation, not just debugging

Wired in early. Without traces it was impossible to know whether a bad answer came from bad retrieval, bad reranking, or bad synthesis. With LangSmith I could click any answer and see all 5 stages' inputs and outputs.

## Outcomes

- Production RAG pipeline delivered end-to-end within the 6-week internship (downstream of teammate-owned Redmine ingestion)
- Presented the completed system to internal stakeholders
- Custom Postgres SQLDocStore extension and the 5-stage retrieval pattern continued in use after the internship

## Representative code

- [`examples/rag_pipeline.py`](examples/rag_pipeline.py) — LangGraph 5-stage state machine, multi-query, parent retriever, double-rerank

## Photos

- [`photos/azeus-office.jpeg`](photos/azeus-office.jpeg) — at the Azeus Systems office during the internship
- [`photos/azeus-team.jpeg`](photos/azeus-team.jpeg) — with the Azeus internship team

## About me

Built by **Wince Dela Fuente**.

- Portfolio: [wience.tech](https://wience.tech)
- LinkedIn: [wince-dela-fuente](https://linkedin.com/in/wince-dela-fuente-61b3a8293/)
- GitHub: [@wience](https://github.com/wience)
