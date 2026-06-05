"""Representative LangGraph 5-stage RAG pipeline for the RGBW chatbot.

This is portfolio documentation of the multi-query, parent-retriever,
double-rerank pattern I built at Azeus Systems. Production code is
private; this captures the architecture.

No Groq, no Llama 3. Main synthesis runs on Claude 3 Sonnet or
GPT-4-turbo; query expansion runs on Claude 3 Haiku or GPT-3.5-turbo.

Run requires:
    pip install langgraph langchain langchain-community langchain-anthropic \
                langchain-openai chromadb cohere supabase

Env vars:
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    COHERE_API_KEY        # for reranking
    SUPABASE_URL
    SUPABASE_KEY
"""

from __future__ import annotations

import os
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langgraph.graph import StateGraph, END


# ──────────────────────────────────────────────────────────────────────────
# Models — tiered by cost
# ──────────────────────────────────────────────────────────────────────────
FAST_LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
SYNTHESIS_LLM = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.2)

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

# Child store — 200-250 char chunks
CHILD_STORE = Chroma(
    collection_name="rgbw_children",
    embedding_function=EMBEDDINGS,
    persist_directory="./chroma_db",
)

# Parent store — up to 2500 char chunks, keyed by parent_id
# In production this lived in Supabase via my custom SQLDocStore extension;
# stubbed here as an in-memory map.
PARENT_STORE: dict[str, Document] = {}


# Reranker — Cohere rerank-english-v3.0
RERANKER = CohereRerank(model="rerank-english-v3.0", top_n=150)
FINAL_RERANKER = CohereRerank(model="rerank-english-v3.0", top_n=30)


# ──────────────────────────────────────────────────────────────────────────
# LangGraph state
# ──────────────────────────────────────────────────────────────────────────
class RagState(TypedDict):
    original_question: str
    expanded_question: str | None
    child_chunks_q1: list[Document]
    child_chunks_q2: list[Document]
    parent_chunks: list[Document]
    reranked_parents: list[Document]
    answer: str | None


# ──────────────────────────────────────────────────────────────────────────
# Stage 1 — Multi-query expansion (one paraphrase via fast LLM)
# ──────────────────────────────────────────────────────────────────────────
EXPAND_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the following question in different words while preserving
its exact meaning. Output only the rewritten question, no preamble.

Question: {question}"""
)


def expand_query(state: RagState) -> RagState:
    chain = EXPAND_PROMPT | FAST_LLM
    expanded = chain.invoke({"question": state["original_question"]}).content.strip()
    return {**state, "expanded_question": expanded}


# ──────────────────────────────────────────────────────────────────────────
# Stage 2 — Child retrieval per question (top 200)
# ──────────────────────────────────────────────────────────────────────────
def retrieve_children(state: RagState) -> RagState:
    q1 = CHILD_STORE.similarity_search(state["original_question"], k=200)
    q2 = CHILD_STORE.similarity_search(state["expanded_question"], k=200)
    return {**state, "child_chunks_q1": q1, "child_chunks_q2": q2}


# ──────────────────────────────────────────────────────────────────────────
# Stage 3 — First rerank — drop < 0.1 relevance, keep up to 150 per question
# ──────────────────────────────────────────────────────────────────────────
def rerank_children(state: RagState) -> RagState:
    q1_ranked = RERANKER.compress_documents(
        state["child_chunks_q1"], state["original_question"]
    )
    q2_ranked = RERANKER.compress_documents(
        state["child_chunks_q2"], state["expanded_question"]
    )

    q1_filtered = [d for d in q1_ranked if d.metadata.get("relevance_score", 0) >= 0.1]
    q2_filtered = [d for d in q2_ranked if d.metadata.get("relevance_score", 0) >= 0.1]

    return {
        **state,
        "child_chunks_q1": q1_filtered,
        "child_chunks_q2": q2_filtered,
    }


# ──────────────────────────────────────────────────────────────────────────
# Stage 4 — Parent expansion — map child IDs to parent docs, dedupe
# ──────────────────────────────────────────────────────────────────────────
def expand_to_parents(state: RagState) -> RagState:
    parent_ids: set[str] = set()
    # Up to 20 parents per question → up to 40 across both
    for doc in (state["child_chunks_q1"][:20] + state["child_chunks_q2"][:20]):
        pid = doc.metadata.get("parent_id")
        if pid:
            parent_ids.add(pid)

    parents = [PARENT_STORE[pid] for pid in parent_ids if pid in PARENT_STORE]
    return {**state, "parent_chunks": parents}


# ──────────────────────────────────────────────────────────────────────────
# Stage 5 — Final rerank against ORIGINAL question, then synthesize
# ──────────────────────────────────────────────────────────────────────────
SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
    """You are the RGBW assistant. Answer the user's question using ONLY
the context below. Cite source documents inline when relevant. If the
context does not contain the answer, say so plainly — do not invent.

Context:
{context}

Question: {question}

Answer:"""
)


def synthesize(state: RagState) -> RagState:
    ranked = FINAL_RERANKER.compress_documents(
        state["parent_chunks"], state["original_question"]
    )
    final_docs = [d for d in ranked if d.metadata.get("relevance_score", 0) >= 0.1]

    # Reorder so most-relevant docs are at the start AND end (helps long-context models)
    reordered = LongContextReorder().transform_documents(final_docs)
    context = "\n\n---\n\n".join(d.page_content for d in reordered)

    chain = SYNTHESIS_PROMPT | SYNTHESIS_LLM
    answer = chain.invoke({
        "context": context,
        "question": state["original_question"],
    }).content

    return {**state, "reranked_parents": reordered, "answer": answer}


# ──────────────────────────────────────────────────────────────────────────
# Wire up the 5-stage LangGraph
# ──────────────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(RagState)
    graph.add_node("expand_query", expand_query)
    graph.add_node("retrieve_children", retrieve_children)
    graph.add_node("rerank_children", rerank_children)
    graph.add_node("expand_to_parents", expand_to_parents)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("expand_query")
    graph.add_edge("expand_query", "retrieve_children")
    graph.add_edge("retrieve_children", "rerank_children")
    graph.add_edge("rerank_children", "expand_to_parents")
    graph.add_edge("expand_to_parents", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


if __name__ == "__main__":
    # LangSmith picks up traces automatically when LANGCHAIN_TRACING_V2=true
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "rgbw-chatbot")

    app = build_graph()
    result = app.invoke({
        "original_question": "What did the team decide about the new pricing tier?",
        "expanded_question": None,
        "child_chunks_q1": [],
        "child_chunks_q2": [],
        "parent_chunks": [],
        "reranked_parents": [],
        "answer": None,
    })
    print(result["answer"])
