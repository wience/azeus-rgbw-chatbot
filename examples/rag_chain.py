"""Representative LangChain agent wiring for the RGBW chatbot.

This is a portfolio-documentation snippet showing the dual-retriever pattern
(SQLDocStore + ChromaDB) routed by an LLM agent. The production code at
Azeus Systems is private; this captures the architecture.

Run requires:
    pip install langchain langchain-community langchain-groq chromadb \
                sentence-transformers supabase

Env vars:
    GROQ_API_KEY
    SUPABASE_URL
    SUPABASE_KEY
"""

import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from supabase import create_client


# ──────────────────────────────────────────────────────────────────────────
# 1. LLM — GROQ-hosted Llama 3 70B, picked for low-latency inference
# ──────────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.2,
    api_key=os.environ["GROQ_API_KEY"],
)


# ──────────────────────────────────────────────────────────────────────────
# 2. Vector store — semantic similarity over unstructured docs
# ──────────────────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
vector_store = Chroma(
    collection_name="rgbw_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)


def search_vector_store(query: str) -> str:
    """Top-k semantic retrieval over unstructured documents."""
    results = vector_store.similarity_search(query, k=5)
    return "\n---\n".join(doc.page_content for doc in results)


# ──────────────────────────────────────────────────────────────────────────
# 3. SQL retriever — structured queries over Supabase Postgres
# ──────────────────────────────────────────────────────────────────────────
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"],
)


def search_sql_store(query_description: str) -> str:
    """LLM-generated SQL over the structured doc store.

    In production this used a LangChain SQLDocStore adapter with guarded
    column access. The pattern below is intentionally simplified — the
    real implementation validated against a schema allowlist.
    """
    # Simplified: real version invoked an LLM SQL-gen chain with allow-list
    # validation, then ran the query through Supabase.
    response = supabase.table("documents").select("*").execute()
    return str(response.data[:5])


# ──────────────────────────────────────────────────────────────────────────
# 4. Tool routing — let the agent pick per query
# ──────────────────────────────────────────────────────────────────────────
tools = [
    Tool(
        name="vector_search",
        func=search_vector_store,
        description=(
            "Search unstructured documents (PDFs, internal notes) by "
            "semantic similarity. Use for fuzzy or topical questions."
        ),
    ),
    Tool(
        name="sql_search",
        func=search_sql_store,
        description=(
            "Query structured document records (IDs, names, exact fields) "
            "from Supabase Postgres. Use for precise lookups."
        ),
    ),
]


# ──────────────────────────────────────────────────────────────────────────
# 5. Agent — ReAct pattern, can chain tools as needed
# ──────────────────────────────────────────────────────────────────────────
prompt = PromptTemplate.from_template(
    """You are the RGBW assistant. Answer the user's question by calling
the right tool. Some questions need both tools — call sql_search first
for any specific record IDs, then vector_search for related context.

Tools available: {tools}
Tool names: {tool_names}

Question: {input}
{agent_scratchpad}"""
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)


if __name__ == "__main__":
    result = agent_executor.invoke({
        "input": "Summarize the documents linked to project ID 4521.",
    })
    print(result["output"])
