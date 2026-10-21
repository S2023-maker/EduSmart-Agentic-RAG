from pathlib import Path
from functools import lru_cache

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "faiss_index"


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load the embedding model once per Streamlit server process."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# BUILT-IN CURRICULUM VECTORSTORE
# ============================================================

@st.cache_resource(show_spinner="Loading curriculum search index...")
def get_vectorstore():
    """Load and cache the built-in curriculum FAISS index."""
    index_file = FAISS_DIR / "index.faiss"
    metadata_file = FAISS_DIR / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"FAISS index files not found in {FAISS_DIR}. "
            "Run 'python ingest.py' from the project root first."
        )

    return FAISS.load_local(
        str(FAISS_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# ============================================================
# GENERIC CURRICULUM RETRIEVER
# ============================================================

def get_retriever(k: int = 4):
    """Create a retriever for the built-in curriculum index."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ============================================================
# SEARCH ONE VECTORSTORE
# ============================================================

def _search_store(store, query: str, k: int):
    """Try MMR retrieval first; fall back to similarity search."""
    if store is None:
        return []

    try:
        retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": max(12, k * 3),
                "lambda_mult": 0.65,
            },
        )
        documents = retriever.invoke(query)
        if documents:
            return documents
    except Exception as exc:
        print(f"MMR retrieval fallback: {exc}")

    try:
        return store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        ).invoke(query)
    except Exception as exc:
        print(f"Similarity retrieval failed: {exc}")
        return []


# ============================================================
# COMBINED RETRIEVAL
# ============================================================

def retrieve_documents(query: str, k: int = 4, vectorstore=None):
    """
    Search the uploaded-document index and built-in curriculum.

    Uploaded-document chunks are placed first, followed by curriculum
    chunks. Exact duplicate chunks are removed. Returns up to 2*k
    documents in total, so each source can contribute up to k chunks.
    """
    if not isinstance(query, str) or not query.strip():
        return []

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        return []

    uploaded_documents = []
    curriculum_documents = []

    # Curriculum search should not prevent uploaded-document search.
    try:
        curriculum_documents = _search_store(get_vectorstore(), query.strip(), k)
    except Exception as exc:
        print(f"Curriculum retrieval unavailable: {exc}")

    if vectorstore is not None:
        uploaded_documents = _search_store(vectorstore, query.strip(), k)

    combined_documents = []
    seen = set()

    for document in uploaded_documents + curriculum_documents:
        metadata = getattr(document, "metadata", {}) or {}
        content = getattr(document, "page_content", "") or ""
        key = (
            str(metadata.get("source", "")),
            str(metadata.get("page", "")),
            str(metadata.get("slide", "")),
            str(metadata.get("sheet", "")),
            content.strip(),
        )
        if key not in seen:
            seen.add(key)
            combined_documents.append(document)

    return combined_documents[: 2 * k]
