
from pathlib import Path

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

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# BUILT-IN CURRICULUM VECTORSTORE
# ============================================================

def get_vectorstore():
    index_file = FAISS_DIR / "index.faiss"
    metadata_file = FAISS_DIR / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"FAISS index files not found in {FAISS_DIR}. "
            "Please run 'python ingest.py' first."
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
    store = get_vectorstore()

    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ============================================================
# SEARCH ONE VECTORSTORE
# ============================================================

def _search_store(store, query: str, k: int):
    """Try MMR retrieval first, then fall back to similarity."""

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

def retrieve_documents(
    query: str,
    k: int = 4,
    vectorstore=None,
):
    """
    Retrieve from the built-in curriculum and, when supplied,
    the user's uploaded-document vectorstore.

    Uploaded-document results are considered first, followed by
    curriculum results. Duplicate chunks are removed.
    """

    if not isinstance(query, str) or not query.strip():
        return []

    if not isinstance(k, int) or k < 1:
        return []

    uploaded_documents = []
    curriculum_documents = []

    # Search the built-in curriculum.
    try:
        curriculum_store = get_vectorstore()
        curriculum_documents = _search_store(
            curriculum_store, query, k
        )
    except Exception as exc:
        print(f"Curriculum retrieval unavailable: {exc}")

    # Search the uploaded-document vectorstore if provided.
    if vectorstore is not None:
        uploaded_documents = _search_store(
            vectorstore, query, k
        )

    # Prioritize uploaded documents, then add curriculum results.
    combined_documents = []
    seen = set()

    for document in uploaded_documents + curriculum_documents:
        metadata = getattr(document, "metadata", {}) or {}
        content = getattr(document, "page_content", "")

        # Identify duplicate chunks using source, location and content.
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

    # Return up to k results from each knowledge source.
    return combined_documents[: 2 * k]
