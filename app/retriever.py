
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PROJECT PATH
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
# CURRICULUM VECTORSTORE
# ============================================================

def get_vectorstore():
    index_file = FAISS_DIR / "index.faiss"
    metadata_file = FAISS_DIR / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"FAISS index files were not found in {FAISS_DIR}. "
            "Please run 'python ingest.py' first."
        )

    return FAISS.load_local(
        str(FAISS_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# ============================================================
# GENERIC RETRIEVER
# ============================================================

def get_retriever(k: int = 4):
    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ============================================================
# CURRICULUM RETRIEVAL
# ============================================================

def retrieve_documents(
    query: str,
    k: int = 4,
    vectorstore=None,
):
    """
    Retrieve relevant document chunks.

    MMR tries to return relevant but less-redundant passages.
    If MMR is unavailable or fails, similarity search is used.
    """

    if not isinstance(query, str) or not query.strip():
        return []

    if k < 1:
        return []

    store = vectorstore if vectorstore is not None else get_vectorstore()

    # First attempt: MMR retrieval.
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
        print(f"MMR retrieval unavailable; using similarity search: {exc}")

    # Fallback: ordinary similarity retrieval.
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    return retriever.invoke(query)
    
