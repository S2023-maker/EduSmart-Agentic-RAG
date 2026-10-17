import os
from pathlib import Path

from dotenv import load_dotenv


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")


# -----------------------------
# LLM Configuration
# -----------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama-3.1-8b-instant"
)

LLM_TEMPERATURE = float(
    os.getenv("LLM_TEMPERATURE", "0.2")
)


# -----------------------------
# Embedding Configuration
# -----------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# RAG Configuration
# -----------------------------

TOP_K = int(
    os.getenv("TOP_K", "4")
)

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "800")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "120")
)


# -----------------------------
# Directory Configuration
# -----------------------------

CURRICULUM_DIR = BASE_DIR / "curriculum"

VECTORSTORE_DIR = (
    BASE_DIR / "vectorstore" / "faiss_index"
)

EVALUATION_DIR = BASE_DIR / "evaluation"


def validate_config():
    """Validate required configuration."""

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to the .env file."
        )

    if not CURRICULUM_DIR.exists():
        raise FileNotFoundError(
            f"Curriculum directory not found: {CURRICULUM_DIR}"
        )