
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)

SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".txt",
    ".docx",
    ".csv",
    ".md",
]


def load_document(file_path):
    """
    Load a supported document and return a list of
    LangChain Document objects.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    elif extension == ".csv":
        loader = CSVLoader(str(path), encoding="utf-8")

    elif extension == ".md":
        loader = TextLoader(str(path), encoding="utf-8")

    else:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    return loader.load()
