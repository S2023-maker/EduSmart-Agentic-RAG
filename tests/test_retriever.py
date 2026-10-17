
from unittest.mock import MagicMock

from app.retriever import retrieve_documents


def test_empty_query_returns_no_documents():
    """An empty query should return an empty list."""
    result = retrieve_documents("   ")

    assert result == []


def test_invalid_k_returns_no_documents():
    """A k value below 1 should return an empty list."""
    result = retrieve_documents("What is photosynthesis?", k=0)

    assert result == []


def test_retrieval_uses_provided_vectorstore():
    """Retrieval should use a supplied vector store."""
    fake_document = MagicMock()
    fake_vectorstore = MagicMock()

    fake_vectorstore.as_retriever.return_value.invoke.return_value = [
        fake_document
    ]

    result = retrieve_documents(
        "How do CFCs affect the ozone layer?",
        k=3,
        vectorstore=fake_vectorstore,
    )

    assert result == [fake_document]
    fake_vectorstore.as_retriever.assert_called_once()
