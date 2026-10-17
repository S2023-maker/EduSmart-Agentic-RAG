
from unittest.mock import MagicMock, patch

from app.rag import answer_question


def test_empty_question_returns_message():
    """An empty question should not call retrieval or the LLM."""
    result = answer_question("   ")

    assert result["answer"] == "Please enter a question."
    assert result["sources"] == []
    assert result["documents"] == []


def test_answer_question_uses_retrieved_context():
    """A question with retrieved documents should be answered by the LLM."""
    fake_document = MagicMock()
    fake_document.metadata = {
        "source": "science_chapter.pdf",
        "page": 2,
    }
    fake_document.page_content = (
        "Plants use sunlight, water, and carbon dioxide to make food."
    )

    fake_response = MagicMock()
    fake_response.content = (
        "Photosynthesis is the process plants use to make food."
    )

    with (
        patch("app.rag.retrieve_documents", return_value=[fake_document]),
        patch("app.rag.get_llm") as mock_get_llm,
    ):
        mock_get_llm.return_value.invoke.return_value = fake_response

        result = answer_question("What is photosynthesis?")

    assert "Photosynthesis" in result["answer"]
    assert result["sources"] == ["science_chapter.pdf"]
    assert len(result["documents"]) == 1
    mock_get_llm.return_value.invoke.assert_called_once()
