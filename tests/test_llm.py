
from unittest.mock import MagicMock, patch

from app.llm import get_llm


def test_get_llm_returns_llm():
    """Check that get_llm returns an LLM object."""
    with patch("app.llm.ChatGroq") as mock_chatgroq:
        mock_chatgroq.return_value = MagicMock()

        llm = get_llm()

        assert llm is not None
        mock_chatgroq.assert_called_once()
