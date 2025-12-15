import pytest
from unittest.mock import Mock, patch, MagicMock
from api.services.chat import ChatService
from api.models.request import QueryRequest
from api.models.response import QueryResponse
from api.models.document import Session as SessionModel
from api.config.settings import settings


@pytest.fixture
def mock_retrieval_service():
    """Mock retrieval service"""
    return Mock()


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service"""
    return Mock()


@pytest.fixture
def mock_session():
    """Mock database session"""
    return Mock()


@pytest.fixture
def chat_service(mock_retrieval_service, mock_embedding_service, mock_session):
    """Create an instance of ChatService with mocked dependencies"""
    return ChatService(
        retrieval_service=mock_retrieval_service,
        embedding_service=mock_embedding_service,
        session=mock_session
    )


class TestChatService:
    """Test cases for ChatService"""

    def test_get_answer_global_mode_success(self, chat_service, mock_retrieval_service):
        """Test getting answer in global mode"""
        # Arrange
        query_request = QueryRequest(
            query="What is artificial intelligence?",
            book_id="test-book",
            mode="GLOBAL"
        )

        # Mock retrieval response
        mock_context_response = Mock()
        mock_context_response.contexts = [
            Mock(content="Artificial intelligence is a branch of computer science...", score=0.9)
        ]
        mock_context_response.answer = "Mock answer from LLM"

        mock_retrieval_service.global_search.return_value = mock_context_response

        # Act
        result = chat_service.get_answer(query_request)

        # Assert
        assert isinstance(result, QueryResponse)
        assert result.answer is not None
        assert len(result.contexts) > 0
        mock_retrieval_service.global_search.assert_called_once()

    def test_get_answer_selection_only_mode_success(self, chat_service, mock_retrieval_service):
        """Test getting answer in selection-only mode"""
        # Arrange
        query_request = QueryRequest(
            query="Explain this concept?",
            book_id="test-book",
            mode="SELECTION_ONLY",
            selected_text="Machine learning is a subset of artificial intelligence..."
        )

        # Mock retrieval response
        mock_context_response = Mock()
        mock_context_response.contexts = [
            Mock(content="Machine learning is a subset of artificial intelligence...", score=0.95)
        ]
        mock_context_response.answer = "Explained concept based on selected text"

        mock_retrieval_service.selection_only_search.return_value = mock_context_response

        # Act
        result = chat_service.get_answer(query_request)

        # Assert
        assert isinstance(result, QueryResponse)
        assert result.answer is not None
        assert len(result.contexts) > 0
        mock_retrieval_service.selection_only_search.assert_called_once()

    def test_get_answer_not_found_global_mode(self, chat_service, mock_retrieval_service):
        """Test getting answer when content not found in global mode"""
        # Arrange
        query_request = QueryRequest(
            query="What is quantum computing?",
            book_id="test-book",
            mode="GLOBAL"
        )

        # Mock retrieval response with no contexts
        mock_context_response = Mock()
        mock_context_response.contexts = []
        mock_context_response.answer = "I couldn't find relevant information in the source text to answer your question."

        mock_retrieval_service.global_search.return_value = mock_context_response

        # Act
        result = chat_service.get_answer(query_request)

        # Assert
        assert isinstance(result, QueryResponse)
        assert "couldn't find relevant information" in result.answer.lower()
        assert len(result.contexts) == 0

    def test_get_answer_not_found_selection_only_mode(self, chat_service, mock_retrieval_service):
        """Test getting answer when content not found in selection-only mode"""
        # Arrange
        query_request = QueryRequest(
            query="Explain this?",
            book_id="test-book",
            mode="SELECTION_ONLY",
            selected_text="Some unrelated text..."
        )

        # Mock retrieval response with no contexts
        mock_context_response = Mock()
        mock_context_response.contexts = []
        mock_context_response.answer = "I couldn't find relevant information in the selected text to answer your question."

        mock_retrieval_service.selection_only_search.return_value = mock_context_response

        # Act
        result = chat_service.get_answer(query_request)

        # Assert
        assert isinstance(result, QueryResponse)
        assert "couldn't find relevant information in the selected text" in result.answer.lower()
        assert len(result.contexts) == 0

    @patch('api.services.chat.openai_client')
    def test_grounding_validation_passes(self, mock_openai_client, chat_service):
        """Test grounding validation passes when response is supported by context"""
        # Arrange
        response = "Artificial intelligence is a branch of computer science."
        contexts = [
            Mock(content="Artificial intelligence is a branch of computer science that aims to create software or machines that exhibit human-like intelligence.")
        ]

        # Mock the OpenAI call to return validation that confirms grounding
        mock_completion = Mock()
        mock_completion.choices[0].message.content = '{"is_supported": true, "confidence_score": 0.9}'
        mock_openai_client.chat.completions.create.return_value = mock_completion

        # Act
        result = chat_service.validate_grounding(response, contexts)

        # Assert
        assert result['is_supported'] is True
        assert result['confidence_score'] > 0.5
        mock_openai_client.chat.completions.create.assert_called_once()

    @patch('api.services.chat.openai_client')
    def test_grounding_validation_fails(self, mock_openai_client, chat_service):
        """Test grounding validation fails when response is not supported by context"""
        # Arrange
        response = "Quantum computing uses quantum bits called qubits."
        contexts = [
            Mock(content="Artificial intelligence is a branch of computer science that aims to create software or machines that exhibit human-like intelligence.")
        ]

        # Mock the OpenAI call to return validation that indicates no support
        mock_completion = Mock()
        mock_completion.choices[0].message.content = '{"is_supported": false, "confidence_score": 0.2}'
        mock_openai_client.chat.completions.create.return_value = mock_completion

        # Act
        result = chat_service.validate_grounding(response, contexts)

        # Assert
        assert result['is_supported'] is False
        assert result['confidence_score'] < 0.5

    def test_create_session_success(self, chat_service, mock_session):
        """Test successful session creation"""
        # Arrange
        session_data = {
            'session_id': 'test-session-123',
            'user_id': 'test-user',
            'book_id': 'test-book',
            'title': 'Test Session'
        }

        # Act
        result = chat_service.create_session(session_data)

        # Assert
        assert result['success'] is True
        assert result['session_id'] == 'test-session-123'
        mock_session.add.assert_called_once()

    def test_get_session_success(self, chat_service, mock_session):
        """Test successful session retrieval"""
        # Arrange
        session_id = 'existing-session-123'
        mock_db_session = Mock(spec=SessionModel)
        mock_db_session.session_id = session_id
        mock_db_session.user_id = 'test-user'
        mock_db_session.book_id = 'test-book'
        mock_session.query().filter().first.return_value = mock_db_session

        # Act
        result = chat_service.get_session(session_id)

        # Assert
        assert result is not None
        assert result.session_id == session_id

    def test_get_session_not_found(self, chat_service, mock_session):
        """Test session retrieval when session doesn't exist"""
        # Arrange
        session_id = 'nonexistent-session'
        mock_session.query().filter().first.return_value = None

        # Act
        result = chat_service.get_session(session_id)

        # Assert
        assert result is None

    @patch('api.services.chat.openai_client')
    def test_answer_generation_with_context(self, mock_openai_client, chat_service, mock_retrieval_service):
        """Test that answers are generated using the retrieved context"""
        # Arrange
        query_request = QueryRequest(
            query="What are the applications of AI?",
            book_id="test-book",
            mode="GLOBAL"
        )

        # Mock retrieval response
        mock_context_response = Mock()
        mock_context_response.contexts = [
            Mock(content="AI has applications in healthcare, finance, and transportation.", score=0.9)
        ]
        mock_context_response.answer = "Based on the retrieved context, AI has applications in healthcare, finance, and transportation."

        mock_retrieval_service.global_search.return_value = mock_context_response

        # Mock the OpenAI completion response
        mock_completion = Mock()
        mock_completion.choices[0].message.content = "Based on the retrieved context, AI has applications in healthcare, finance, and transportation."
        mock_openai_client.chat.completions.create.return_value = mock_completion

        # Act
        result = chat_service.get_answer(query_request)

        # Assert
        assert "applications" in result.answer.lower()
        assert len(result.contexts) == 1
        # Verify that the OpenAI client was called to generate the answer
        mock_openai_client.chat.completions.create.assert_called()

    def test_validate_input_success(self, chat_service):
        """Test input validation passes for valid inputs"""
        # Arrange
        query = "What is AI?"
        book_id = "valid-book-id"

        # Act & Assert (should not raise any exceptions)
        try:
            chat_service._validate_input(query, book_id)
        except ValueError:
            pytest.fail("_validate_input raised ValueError unexpectedly!")

    def test_validate_input_empty_query(self, chat_service):
        """Test input validation fails for empty query"""
        # Arrange
        query = ""
        book_id = "valid-book-id"

        # Act & Assert
        with pytest.raises(ValueError, match="Query cannot be empty"):
            chat_service._validate_input(query, book_id)

    def test_validate_input_empty_book_id(self, chat_service):
        """Test input validation fails for empty book ID"""
        # Arrange
        query = "Valid query"
        book_id = ""

        # Act & Assert
        with pytest.raises(ValueError, match="Book ID cannot be empty"):
            chat_service._validate_input(query, book_id)

    def test_get_answer_invalid_mode(self, chat_service):
        """Test getting answer with invalid mode raises exception"""
        # Arrange
        query_request = QueryRequest(
            query="What is AI?",
            book_id="test-book",
            mode="INVALID_MODE"
        )

        # Act & Assert
        with pytest.raises(ValueError):
            chat_service.get_answer(query_request)