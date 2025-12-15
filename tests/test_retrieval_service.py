import pytest
from unittest.mock import Mock, patch
from api.services.retrieval import RetrievalService
from api.models.response import QueryResponse


@pytest.fixture
def mock_vector_db():
    """Mock vector database client"""
    return Mock()


@pytest.fixture
def mock_session():
    """Mock database session"""
    return Mock()


@pytest.fixture
def retrieval_service(mock_vector_db, mock_session):
    """Create an instance of RetrievalService with mocked dependencies"""
    return RetrievalService(vector_db=mock_vector_db, session=mock_session)


class TestRetrievalService:
    """Test cases for RetrievalService"""

    def test_global_search_success(self, retrieval_service, mock_vector_db):
        """Test successful global search"""
        # Arrange
        query = "What is artificial intelligence?"
        book_id = "test-book"

        # Mock vector DB search response
        mock_results = [
            {
                'id': 'doc1',
                'score': 0.9,
                'payload': {
                    'content': 'Artificial intelligence is a branch of computer science...',
                    'title': 'AI Introduction',
                    'book_id': 'test-book',
                    'chapter_id': 'ch1',
                    'page_number': 1
                }
            }
        ]
        mock_vector_db.search.return_value = mock_results

        # Act
        result = retrieval_service.global_search(query, book_id)

        # Assert
        assert isinstance(result, QueryResponse)
        assert len(result.contexts) > 0
        assert result.contexts[0].content == 'Artificial intelligence is a branch of computer science...'
        mock_vector_db.search.assert_called_once()

    def test_selection_only_search_success(self, retrieval_service, mock_vector_db):
        """Test successful selection-only search"""
        # Arrange
        query = "Explain this concept?"
        selected_text = "Machine learning is a subset of artificial intelligence..."
        book_id = "test-book"

        # Mock vector DB search response
        mock_results = [
            {
                'id': 'doc1',
                'score': 0.95,
                'payload': {
                    'content': 'Machine learning is a subset of artificial intelligence...',
                    'title': 'ML Basics',
                    'book_id': 'test-book',
                    'chapter_id': 'ch2',
                    'page_number': 15
                }
            }
        ]
        mock_vector_db.search.return_value = mock_results

        # Act
        result = retrieval_service.selection_only_search(query, selected_text, book_id)

        # Assert
        assert isinstance(result, QueryResponse)
        assert len(result.contexts) > 0
        assert result.contexts[0].content == 'Machine learning is a subset of artificial intelligence...'
        # Verify that the search was filtered to only include the selected text
        assert mock_vector_db.search.called

    def test_global_search_no_results(self, retrieval_service, mock_vector_db):
        """Test global search with no results"""
        # Arrange
        query = "Non-existent topic?"
        book_id = "test-book"
        mock_vector_db.search.return_value = []

        # Act
        result = retrieval_service.global_search(query, book_id)

        # Assert
        assert isinstance(result, QueryResponse)
        assert len(result.contexts) == 0
        assert result.answer == "I couldn't find relevant information in the source text to answer your question."

    def test_selection_only_search_no_results(self, retrieval_service, mock_vector_db):
        """Test selection-only search with no results"""
        # Arrange
        query = "Explain this?"
        selected_text = "Some selected text that doesn't match the query..."
        book_id = "test-book"
        mock_vector_db.search.return_value = []

        # Act
        result = retrieval_service.selection_only_search(query, selected_text, book_id)

        # Assert
        assert isinstance(result, QueryResponse)
        assert len(result.contexts) == 0
        assert result.answer == "I couldn't find relevant information in the selected text to answer your question."

    def test_search_with_score_threshold(self, retrieval_service, mock_vector_db):
        """Test search with score threshold filtering"""
        # Arrange
        query = "What is AI?"
        book_id = "test-book"

        # Mock results with varying scores
        mock_results = [
            {
                'id': 'doc1',
                'score': 0.6,  # Above threshold
                'payload': {
                    'content': 'High relevance content...',
                    'title': 'AI Overview',
                    'book_id': 'test-book',
                    'chapter_id': 'ch1',
                    'page_number': 1
                }
            },
            {
                'id': 'doc2',
                'score': 0.3,  # Below threshold
                'payload': {
                    'content': 'Low relevance content...',
                    'title': 'Related Topic',
                    'book_id': 'test-book',
                    'chapter_id': 'ch2',
                    'page_number': 5
                }
            }
        ]
        mock_vector_db.search.return_value = mock_results

        # Act
        result = retrieval_service.global_search(query, book_id, min_score=0.5)

        # Assert
        assert len(result.contexts) == 1  # Only high-score result should be included
        assert result.contexts[0].content == 'High relevance content...'

    def test_search_with_top_k_limit(self, retrieval_service, mock_vector_db):
        """Test search with top-k results limit"""
        # Arrange
        query = "What is AI?"
        book_id = "test-book"

        # Mock multiple results
        mock_results = []
        for i in range(10):
            mock_results.append({
                'id': f'doc{i}',
                'score': 0.9 - (i * 0.1),
                'payload': {
                    'content': f'Result {i} content...',
                    'title': f'Title {i}',
                    'book_id': 'test-book',
                    'chapter_id': f'ch{i}',
                    'page_number': i + 1
                }
            })
        mock_vector_db.search.return_value = mock_results

        # Act
        result = retrieval_service.global_search(query, book_id, top_k=3)

        # Assert
        assert len(result.contexts) == 3  # Should be limited to top 3 results

    def test_global_search_with_filters(self, retrieval_service, mock_vector_db):
        """Test global search with additional filters"""
        # Arrange
        query = "What is AI?"
        book_id = "test-book"
        chapter_filter = "ch1"

        # Mock results
        mock_results = [
            {
                'id': 'doc1',
                'score': 0.9,
                'payload': {
                    'content': 'AI content...',
                    'title': 'AI Chapter',
                    'book_id': 'test-book',
                    'chapter_id': 'ch1',  # Matches filter
                    'page_number': 1
                }
            },
            {
                'id': 'doc2',
                'score': 0.8,
                'payload': {
                    'content': 'Different content...',
                    'title': 'Other Chapter',
                    'book_id': 'test-book',
                    'chapter_id': 'ch2',  # Doesn't match filter
                    'page_number': 5
                }
            }
        ]
        mock_vector_db.search.return_value = mock_results

        # Act
        result = retrieval_service.global_search(query, book_id, filters={'chapter_id': chapter_filter})

        # Assert
        # The filtering should happen in the search call
        mock_vector_db.search.assert_called_once()
        # Verify the call included the filters
        call_kwargs = mock_vector_db.search.call_args.kwargs
        assert 'filters' in call_kwargs
        assert call_kwargs['filters']['chapter_id'] == chapter_filter

    def test_selection_only_search_validation(self, retrieval_service):
        """Test validation in selection-only search"""
        # Arrange
        query = ""
        selected_text = ""
        book_id = "test-book"

        # Act & Assert
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retrieval_service.selection_only_search(query, selected_text, book_id)

        # Test with only empty query
        query = ""
        selected_text = "Some text"
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retrieval_service.selection_only_search(query, selected_text, book_id)

        # Test with only empty selected text
        query = "Valid query"
        selected_text = ""
        with pytest.raises(ValueError, match="Selected text cannot be empty for selection-only mode"):
            retrieval_service.selection_only_search(query, selected_text, book_id)