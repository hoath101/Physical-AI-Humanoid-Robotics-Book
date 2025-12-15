import pytest
from unittest.mock import Mock, patch, MagicMock
from api.services.ingestion import IngestionService
from api.models.document import BookContent
from api.config.settings import settings


@pytest.fixture
def mock_vector_db():
    """Mock vector database client"""
    return Mock()


@pytest.fixture
def mock_session():
    """Mock database session"""
    return Mock()


@pytest.fixture
def ingestion_service(mock_vector_db, mock_session):
    """Create an instance of IngestionService with mocked dependencies"""
    return IngestionService(vector_db=mock_vector_db, session=mock_session)


class TestIngestionService:
    """Test cases for IngestionService"""

    def test_create_content_success(self, ingestion_service, mock_vector_db, mock_session):
        """Test successful content creation"""
        # Arrange
        content_data = {
            'title': 'Test Chapter',
            'content': 'This is test content.',
            'book_id': 'test-book',
            'chapter_id': 'ch1',
            'page_number': 1,
            'paragraph_id': 'p1'
        }

        # Mock the vector DB response
        mock_vector_db.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        # Act
        result = ingestion_service.create(content_data)

        # Assert
        assert result['success'] is True
        assert result['document_id'] is not None
        assert 'vector_id' in result
        mock_vector_db.store_vectors.assert_called_once()
        mock_session.add.assert_called_once()

    def test_create_content_empty_content(self, ingestion_service):
        """Test content creation with empty content"""
        # Arrange
        content_data = {
            'title': 'Test Chapter',
            'content': '',
            'book_id': 'test-book',
            'chapter_id': 'ch1',
            'page_number': 1,
            'paragraph_id': 'p1'
        }

        # Act & Assert
        with pytest.raises(ValueError):
            ingestion_service.create(content_data)

    def test_update_content_success(self, ingestion_service, mock_vector_db):
        """Test successful content update"""
        # Arrange
        document_id = 'doc123'
        content_data = {
            'title': 'Updated Chapter',
            'content': 'Updated content here.',
            'book_id': 'test-book',
            'chapter_id': 'ch1',
            'page_number': 1,
            'paragraph_id': 'p1'
        }

        # Mock the vector DB response
        mock_vector_db.generate_embeddings.return_value = [[0.4, 0.5, 0.6]]

        # Act
        result = ingestion_service.update(document_id, content_data)

        # Assert
        assert result['success'] is True
        assert result['document_id'] == document_id
        mock_vector_db.update_vectors.assert_called_once()

    def test_delete_content_success(self, ingestion_service, mock_vector_db, mock_session):
        """Test successful content deletion"""
        # Arrange
        document_id = 'doc123'
        vector_id = 'vec123'

        # Mock the query to find the document
        mock_book_content = Mock(spec=BookContent)
        mock_book_content.vector_id = vector_id
        mock_session.query().filter().first.return_value = mock_book_content

        # Act
        result = ingestion_service.delete(document_id)

        # Assert
        assert result['success'] is True
        mock_session.delete.assert_called_once()
        mock_vector_db.delete_vectors.assert_called_once_with([vector_id])

    def test_delete_content_not_found(self, ingestion_service, mock_session):
        """Test content deletion when document doesn't exist"""
        # Arrange
        document_id = 'nonexistent-doc'
        mock_session.query().filter().first.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Document not found"):
            ingestion_service.delete(document_id)

    @patch('api.services.ingestion.text_splitter')
    def test_ingest_large_content_chunks(self, mock_text_splitter, ingestion_service, mock_vector_db):
        """Test ingestion of large content that gets split into chunks"""
        # Arrange
        large_content = "This is a large content. " * 100
        content_data = {
            'title': 'Large Chapter',
            'content': large_content,
            'book_id': 'test-book',
            'chapter_id': 'ch1',
            'page_number': 1,
            'paragraph_id': 'p1'
        }

        # Mock text splitter to return multiple chunks
        chunks = ['chunk1', 'chunk2', 'chunk3']
        mock_text_splitter.split_text.return_value = chunks

        # Mock embeddings for each chunk
        mock_vector_db.generate_embeddings.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]

        # Act
        result = ingestion_service.create(content_data)

        # Assert
        assert result['success'] is True
        assert result['chunks_processed'] == 3
        assert mock_vector_db.store_vectors.call_count == 3

    def test_create_content_with_metadata(self, ingestion_service, mock_vector_db):
        """Test content creation with additional metadata"""
        # Arrange
        content_data = {
            'title': 'Test Chapter',
            'content': 'This is test content with metadata.',
            'book_id': 'test-book',
            'chapter_id': 'ch1',
            'page_number': 1,
            'paragraph_id': 'p1',
            'metadata': {'author': 'Test Author', 'year': 2023}
        }

        mock_vector_db.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        # Act
        result = ingestion_service.create(content_data)

        # Assert
        assert result['success'] is True
        mock_vector_db.store_vectors.assert_called_once()
        # Check that metadata is included in the call
        call_args = mock_vector_db.store_vectors.call_args
        assert 'metadata' in call_args[1]['payload'][0]