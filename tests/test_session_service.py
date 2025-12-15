import pytest
from unittest.mock import Mock, patch
from api.services.session import SessionService
from api.models.document import Session as SessionModel


@pytest.fixture
def mock_session():
    """Mock database session"""
    return Mock()


@pytest.fixture
def session_service(mock_session):
    """Create an instance of SessionService with mocked dependencies"""
    return SessionService(session=mock_session)


class TestSessionService:
    """Test cases for SessionService"""

    def test_create_session_success(self, session_service, mock_session):
        """Test successful session creation"""
        # Arrange
        session_data = {
            'session_id': 'test-session-123',
            'user_id': 'test-user-456',
            'book_id': 'test-book-789',
            'title': 'Test Session Title'
        }

        # Act
        result = session_service.create_session(**session_data)

        # Assert
        assert result['success'] is True
        assert result['session_id'] == 'test-session-123'
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_create_session_missing_required_fields(self, session_service):
        """Test session creation with missing required fields"""
        # Arrange
        session_data = {
            'session_id': 'test-session-123',
            # Missing user_id, book_id, and title
        }

        # Act & Assert
        with pytest.raises(ValueError):
            session_service.create_session(**session_data)

    def test_get_session_by_id_success(self, session_service, mock_session):
        """Test successful session retrieval by ID"""
        # Arrange
        session_id = 'existing-session-123'
        mock_db_session = Mock(spec=SessionModel)
        mock_db_session.session_id = session_id
        mock_db_session.user_id = 'test-user'
        mock_db_session.book_id = 'test-book'
        mock_db_session.title = 'Test Session'
        mock_session.query().filter().first.return_value = mock_db_session

        # Act
        result = session_service.get_session_by_id(session_id)

        # Assert
        assert result is not None
        assert result.session_id == session_id
        assert result.user_id == 'test-user'

    def test_get_session_by_id_not_found(self, session_service, mock_session):
        """Test session retrieval when session doesn't exist"""
        # Arrange
        session_id = 'nonexistent-session'
        mock_session.query().filter().first.return_value = None

        # Act
        result = session_service.get_session_by_id(session_id)

        # Assert
        assert result is None

    def test_update_session_success(self, session_service, mock_session):
        """Test successful session update"""
        # Arrange
        session_id = 'session-to-update'
        update_data = {
            'title': 'Updated Title',
            'metadata': {'last_accessed': '2023-01-01'}
        }

        # Mock existing session
        mock_db_session = Mock(spec=SessionModel)
        mock_db_session.session_id = session_id
        mock_db_session.title = 'Old Title'
        mock_session.query().filter().first.return_value = mock_db_session

        # Act
        result = session_service.update_session(session_id, **update_data)

        # Assert
        assert result['success'] is True
        assert mock_db_session.title == 'Updated Title'
        mock_session.commit.assert_called_once()

    def test_update_session_not_found(self, session_service, mock_session):
        """Test session update when session doesn't exist"""
        # Arrange
        session_id = 'nonexistent-session'
        update_data = {'title': 'New Title'}
        mock_session.query().filter().first.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Session not found"):
            session_service.update_session(session_id, **update_data)

    def test_delete_session_success(self, session_service, mock_session):
        """Test successful session deletion"""
        # Arrange
        session_id = 'session-to-delete'

        # Mock existing session
        mock_db_session = Mock(spec=SessionModel)
        mock_db_session.session_id = session_id
        mock_session.query().filter().first.return_value = mock_db_session

        # Act
        result = session_service.delete_session(session_id)

        # Assert
        assert result['success'] is True
        mock_session.delete.assert_called_once_with(mock_db_session)
        mock_session.commit.assert_called_once()

    def test_delete_session_not_found(self, session_service, mock_session):
        """Test session deletion when session doesn't exist"""
        # Arrange
        session_id = 'nonexistent-session'
        mock_session.query().filter().first.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Session not found"):
            session_service.delete_session(session_id)

    def test_get_sessions_by_user_success(self, session_service, mock_session):
        """Test retrieving all sessions for a user"""
        # Arrange
        user_id = 'test-user'
        mock_sessions = [
            Mock(spec=SessionModel, session_id='session1', user_id=user_id, book_id='book1', title='Session 1'),
            Mock(spec=SessionModel, session_id='session2', user_id=user_id, book_id='book2', title='Session 2'),
        ]
        mock_session.query().filter().all.return_value = mock_sessions

        # Act
        results = session_service.get_sessions_by_user(user_id)

        # Assert
        assert len(results) == 2
        assert all(session.user_id == user_id for session in results)

    def test_get_sessions_by_user_no_results(self, session_service, mock_session):
        """Test retrieving sessions for a user with no sessions"""
        # Arrange
        user_id = 'user-with-no-sessions'
        mock_session.query().filter().all.return_value = []

        # Act
        results = session_service.get_sessions_by_user(user_id)

        # Assert
        assert len(results) == 0

    def test_get_sessions_by_book_success(self, session_service, mock_session):
        """Test retrieving all sessions for a book"""
        # Arrange
        book_id = 'test-book'
        mock_sessions = [
            Mock(spec=SessionModel, session_id='session1', user_id='user1', book_id=book_id, title='Session 1'),
            Mock(spec=SessionModel, session_id='session2', user_id='user2', book_id=book_id, title='Session 2'),
        ]
        mock_session.query().filter().all.return_value = mock_sessions

        # Act
        results = session_service.get_sessions_by_book(book_id)

        # Assert
        assert len(results) == 2
        assert all(session.book_id == book_id for session in results)

    def test_create_session_duplicate_id(self, session_service, mock_session):
        """Test session creation with duplicate ID"""
        # Arrange
        session_data = {
            'session_id': 'duplicate-session',
            'user_id': 'test-user',
            'book_id': 'test-book',
            'title': 'Test Session'
        }

        # Mock that a session with this ID already exists
        existing_session = Mock(spec=SessionModel)
        mock_session.query().filter().first.return_value = existing_session

        # Act & Assert
        with pytest.raises(ValueError, match="Session with this ID already exists"):
            session_service.create_session(**session_data)

    def test_update_session_partial_fields(self, session_service, mock_session):
        """Test updating only some fields of a session"""
        # Arrange
        session_id = 'session-to-update'
        update_data = {
            'title': 'Updated Title'
            # Not updating metadata or other fields
        }

        # Mock existing session
        mock_db_session = Mock(spec=SessionModel)
        mock_db_session.session_id = session_id
        mock_db_session.title = 'Old Title'
        mock_db_session.metadata = {'existing': 'data'}
        mock_session.query().filter().first.return_value = mock_db_session

        # Act
        result = session_service.update_session(session_id, **update_data)

        # Assert
        assert result['success'] is True
        assert mock_db_session.title == 'Updated Title'
        # The metadata should remain unchanged
        assert mock_db_session.metadata == {'existing': 'data'}
        mock_session.commit.assert_called_once()

    def test_get_session_by_id_empty_session_id(self, session_service):
        """Test session retrieval with empty session ID"""
        # Arrange
        session_id = ""

        # Act & Assert
        with pytest.raises(ValueError, match="Session ID cannot be empty"):
            session_service.get_session_by_id(session_id)