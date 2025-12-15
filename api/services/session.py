from typing import Optional
import logging
from datetime import datetime
from sqlalchemy.orm import Session as DbSession
from api.models.document import Session as SessionModel
from api.config.database import SessionLocal
from api.config.settings import settings

# Set up logging
logger = logging.getLogger(__name__)

class SessionService:
    """
    Service for managing chat sessions in the database.
    """

    def __init__(self):
        self.db: DbSession = SessionLocal()

    async def create_session(self, user_id: Optional[str] = None, book_id: Optional[str] = None) -> SessionModel:
        """
        Create a new chat session in the database.
        """
        try:
            session = SessionModel(
                id=f"session_{datetime.now().timestamp()}",
                user_id=user_id,
                book_id=book_id,
                start_time=datetime.now(),
                last_activity=datetime.now(),
                active=True
            )

            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)

            logger.info(f"Created new session in database: {session.id}")
            return session

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating session: {str(e)}")
            raise

    async def get_session(self, session_id: str) -> Optional[SessionModel]:
        """
        Retrieve a session from the database by ID.
        """
        try:
            session = self.db.query(SessionModel).filter(SessionModel.id == session_id).first()
            return session
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {str(e)}")
            return None

    async def update_session_activity(self, session_id: str):
        """
        Update the last activity timestamp for a session.
        """
        try:
            session = await self.get_session(session_id)
            if session:
                session.last_activity = datetime.now()
                self.db.commit()
                logger.info(f"Updated activity for session: {session_id}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating session activity: {str(e)}")
            raise

    async def deactivate_session(self, session_id: str):
        """
        Mark a session as inactive.
        """
        try:
            session = await self.get_session(session_id)
            if session:
                session.active = False
                session.last_activity = datetime.now()
                self.db.commit()
                logger.info(f"Deactivated session: {session_id}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deactivating session: {str(e)}")
            raise

    async def get_active_sessions(self, user_id: Optional[str] = None) -> list:
        """
        Get all active sessions, optionally filtered by user_id.
        """
        try:
            query = self.db.query(SessionModel).filter(SessionModel.active == True)

            if user_id:
                query = query.filter(SessionModel.user_id == user_id)

            sessions = query.all()
            return sessions
        except Exception as e:
            logger.error(f"Error retrieving active sessions: {str(e)}")
            return []

    def close(self):
        """
        Close the database session.
        """
        self.db.close()

# Create a singleton instance
session_service = SessionService()