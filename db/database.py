"""
Database manager for the Physical AI & Humanoid Robotics Book RAG Chatbot.
Handles database operations for storing questions, answers, and interactions.
"""

import asyncio
import asyncpg
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from config.ingestion_config import get_config_value
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Asynchronous database manager for PostgreSQL using asyncpg.
    Handles storing and retrieving question-answer pairs, user interactions, and analytics.
    """

    def __init__(self):
        self.pool = None
        self.connection_string = get_config_value(
            'DATABASE_URL',
            'postgresql://user:password@localhost:5432/book_rag_chatbot'
        )

    async def connect(self):
        """Establish connection to the database."""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                dsn=self.connection_string,
                min_size=1,
                max_size=10,
                command_timeout=60
            )

            # Create required tables if they don't exist
            await self._create_tables()

            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    async def close(self):
        """Close the database connection."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection closed")

    async def _create_tables(self):
        """Create required database tables if they don't exist."""
        async with self.pool.acquire() as conn:
            # Create questions_answers table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS questions_answers (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    document_ids TEXT[], -- Array of document IDs used in the response
                    selected_text_used BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create user_interactions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255),
                    user_id VARCHAR(255),
                    question_id INTEGER REFERENCES questions_answers(id),
                    interaction_type VARCHAR(50), -- 'query', 'feedback', 'followup'
                    interaction_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create feedback table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    question_id INTEGER REFERENCES questions_answers(id),
                    user_id VARCHAR(255),
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5), -- 1-5 star rating
                    comment TEXT,
                    helpful BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create analytics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for better performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_questions_answers_created_at
                ON questions_answers(created_at);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_questions_answers_selected_text_used
                ON questions_answers(selected_text_used);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_interactions_session_id
                ON user_interactions(session_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_question_id
                ON feedback(question_id);
            """)

            logger.info("Database tables created successfully")

    async def save_question_answer(
        self,
        question: str,
        answer: str,
        document_ids: List[str],
        selected_text_used: bool = False
    ) -> int:
        """
        Save a question-answer pair to the database.

        Args:
            question: The user's question
            answer: The AI's answer
            document_ids: List of document IDs used in the response
            selected_text_used: Whether selected text mode was used

        Returns:
            ID of the inserted record
        """
        try:
            async with self.pool.acquire() as conn:
                query = """
                    INSERT INTO questions_answers
                    (question, answer, document_ids, selected_text_used)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                """
                result = await conn.fetchval(
                    query,
                    question,
                    answer,
                    document_ids,
                    selected_text_used
                )

                logger.info(f"Saved question-answer pair with ID: {result}")
                return result
        except Exception as e:
            logger.error(f"Error saving question-answer pair: {e}")
            raise

    async def get_question_answer(self, qa_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific question-answer pair by ID.

        Args:
            qa_id: ID of the question-answer pair

        Returns:
            Dictionary with question-answer data or None if not found
        """
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT id, question, answer, document_ids, selected_text_used, created_at
                    FROM questions_answers
                    WHERE id = $1
                """
                record = await conn.fetchrow(query, qa_id)

                if record:
                    return {
                        'id': record['id'],
                        'question': record['question'],
                        'answer': record['answer'],
                        'document_ids': record['document_ids'],
                        'selected_text_used': record['selected_text_used'],
                        'created_at': record['created_at']
                    }
                return None
        except Exception as e:
            logger.error(f"Error retrieving question-answer pair: {e}")
            raise

    async def get_recent_questions_answers(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent question-answer pairs.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of question-answer dictionaries
        """
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT id, question, answer, document_ids, selected_text_used, created_at
                    FROM questions_answers
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                """
                records = await conn.fetch(query, limit, offset)

                return [
                    {
                        'id': record['id'],
                        'question': record['question'],
                        'answer': record['answer'],
                        'document_ids': record['document_ids'],
                        'selected_text_used': record['selected_text_used'],
                        'created_at': record['created_at']
                    }
                    for record in records
                ]
        except Exception as e:
            logger.error(f"Error retrieving recent question-answer pairs: {e}")
            raise

    async def save_user_interaction(
        self,
        session_id: str,
        user_id: str,
        question_id: int,
        interaction_type: str,
        interaction_data: Dict[str, Any]
    ) -> int:
        """
        Save a user interaction to the database.

        Args:
            session_id: Unique session identifier
            user_id: User identifier
            question_id: Related question-answer pair ID
            interaction_type: Type of interaction ('query', 'feedback', 'followup')
            interaction_data: Additional interaction data

        Returns:
            ID of the inserted record
        """
        try:
            async with self.pool.acquire() as conn:
                query = """
                    INSERT INTO user_interactions
                    (session_id, user_id, question_id, interaction_type, interaction_data)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                """
                result = await conn.fetchval(
                    query,
                    session_id,
                    user_id,
                    question_id,
                    interaction_type,
                    json.dumps(interaction_data)
                )

                logger.info(f"Saved user interaction with ID: {result}")
                return result
        except Exception as e:
            logger.error(f"Error saving user interaction: {e}")
            raise

    async def save_feedback(
        self,
        question_id: int,
        user_id: str,
        rating: int,
        comment: Optional[str] = None,
        helpful: Optional[bool] = None
    ) -> int:
        """
        Save feedback for a question-answer pair.

        Args:
            question_id: ID of the question-answer pair being rated
            user_id: User identifier
            rating: Rating from 1-5
            comment: Optional feedback comment
            helpful: Whether the response was helpful

        Returns:
            ID of the inserted record
        """
        try:
            if rating < 1 or rating > 5:
                raise ValueError("Rating must be between 1 and 5")

            async with self.pool.acquire() as conn:
                query = """
                    INSERT INTO feedback
                    (question_id, user_id, rating, comment, helpful)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                """
                result = await conn.fetchval(
                    query,
                    question_id,
                    user_id,
                    rating,
                    comment,
                    helpful
                )

                logger.info(f"Saved feedback with ID: {result}")
                return result
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            raise

    async def get_feedback_stats(self, question_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get feedback statistics.

        Args:
            question_id: Optional question ID to get stats for specific question

        Returns:
            Dictionary with feedback statistics
        """
        try:
            async with self.pool.acquire() as conn:
                if question_id:
                    query = """
                        SELECT
                            AVG(rating) as avg_rating,
                            COUNT(*) as total_feedback,
                            COUNT(CASE WHEN helpful = true THEN 1 END) as helpful_count,
                            COUNT(CASE WHEN helpful = false THEN 1 END) as not_helpful_count
                        FROM feedback
                        WHERE question_id = $1
                    """
                    record = await conn.fetchrow(query, question_id)
                else:
                    query = """
                        SELECT
                            AVG(rating) as avg_rating,
                            COUNT(*) as total_feedback,
                            COUNT(CASE WHEN helpful = true THEN 1 END) as helpful_count,
                            COUNT(CASE WHEN helpful = false THEN 1 END) as not_helpful_count
                        FROM feedback
                    """
                    record = await conn.fetchrow(query)

                return {
                    'average_rating': float(record['avg_rating']) if record['avg_rating'] else 0,
                    'total_feedback': record['total_feedback'],
                    'helpful_count': record['helpful_count'],
                    'not_helpful_count': record['not_helpful_count']
                }
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            raise

    async def save_analytics_metric(
        self,
        metric_name: str,
        metric_value: Dict[str, Any]
    ) -> int:
        """
        Save an analytics metric to the database.

        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric (as JSON)

        Returns:
            ID of the inserted record
        """
        try:
            async with self.pool.acquire() as conn:
                query = """
                    INSERT INTO analytics
                    (metric_name, metric_value)
                    VALUES ($1, $2)
                    RETURNING id
                """
                result = await conn.fetchval(
                    query,
                    metric_name,
                    json.dumps(metric_value)
                )

                logger.info(f"Saved analytics metric: {metric_name}")
                return result
        except Exception as e:
            logger.error(f"Error saving analytics metric: {e}")
            raise

    async def get_analytics_metrics(
        self,
        metric_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve analytics metrics.

        Args:
            metric_name: Optional specific metric name to retrieve
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            limit: Maximum number of records to return

        Returns:
            List of analytics metric records
        """
        try:
            async with self.pool.acquire() as conn:
                query = "SELECT id, metric_name, metric_value, created_at FROM analytics"
                params = []
                param_index = 1

                conditions = []
                if metric_name:
                    conditions.append(f"metric_name = ${param_index}")
                    params.append(metric_name)
                    param_index += 1

                if start_date:
                    conditions.append(f"created_at >= ${param_index}")
                    params.append(start_date)
                    param_index += 1

                if end_date:
                    conditions.append(f"created_at <= ${param_index}")
                    params.append(end_date)
                    param_index += 1

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += f" ORDER BY created_at DESC LIMIT ${param_index}"
                params.append(limit)

                records = await conn.fetch(query, *params)

                return [
                    {
                        'id': record['id'],
                        'metric_name': record['metric_name'],
                        'metric_value': record['metric_value'],
                        'created_at': record['created_at']
                    }
                    for record in records
                ]
        except Exception as e:
            logger.error(f"Error retrieving analytics metrics: {e}")
            raise

    async def get_question_answer_count(self) -> int:
        """
        Get the total count of question-answer pairs.

        Returns:
            Total number of question-answer pairs
        """
        try:
            async with self.pool.acquire() as conn:
                query = "SELECT COUNT(*) FROM questions_answers"
                count = await conn.fetchval(query)
                return count
        except Exception as e:
            logger.error(f"Error getting question-answer count: {e}")
            raise

    async def search_questions(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for questions containing the search term.

        Args:
            search_term: Term to search for in questions
            limit: Maximum number of results to return

        Returns:
            List of matching question-answer pairs
        """
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT id, question, answer, document_ids, selected_text_used, created_at
                    FROM questions_answers
                    WHERE LOWER(question) LIKE LOWER($1)
                    ORDER BY created_at DESC
                    LIMIT $2
                """
                records = await conn.fetch(query, f"%{search_term}%", limit)

                return [
                    {
                        'id': record['id'],
                        'question': record['question'],
                        'answer': record['answer'],
                        'document_ids': record['document_ids'],
                        'selected_text_used': record['selected_text_used'],
                        'created_at': record['created_at']
                    }
                    for record in records
                ]
        except Exception as e:
            logger.error(f"Error searching questions: {e}")
            raise


# Global database manager instance
db_manager = DatabaseManager()


# Additional utility functions
async def init_db():
    """Initialize the global database manager."""
    await db_manager.connect()


async def close_db():
    """Close the global database manager."""
    await db_manager.close()


async def save_qa_pair(
    question: str,
    answer: str,
    document_ids: List[str],
    selected_text_used: bool = False
) -> int:
    """Save a question-answer pair using the global manager."""
    return await db_manager.save_question_answer(
        question,
        answer,
        document_ids,
        selected_text_used
    )


async def get_qa_pair(qa_id: int) -> Optional[Dict[str, Any]]:
    """Get a question-answer pair using the global manager."""
    return await db_manager.get_question_answer(qa_id)


if __name__ == "__main__":
    # Example usage (this would typically be called from the main API)
    print("Database manager module loaded. Use db_manager instance for database operations.")