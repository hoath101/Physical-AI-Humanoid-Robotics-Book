from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio
from openai import AsyncOpenAI

from api.models.document import Query, Response, Session, SessionDB
from api.services.session import session_service
from api.models.request import QueryMode
from api.services.retrieval import retrieval_service
from api.services.embedding import embedding_service
from api.config.settings import settings
from api.utils.citations import format_citations, extract_citation_info
from api.utils.validators import validate_query_request

# Set up logging
logger = logging.getLogger(__name__)

class ChatService:
    """
    Service for handling chat interactions and response generation.
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.retrieval_service = retrieval_service
        self.embedding_service = embedding_service
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens

    async def generate_response(
        self,
        query: str,
        book_id: str,
        query_mode: QueryMode,
        selected_text: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response to the user's query based on retrieved context.
        """
        # Validate the query request
        validation_errors = validate_query_request(query, query_mode.value, selected_text)
        if validation_errors:
            raise ValueError(f"Validation errors: {', '.join(validation_errors)}")

        # Get or create session
        session = await self.get_or_create_session(session_id, user_id, book_id)

        # Update session activity
        await self.update_session_activity(session.id)

        # Retrieve relevant context
        if query_mode == QueryMode.SELECTION_ONLY and selected_text:
            # Validate that the selected text exists in the book
            is_valid_selection = await self.retrieval_service.validate_selection_context(
                selected_text, book_id
            )

            if not is_valid_selection:
                logger.warning(f"Selection context validation failed for book {book_id}")
                # Even if validation fails, we'll proceed but with a warning

            # Use the enhanced retrieval with strict filtering for selection-only mode
            retrieved_context = await self.retrieval_service.retrieve_content_with_strict_selection_filtering(
                query, selected_text, book_id
            )
        else:
            retrieved_context = await self.retrieval_service.retrieve_content(
                query, book_id, query_mode
            )

        # Check if we found relevant context
        if not retrieved_context.content_segments:
            return {
                "answer": "Not found in source text",
                "citations": [],
                "query_mode": query_mode,
                "retrieved_context_count": 0,
                "status": "no_content_found"
            }

        # Prepare context for the LLM
        context_texts = [segment.text for segment in retrieved_context.content_segments]
        context_str = "\n\n".join(context_texts)

        # Prepare the prompt for the LLM
        if query_mode == QueryMode.SELECTION_ONLY and selected_text:
            prompt = f"""
            You are an AI assistant that answers questions based only on the provided context from a specific book.
            The user has selected specific text and asked a question about it.
            You must answer the question based ONLY on the following context, which is related to the selected text.
            Do not use any external knowledge or make assumptions beyond what is provided in the context.
            If the answer cannot be found in the provided context, respond with "Not found in source text".

            Selected text: {selected_text}

            Context for answering the question:
            {context_str}

            Question: {query}

            Answer:
            """
        else:
            prompt = f"""
            You are an AI assistant that answers questions based only on the provided context from a specific book.
            You must answer the question based ONLY on the following context.
            Do not use any external knowledge or make assumptions beyond what is provided in the context.
            If the answer cannot be found in the provided context, respond with "Not found in source text".

            Context:
            {context_str}

            Question: {query}

            Answer:
            """

        try:
            # Generate response using OpenAI
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # You can change this to gpt-4 if preferred
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based only on provided context. If the answer is not in the context, respond with 'Not found in source text'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = response.choices[0].message.content.strip()

            # Format citations from the retrieved context
            citations = []
            for segment in retrieved_context.content_segments:
                citation_data = {
                    "text": segment.text[:200] + "..." if len(segment.text) > 200 else segment.text,
                    "chapter": segment.chapter or "Unknown",
                    "page": segment.page,
                    "section": segment.section,
                    "paragraph_id": segment.paragraph_id
                }
                citation = extract_citation_info(citation_data)
                citations.append(citation)

            # Validate that the response is grounded in the context
            context_text = "\n".join([seg.text for seg in retrieved_context.content_segments])
            is_answer_grounded = await self.validate_grounding(answer, context_text)

            if not is_answer_grounded and answer != "Not found in source text":
                logger.warning(f"Response may not be fully grounded for query in book {book_id}")
                # In a production system, you might want to handle this differently
                # For now, we'll proceed but log the issue

            result = {
                "answer": answer,
                "citations": citations,
                "query_mode": query_mode,
                "retrieved_context_count": len(retrieved_context.content_segments),
                "status": "success" if answer != "Not found in source text" else "no_content_found"
            }

            logger.info(f"Generated response for query in book {book_id}")
            return result

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    async def create_session(self, user_id: Optional[str] = None, book_id: Optional[str] = None) -> Session:
        """
        Create a new chat session.
        """
        # Use the session service to create a session in the database
        session = await session_service.create_session(user_id, book_id)

        # Convert SQLAlchemy model to Pydantic model
        session_model = Session(
            id=session.id,
            user_id=session.user_id,
            book_id=session.book_id,
            start_time=session.start_time,
            last_activity=session.last_activity,
            active=session.active
        )

        logger.info(f"Created new session: {session_model.id}")
        return session_model

    async def get_or_create_session(self, session_id: Optional[str], user_id: Optional[str] = None, book_id: Optional[str] = None) -> Session:
        """
        Get an existing session or create a new one if it doesn't exist.
        """
        if session_id:
            # Try to get the session from the database
            session = await session_service.get_session(session_id)
            if session:
                logger.info(f"Using existing session: {session_id}")
                # Convert SQLAlchemy model to Pydantic model
                return Session(
                    id=session.id,
                    user_id=session.user_id,
                    book_id=session.book_id,
                    start_time=session.start_time,
                    last_activity=session.last_activity,
                    active=session.active
                )
            else:
                logger.warning(f"Session {session_id} not found in database, creating new one")

        logger.info("Creating new session")
        return await self.create_session(user_id, book_id)

    async def update_session_activity(self, session_id: str):
        """
        Update the last activity timestamp for a session.
        """
        await session_service.update_session_activity(session_id)

    async def validate_grounding(self, response: str, context: str) -> bool:
        """
        Validate that the response is grounded in the provided context.
        This is a simplified implementation - a more sophisticated approach
        would use semantic similarity or other NLP techniques.
        """
        # Convert both to lowercase for comparison
        response_lower = response.lower()
        context_lower = context.lower()

        # Simple check: if the response contains key phrases from the context,
        # it's likely grounded
        context_words = set(context_lower.split()[:50])  # Use first 50 words as representative
        response_words = set(response_lower.split())

        # Calculate overlap
        if not context_words:
            return False

        overlap = len(context_words.intersection(response_words))
        overlap_ratio = overlap / len(context_words)

        # Consider it grounded if there's at least 10% overlap
        return overlap_ratio >= 0.1

    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a session.
        """
        # In a real implementation, you would fetch this from the database
        # For now, return an empty list
        logger.info(f"Retrieving history for session: {session_id}")
        return []

# Create a singleton instance
chat_service = ChatService()