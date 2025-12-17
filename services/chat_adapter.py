"""
Custom Chat Adapter for integrating ChatKit UI with RAG backend.
This service bridges the ChatKit frontend with the RAG backend system.
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from uuid import uuid4

from openai import OpenAI
from pydantic import BaseModel

from rag.retriever import RAGRetriever
from db.database import db_manager


class Message(BaseModel):
    id: str
    text: str
    sender_id: str
    created_at: str
    updated_at: str


class Thread(BaseModel):
    id: str
    name: str
    messages: List[Message]
    created_at: str
    updated_at: str


class User(BaseModel):
    id: str
    name: str
    created_at: str


class ChatAdapter:
    """
    Adapter class that provides ChatKit-like interface but uses RAG backend
    """

    def __init__(self, openai_client: OpenAI, rag_retriever: RAGRetriever):
        self.openai_client = openai_client
        self.rag_retriever = rag_retriever
        self.logger = logging.getLogger(__name__)

        # In-memory storage for threads and messages (in production, use database)
        self.threads: Dict[str, Thread] = {}
        self.users: Dict[str, User] = {}

    async def create_user(self, user_id: str, name: str) -> User:
        """Create a new user"""
        user = User(
            id=user_id,
            name=name,
            created_at=datetime.now().isoformat()
        )
        self.users[user_id] = user
        return user

    async def get_or_create_user(self, user_id: str, name: str = "Anonymous User") -> User:
        """Get existing user or create new one"""
        if user_id in self.users:
            return self.users[user_id]
        return await self.create_user(user_id, name)

    async def create_thread(self, name: str = "Book Discussion") -> Thread:
        """Create a new thread for conversation"""
        thread_id = str(uuid4())
        thread = Thread(
            id=thread_id,
            name=name,
            messages=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        self.threads[thread_id] = thread
        return thread

    async def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get thread by ID"""
        return self.threads.get(thread_id)

    async def get_threads(self) -> List[Thread]:
        """Get all threads for a user (simplified for this implementation)"""
        return list(self.threads.values())

    async def send_message(self, thread_id: str, user_id: str, text: str) -> Message:
        """Send a message and get RAG-enhanced response"""
        # Get or create user
        user = await self.get_or_create_user(user_id)

        # Create user message
        user_message = Message(
            id=str(uuid4()),
            text=text,
            sender_id=user_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

        # Add user message to thread
        if thread_id not in self.threads:
            thread = await self.create_thread(f"Discussion about: {text[:50]}...")
        else:
            thread = self.threads[thread_id]

        thread.messages.append(user_message)

        # Get RAG response
        try:
            # Retrieve relevant chunks from the book
            retrieved_chunks = await self.rag_retriever.retrieve_relevant_chunks(
                text,
                book_section=None  # We'll search across all sections
            )

            if not retrieved_chunks:
                # If no relevant chunks found, return appropriate response
                response_text = (
                    "I couldn't find specific information in the book related to your question: "
                    f"{text}. Please check if your question is covered in the book content."
                )
            else:
                # Prepare context for the LLM from retrieved chunks
                context_str = "\n\n".join([chunk['content'] for chunk in retrieved_chunks])

                # Check if the context is too long for the model
                max_context_length = 10000  # Approximate limit for model context
                if len(context_str) > max_context_length:
                    # Truncate context to fit in model limits
                    context_str = context_str[:max_context_length]
                    self.logger.info(f"Context was too long and was truncated to {max_context_length} characters")

                # Format prompt for OpenAI
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert assistant for the 'Physical AI & Humanoid Robotics' book. "
                            "Answer the user's question using ONLY the provided context from the book. "
                            "Reference specific chapters or sections mentioned in the context when possible. "
                            "If the answer cannot be found in the provided context, clearly state that the information is not available in the provided book excerpts."
                            "Be concise and accurate, and always cite information from the provided context."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Question: {text}\n\nContext from the book:\n{context_str}"
                    }
                ]

                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000,
                    timeout=30  # Add timeout to prevent hanging requests
                )

                response_text = response.choices[0].message.content

            # Create AI response message
            ai_message = Message(
                id=str(uuid4()),
                text=response_text,
                sender_id="ai_assistant",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )

            # Add AI response to thread
            thread.messages.append(ai_message)
            thread.updated_at = datetime.now().isoformat()

            # Save the interaction to database (if available)
            try:
                await db_manager.save_question_answer(
                    question=text,
                    answer=response_text,
                    document_ids=[chunk['id'] for chunk in retrieved_chunks] if retrieved_chunks else [],
                    selected_text_used=False
                )
            except Exception as e:
                self.logger.warning(f"Could not save question-answer to database: {e}")
                # Continue without database logging if it fails

            return ai_message

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            # Create error response message
            error_message = Message(
                id=str(uuid4()),
                text=f"I encountered an error processing your request: {str(e)}",
                sender_id="ai_assistant",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )

            thread.messages.append(error_message)
            thread.updated_at = datetime.now().isoformat()

            # Try to save error interaction to database (if available)
            try:
                await db_manager.save_question_answer(
                    question=text,
                    answer=f"Error processing request: {str(e)}",
                    document_ids=[],
                    selected_text_used=False
                )
            except Exception as db_e:
                self.logger.warning(f"Could not save error to database: {db_e}")
                # Continue without database logging if it fails

            return error_message

    async def get_or_create_thread(self, thread_id: str, name: str = "Book Discussion") -> Thread:
        """Get existing thread or create new one"""
        if thread_id in self.threads:
            return self.threads[thread_id]
        return await self.create_thread(name)

    async def get_messages(self, thread_id: str, limit: int = 100) -> List[Message]:
        """Get messages from a thread"""
        thread = self.threads.get(thread_id)
        if not thread:
            return []

        # Return last 'limit' messages
        return thread.messages[-limit:]

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread"""
        if thread_id in self.threads:
            del self.threads[thread_id]
            return True
        return False


# Global chat adapter instance (in production, this would be managed differently)
chat_adapter: Optional[ChatAdapter] = None


def get_chat_adapter() -> ChatAdapter:
    """Get the global chat adapter instance"""
    global chat_adapter
    if chat_adapter is None:
        raise RuntimeError("ChatAdapter not initialized. Call init_chat_adapter first.")
    return chat_adapter


async def init_chat_adapter(openai_client: OpenAI, rag_retriever: RAGRetriever):
    """Initialize the chat adapter with required dependencies"""
    global chat_adapter
    chat_adapter = ChatAdapter(openai_client, rag_retriever)
    return chat_adapter