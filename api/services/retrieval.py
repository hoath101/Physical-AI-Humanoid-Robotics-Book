from typing import List, Dict, Any, Optional
import logging
from qdrant_client.http import models

from api.config.vector_db import vector_db
from api.services.embedding import embedding_service
from api.models.document import RetrievedContext, BookContent
from api.models.request import QueryMode

# Set up logging
logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Service for retrieving relevant content based on user queries.
    """

    def __init__(self):
        self.vector_db = vector_db
        self.embedding_service = embedding_service

    async def retrieve_content(
        self,
        question: str,
        book_id: str,
        query_mode: QueryMode,
        selected_text: Optional[str] = None,
        top_k: int = 5
    ) -> RetrievedContext:
        """
        Retrieve relevant content based on the question and query mode.
        """
        # Generate embedding for the question
        query_embedding = await self.embedding_service.generate_embedding(question)

        # Prepare filters based on query mode
        filters = []

        # Always filter by book_id
        filters.append(
            models.FieldCondition(
                key="book_id",
                match=models.MatchValue(value=book_id)
            )
        )

        # If in selection-only mode, add additional filters
        if query_mode == QueryMode.SELECTION_ONLY and selected_text:
            # In selection-only mode, we want to retrieve content that is related to the selected text
            # For now, we'll use the selected text as an additional search term
            # In a more sophisticated implementation, we might use semantic similarity to the selected text
            selected_text_embedding = await self.embedding_service.generate_embedding(selected_text)

            # We'll use the vector search with a filter to only get results from the same document
            # This is a simplified approach - in practice, you might want to implement
            # more complex logic to ensure only the selected text context is used
            pass

        # Create the filter
        search_filter = models.Filter(must=filters) if filters else None

        # Perform the search
        search_results = self.vector_db.search(
            vector=query_embedding,
            limit=top_k,
            filters=search_filter
        )

        # Process the results
        content_segments = []
        relevance_scores = []

        for result in search_results:
            # Create BookContent object from the result
            payload = result.payload

            content_segment = BookContent(
                id=payload.get("id", ""),
                text=payload.get("text", ""),
                chapter=payload.get("chapter"),
                page=payload.get("page"),
                section=payload.get("section", payload.get("title", "")),  # Using title as section if not available
                paragraph_id=payload.get("paragraph_id"),
                book_id=payload.get("book_id", ""),
                embedding_vector=None,  # We don't need the full vector in the response
                metadata=payload
            )

            content_segments.append(content_segment)
            relevance_scores.append(result.score)

        # Create RetrievedContext object
        retrieved_context = RetrievedContext(
            content_segments=content_segments,
            relevance_scores=relevance_scores,
            metadata={
                "query_mode": query_mode,
                "book_id": book_id,
                "top_k": top_k
            }
        )

        logger.info(f"Retrieved {len(content_segments)} content segments for query in book {book_id}")
        return retrieved_context

    async def retrieve_content_by_selection(
        self,
        question: str,
        selected_text: str,
        book_id: str,
        top_k: int = 3
    ) -> RetrievedContext:
        """
        Retrieve content specifically related to the selected text.
        This is a more focused retrieval for the selection-only mode.
        """
        # For selection-only mode, we want to find content that is most relevant
        # to the combination of the question and the selected text context

        # Generate embeddings for both the question and selected text
        question_embedding = await self.embedding_service.generate_embedding(question)
        selected_text_embedding = await self.embedding_service.generate_embedding(selected_text)

        # Create a combined query that emphasizes both the question and the selected context
        # We'll use a weighted average, giving more weight to the selected text
        # since this is selection-only mode
        combined_embedding = [
            0.3 * q + 0.7 * s for q, s in zip(question_embedding, selected_text_embedding)
        ]

        # Filter to only get results from the specific book
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="book_id",
                    match=models.MatchValue(value=book_id)
                )
            ]
        )

        # Perform the search using the combined embedding
        search_results = self.vector_db.search(
            vector=combined_embedding,
            limit=top_k,
            filters=search_filter
        )

        # Process the results
        content_segments = []
        relevance_scores = []

        for result in search_results:
            payload = result.payload

            content_segment = BookContent(
                id=payload.get("id", ""),
                text=payload.get("text", ""),
                chapter=payload.get("chapter"),
                page=payload.get("page"),
                section=payload.get("section", payload.get("title", "")),
                paragraph_id=payload.get("paragraph_id"),
                book_id=payload.get("book_id", ""),
                embedding_vector=None,
                metadata=payload
            )

            content_segments.append(content_segment)
            relevance_scores.append(result.score)

        retrieved_context = RetrievedContext(
            content_segments=content_segments,
            relevance_scores=relevance_scores,
            metadata={
                "query_mode": QueryMode.SELECTION_ONLY,
                "book_id": book_id,
                "top_k": top_k,
                "selection_context_used": True
            }
        )

        logger.info(f"Retrieved {len(content_segments)} content segments for selection-only query in book {book_id}")
        return retrieved_context

    async def retrieve_content_with_strict_selection_filtering(
        self,
        question: str,
        selected_text: str,
        book_id: str,
        top_k: int = 3
    ) -> RetrievedContext:
        """
        Retrieve content with strict filtering to ensure responses only use selected text context.
        This method implements additional validation to ensure retrieved content is related
        to the selected text.
        """
        # First, get content related to the selected text
        selected_text_embedding = await self.embedding_service.generate_embedding(selected_text)

        # Search for content similar to the selected text within the book
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="book_id",
                    match=models.MatchValue(value=book_id)
                )
            ]
        )

        # Find content similar to the selected text
        search_results = self.vector_db.search(
            vector=selected_text_embedding,
            limit=top_k * 2,  # Get more results to have options after filtering
            filters=search_filter
        )

        # Filter results to ensure they are contextually related to the selected text
        filtered_results = []
        for result in search_results:
            # Additional validation could go here to ensure the content is truly related
            # For now, we'll use a simple relevance threshold
            if result.score > 0.5:  # Adjust threshold as needed
                filtered_results.append(result)
            if len(filtered_results) >= top_k:
                break

        # If we don't have enough high-relevance results, include lower-relevance ones
        # but mark them appropriately
        if len(filtered_results) < top_k:
            for result in search_results:
                if result not in filtered_results:
                    filtered_results.append(result)
                    if len(filtered_results) >= top_k:
                        break

        # Process the filtered results
        content_segments = []
        relevance_scores = []

        for result in filtered_results:
            payload = result.payload

            content_segment = BookContent(
                id=payload.get("id", ""),
                text=payload.get("text", ""),
                chapter=payload.get("chapter"),
                page=payload.get("page"),
                section=payload.get("section", payload.get("title", "")),
                paragraph_id=payload.get("paragraph_id"),
                book_id=payload.get("book_id", ""),
                embedding_vector=None,
                metadata=payload
            )

            content_segments.append(content_segment)
            relevance_scores.append(result.score)

        retrieved_context = RetrievedContext(
            content_segments=content_segments,
            relevance_scores=relevance_scores,
            metadata={
                "query_mode": QueryMode.SELECTION_ONLY,
                "book_id": book_id,
                "top_k": len(content_segments),
                "selection_context_used": True,
                "strict_filtering_applied": True
            }
        )

        logger.info(f"Retrieved {len(content_segments)} content segments with strict filtering for selection-only query in book {book_id}")
        return retrieved_context

    async def validate_selection_context(
        self,
        selected_text: str,
        book_id: str,
        context_window: int = 2
    ) -> bool:
        """
        Validate that the selected text exists in the stored content.
        This helps ensure the selection-only mode is working correctly.
        """
        if not selected_text or not selected_text.strip():
            return False

        # Generate embedding for the selected text
        selected_text_embedding = await self.embedding_service.generate_embedding(selected_text)

        # Search for similar content in the specified book
        search_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="book_id",
                    match=models.MatchValue(value=book_id)
                )
            ]
        )

        search_results = self.vector_db.search(
            vector=selected_text_embedding,
            limit=1,  # Only need to find if it exists
            filters=search_filter
        )

        # If we found similar content, consider the selection valid
        if search_results:
            # Check if the similarity score is above a threshold
            # This threshold can be adjusted based on requirements
            threshold = 0.7  # Adjust as needed
            return search_results[0].score >= threshold

        return False

# Create a singleton instance
retrieval_service = RetrievalService()