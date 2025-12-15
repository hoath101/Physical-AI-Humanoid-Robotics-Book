from typing import Any, Dict, List
from pydantic import BaseModel, ValidationError
import re

def validate_book_content(content: str) -> bool:
    """
    Validate that the book content is not empty and meets basic requirements.
    """
    if not content or not content.strip():
        return False

    # Check if content has a reasonable length (at least 100 characters)
    if len(content.strip()) < 100:
        return False

    return True

def validate_book_metadata(metadata: Dict[str, Any]) -> List[str]:
    """
    Validate book metadata and return a list of validation errors.
    """
    errors = []

    if not metadata.get('title'):
        errors.append("Book title is required")

    if not metadata.get('author'):
        errors.append("Book author is required")

    # Validate optional fields if provided
    isbn = metadata.get('isbn')
    if isbn:
        # Basic ISBN validation (ISBN-10 or ISBN-13 format)
        isbn_clean = re.sub(r'[^0-9X]', '', str(isbn))
        if len(isbn_clean) not in [10, 13]:
            errors.append("ISBN must be either 10 or 13 digits (with optional X for ISBN-10)")

    return errors

def validate_query_request(question: str, query_mode: str, selected_text: str = None) -> List[str]:
    """
    Validate query request parameters and return a list of validation errors.
    """
    errors = []

    if not question or not question.strip():
        errors.append("Question cannot be empty")

    if len(question.strip()) < 3:
        errors.append("Question must be at least 3 characters long")

    if query_mode not in ['global', 'selection_only']:
        errors.append("Query mode must be either 'global' or 'selection_only'")

    if query_mode == 'selection_only':
        if not selected_text or not selected_text.strip():
            errors.append("Selected text is required for selection_only mode")

    return errors

def validate_ingestion_request(book_content: str, book_metadata: Dict[str, Any],
                              chunk_size: int, overlap_size: int, book_id: str) -> List[str]:
    """
    Validate ingestion request parameters and return a list of validation errors.
    """
    errors = []

    # Validate book content
    if not validate_book_content(book_content):
        errors.append("Book content is invalid - must not be empty and should have at least 100 characters")

    # Validate book metadata
    metadata_errors = validate_book_metadata(book_metadata)
    errors.extend(metadata_errors)

    # Validate chunk size
    if not isinstance(chunk_size, int) or chunk_size < 100 or chunk_size > 5000:
        errors.append("Chunk size must be an integer between 100 and 5000")

    # Validate overlap size
    if not isinstance(overlap_size, int) or overlap_size < 0 or overlap_size > 1000:
        errors.append("Overlap size must be an integer between 0 and 1000")

    # Validate that overlap is not larger than chunk size
    if overlap_size >= chunk_size:
        errors.append("Overlap size must be smaller than chunk size")

    # Validate book ID
    if not book_id or not book_id.strip():
        errors.append("Book ID is required")

    if len(book_id.strip()) < 3:
        errors.append("Book ID must be at least 3 characters long")

    return errors

def validate_book_id(book_id: str) -> List[str]:
    """
    Validate book ID format and return a list of validation errors.
    """
    errors = []

    if not book_id or not book_id.strip():
        errors.append("Book ID cannot be empty")

    # Basic validation for a reasonable book ID format
    if len(book_id.strip()) < 3:
        errors.append("Book ID must be at least 3 characters long")

    if len(book_id.strip()) > 100:
        errors.append("Book ID must not exceed 100 characters")

    # Check for valid characters (alphanumeric, hyphens, underscores)
    if not re.match(r'^[a-zA-Z0-9_-]+$', book_id.strip()):
        errors.append("Book ID can only contain alphanumeric characters, hyphens, and underscores")

    return errors

def is_valid_url(url: str) -> bool:
    """
    Basic URL validation.
    """
    if not url:
        return False

    # Basic URL pattern
    pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
    return re.match(pattern, url) is not None

def is_valid_api_key(api_key: str) -> bool:
    """
    Basic API key validation.
    """
    if not api_key or len(api_key) < 10:
        return False

    # Check if it looks like a typical API key (has common prefixes)
    return api_key.startswith(('sk-', 'pk-', 'api_', 'key_')) or len(api_key) > 20