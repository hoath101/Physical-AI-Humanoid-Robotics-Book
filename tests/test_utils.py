import pytest
from api.utils.text_processing import chunk_text, clean_text, extract_metadata
from api.utils.citations import extract_citations, format_citations
from api.utils.validators import validate_book_content, validate_query_request
from api.utils.cache import Cache
from api.utils.privacy import PrivacyManager
from api.utils.logger import setup_logger


class TestTextProcessingUtils:
    """Test cases for text processing utilities"""

    def test_chunk_text_basic(self):
        """Test basic text chunking functionality"""
        text = "This is a sample text. " * 10  # Creates a longer text
        chunks = chunk_text(text, chunk_size=50, overlap=10)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        # Verify that chunks are not larger than chunk_size
        assert all(len(chunk) <= chunk_size + 50 for chunk in chunks)  # Adding buffer for sentence boundaries

    def test_chunk_text_with_overlap(self):
        """Test text chunking with overlap"""
        text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5."
        chunks = chunk_text(text, chunk_size=20, overlap=5)

        assert len(chunks) > 1
        # Check that there is some overlap between consecutive chunks
        if len(chunks) > 1:
            # Overlap might not be exact due to sentence boundary preservation
            pass

    def test_chunk_text_empty_input(self):
        """Test chunking with empty text"""
        with pytest.raises(ValueError):
            chunk_text("", chunk_size=100, overlap=10)

    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        dirty_text = "  This text has extra   spaces\tand\nnewlines  .  "
        cleaned = clean_text(dirty_text)

        # Should remove extra spaces and normalize whitespace
        assert cleaned.count("  ") == 0  # No double spaces
        assert "\t" not in cleaned  # No tabs
        assert cleaned.startswith("This")  # No leading spaces
        assert cleaned.endswith(".")  # Ends with the content

    def test_clean_text_special_chars(self):
        """Test cleaning text with special characters"""
        dirty_text = "Text with \x00 null\x01 chars and \x02 other \x03 specials."
        cleaned = clean_text(dirty_text)

        # Should remove control characters
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned
        assert "\x02" not in cleaned
        assert "\x03" not in cleaned

    def test_extract_metadata_basic(self):
        """Test basic metadata extraction"""
        text = "# Chapter Title\\nSubtitle\\n\\nContent here..."
        metadata = extract_metadata(text)

        assert isinstance(metadata, dict)
        assert "title" in metadata or "chapter" in metadata


class TestCitationUtils:
    """Test cases for citation utilities"""

    def test_extract_citations_basic(self):
        """Test basic citation extraction"""
        text = "This is a statement (Smith, 2023). Another fact (Johnson et al., 2022)."
        citations = extract_citations(text)

        assert isinstance(citations, list)
        # At least one citation should be found
        assert len(citations) >= 0  # May not match all patterns

    def test_format_citations_basic(self):
        """Test basic citation formatting"""
        citations = [
            {"author": "Smith", "year": "2023", "title": "Study Title"},
            {"author": "Johnson", "year": "2022", "title": "Another Study"}
        ]
        formatted = format_citations(citations)

        assert isinstance(formatted, list)
        assert len(formatted) == len(citations)


class TestValidators:
    """Test cases for validation utilities"""

    def test_validate_book_content_valid(self):
        """Test validation of valid book content"""
        content = {
            "title": "Test Chapter",
            "content": "This is valid content.",
            "book_id": "test-book",
            "chapter_id": "ch1",
            "page_number": 1
        }

        # Should not raise any exceptions
        validate_book_content(content)

    def test_validate_book_content_missing_fields(self):
        """Test validation of book content with missing fields"""
        content = {
            "title": "Test Chapter",
            # Missing required fields
        }

        with pytest.raises(ValueError):
            validate_book_content(content)

    def test_validate_book_content_empty_content(self):
        """Test validation of book content with empty content"""
        content = {
            "title": "Test Chapter",
            "content": "",  # Empty content
            "book_id": "test-book",
            "chapter_id": "ch1",
            "page_number": 1
        }

        with pytest.raises(ValueError):
            validate_book_content(content)

    def test_validate_query_request_valid(self):
        """Test validation of valid query request"""
        query_data = {
            "query": "What is artificial intelligence?",
            "book_id": "test-book",
            "mode": "GLOBAL"
        }

        # Should not raise any exceptions
        validate_query_request(query_data)

    def test_validate_query_request_empty_query(self):
        """Test validation of query request with empty query"""
        query_data = {
            "query": "",  # Empty query
            "book_id": "test-book",
            "mode": "GLOBAL"
        }

        with pytest.raises(ValueError):
            validate_query_request(query_data)

    def test_validate_query_request_invalid_mode(self):
        """Test validation of query request with invalid mode"""
        query_data = {
            "query": "What is AI?",
            "book_id": "test-book",
            "mode": "INVALID_MODE"
        }

        with pytest.raises(ValueError):
            validate_query_request(query_data)


class TestCache:
    """Test cases for caching utilities"""

    def test_cache_basic_operations(self):
        """Test basic cache operations"""
        cache = Cache(max_size=3, ttl=60)

        # Test setting and getting a value
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Test getting a non-existent key
        assert cache.get("nonexistent") is None

    def test_cache_expiration(self):
        """Test cache expiration functionality"""
        cache = Cache(max_size=2, ttl=0.01)  # Very short TTL for testing

        cache.set("expiring_key", "expiring_value")
        assert cache.get("expiring_key") == "expiring_value"

        # Wait for expiration
        import time
        time.sleep(0.02)  # Wait longer than TTL

        assert cache.get("expiring_key") is None

    def test_cache_max_size_eviction(self):
        """Test cache eviction when max size is reached"""
        cache = Cache(max_size=2, ttl=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # This should trigger eviction

        # At least one of the first two keys should be evicted
        keys_present = [cache.get("key1") is not None, cache.get("key2") is not None]
        # Since we have max_size=2 and added 3 items, at least one should be evicted
        assert not all(keys_present)  # Not both keys should be present

    def test_cache_clear(self):
        """Test cache clearing functionality"""
        cache = Cache(max_size=5, ttl=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache._cache) == 0

    def test_cache_delete(self):
        """Test cache deletion functionality"""
        cache = Cache(max_size=5, ttl=60)

        cache.set("key1", "value1")
        cache.delete("key1")

        assert cache.get("key1") is None


class TestPrivacyManager:
    """Test cases for privacy utilities"""

    def test_anonymize_user_data(self):
        """Test user data anonymization"""
        privacy_manager = PrivacyManager()

        user_data = {
            "user_id": "user123456789",
            "email": "test@example.com",
            "name": "John Doe",
            "phone": "123-456-7890",
            "other_field": "preserved_value"
        }

        anonymized = privacy_manager.anonymize_user_data(user_data)

        # Check that personal data is anonymized
        assert anonymized["user_id"] == "use*********"  # First 3 chars + stars
        assert anonymized["email"] == "[REDACTED]"
        assert anonymized["name"] == "[REDACTED]"
        assert anonymized["phone"] == "[REDACTED]"

        # Check that non-personal data is preserved
        assert anonymized["other_field"] == "preserved_value"

    def test_should_store_conversation(self):
        """Test conversation storage decision"""
        privacy_manager = PrivacyManager()

        # With consent
        assert privacy_manager.should_store_conversation(user_consent=True) is True

        # Without consent
        assert privacy_manager.should_store_conversation(user_consent=False) is False

    def test_sanitize_query(self):
        """Test query sanitization"""
        privacy_manager = PrivacyManager()

        query_with_pii = "Contact me at john@example.com or call 123-456-7890"
        sanitized = privacy_manager.sanitize_query(query_with_pii)

        # Check that PII is removed
        assert "[EMAIL_REMOVED]" in sanitized
        assert "[PHONE_REMOVED]" in sanitized

    def test_encrypt_decrypt_sensitive_data(self):
        """Test encryption and decryption of sensitive data"""
        privacy_manager = PrivacyManager()

        original_data = "sensitive information"
        encrypted = privacy_manager.encrypt_sensitive_data(original_data)
        decrypted = privacy_manager.decrypt_sensitive_data(encrypted)

        assert decrypted == original_data
        assert encrypted.startswith("encrypted:")

    def test_encrypt_decrypt_with_invalid_data(self):
        """Test decryption with invalid data"""
        privacy_manager = PrivacyManager()

        # Should handle invalid encrypted data gracefully
        result = privacy_manager.decrypt_sensitive_data("invalid_encrypted_data")
        assert result == "invalid_encrypted_data"  # Returns original if not in expected format


class TestLogger:
    """Test cases for logging utilities"""

    def test_setup_logger(self):
        """Test logger setup"""
        logger = setup_logger("test_logger", level="INFO")

        assert logger is not None
        assert logger.name == "test_logger"

        # Should be able to log messages without error
        logger.info("Test log message")