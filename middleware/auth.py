"""
Authentication middleware for the RAG Chatbot API.
This module provides API key validation and authentication functionality.
"""

import os
import time
import hashlib
from typing import Optional, List
from fastapi import HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize security scheme
security = HTTPBearer(auto_error=False)  # Set auto_error to False to make it optional

class APIKeyManager:
    """
    Manages API keys and authentication for the RAG Chatbot API.
    """

    def __init__(self):
        # Load API keys from environment variables
        self.api_keys = self._load_api_keys()
        logger.info(f"Loaded {len(self.api_keys)} API keys")

    def _load_api_keys(self) -> set:
        """
        Load API keys from environment variables.
        Supports multiple keys separated by commas.
        """
        api_keys_str = os.getenv("API_KEYS", "")
        if api_keys_str:
            # Split by comma and strip whitespace
            keys = {key.strip() for key in api_keys_str.split(",") if key.strip()}
            return keys
        else:
            # Fallback to single API_KEY if API_KEYS is not set
            single_key = os.getenv("API_KEY")
            if single_key:
                return {single_key}
            else:
                logger.warning("No API keys configured - authentication disabled in development mode")
                return set()

    def is_valid_api_key(self, api_key: str) -> bool:
        """
        Check if the provided API key is valid.

        Args:
            api_key: The API key to validate

        Returns:
            True if the API key is valid, False otherwise
        """
        # Check if in development mode (no keys configured OR development environment)
        if not self.api_keys or os.getenv("ENVIRONMENT", "development") == "development":
            # In development mode, allow valid API keys or allow all requests
            if not self.api_keys:
                return True
            else:
                # If keys are configured but in development, still check them
                return api_key in self.api_keys
        return api_key in self.api_keys

    def add_api_key(self, api_key: str):
        """
        Add a new API key to the manager.

        Args:
            api_key: The API key to add
        """
        self.api_keys.add(api_key)
        logger.info(f"Added new API key")

    def remove_api_key(self, api_key: str):
        """
        Remove an API key from the manager.

        Args:
            api_key: The API key to remove
        """
        if api_key in self.api_keys:
            self.api_keys.remove(api_key)
            logger.info(f"Removed API key")


# Global API key manager instance
api_key_manager = APIKeyManager()


async def authenticate_request(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Authenticate the incoming request using API key.
    In development mode, allows requests without API key from localhost.

    Args:
        request: The incoming request (for development mode check)
        credentials: HTTP authorization credentials (may be None)

    Returns:
        The validated API key or a default key in development

    Raises:
        HTTPException: If authentication fails in production
    """
    # Check if in development mode
    if os.getenv("ENVIRONMENT", "development") == "development":
        # In development, allow requests from localhost without API key
        client_host = request.client.host if request.client else ""
        if client_host in ["127.0.0.1", "localhost", "::1"]:
            logger.info("Development mode: Allowing localhost request without API key")
            # Return the configured API key if available, otherwise a development key
            if api_key_manager.api_keys:
                return next(iter(api_key_manager.api_keys))
            else:
                return "development-key"

    # Production mode or non-localhost request: require API key
    if credentials is None or credentials.credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = credentials.credentials

    if not api_key_manager.is_valid_api_key(api_key):
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info(f"Successfully authenticated request with API key: {api_key[:8]}...")
    return api_key


def require_api_key():
    """
    Dependency to require API key authentication for specific endpoints.
    """
    def api_key_dependency(credentials: HTTPAuthorizationCredentials = security):
        return authenticate_request(credentials)
    return api_key_dependency


class RateLimitManager:
    """
    Simple in-memory rate limiter to prevent API abuse.
    """

    def __init__(self):
        self.requests = {}  # key: api_key, value: list of request timestamps
        self.rate_limit = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))  # requests per window
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # window in seconds (1 hour)

    def is_allowed(self, api_key: str) -> bool:
        """
        Check if the API key is allowed to make a request based on rate limits.

        Args:
            api_key: The API key making the request

        Returns:
            True if allowed, False if rate limited
        """
        current_time = time.time()

        # Clean old requests outside the window
        if api_key in self.requests:
            self.requests[api_key] = [
                req_time for req_time in self.requests[api_key]
                if current_time - req_time < self.rate_limit_window
            ]
        else:
            self.requests[api_key] = []

        # Check if under rate limit
        if len(self.requests[api_key]) < self.rate_limit:
            self.requests[api_key].append(current_time)
            return True

        return False

    def get_reset_time(self, api_key: str) -> int:
        """
        Get the time when the rate limit will reset for this API key.

        Args:
            api_key: The API key to check

        Returns:
            Unix timestamp when the rate limit will reset
        """
        if api_key not in self.requests:
            return int(time.time())

        # Find the earliest request that's still within the window
        current_time = time.time()
        valid_requests = [
            req_time for req_time in self.requests[api_key]
            if current_time - req_time < self.rate_limit_window
        ]

        if not valid_requests:
            return int(time.time())

        # Reset time is when the earliest request expires
        earliest_request = min(valid_requests)
        return int(earliest_request + self.rate_limit_window)


# Global rate limiter instance
rate_limiter = RateLimitManager()


async def check_rate_limit(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Check if the request is within rate limits.
    In development mode, allows requests without rate limiting from localhost.

    Args:
        request: The incoming request
        credentials: HTTP authorization credentials

    Raises:
        HTTPException: If rate limit is exceeded
    """
    # Check if in development mode
    if os.getenv("ENVIRONMENT", "development") == "development":
        # In development, allow requests from localhost without rate limiting
        client_host = request.client.host if request.client else ""
        if client_host in ["127.0.0.1", "localhost", "::1"]:
            logger.info("Development mode: Skipping rate limit for localhost request")
            # Return the configured API key if available, otherwise a development key
            if api_key_manager.api_keys:
                return next(iter(api_key_manager.api_keys))
            else:
                return "development-key"

    # Production mode: require credentials and check rate limit
    if credentials is None or credentials.credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = credentials.credentials

    if not rate_limiter.is_allowed(api_key):
        reset_time = rate_limiter.get_reset_time(api_key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "reset_time": reset_time,
                "message": f"Rate limit exceeded. Try again after {reset_time}"
            }
        )

    return api_key