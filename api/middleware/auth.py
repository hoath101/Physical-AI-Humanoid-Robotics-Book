from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

class APIKeyAuth:
    """
    API Key authentication middleware.
    """
    def __init__(self):
        self.api_key = settings.api_key
        self.security = HTTPBearer(auto_error=False)

    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        # Get the authorization header
        authorization: str = request.headers.get("Authorization")

        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header"
            )

        # Check if it's a Bearer token
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format. Use 'Bearer {api_key}'"
            )

        # Extract the token
        token = authorization[7:]  # Remove "Bearer " prefix

        # Validate the token
        if token != self.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

        # Log successful authentication
        logger.info(f"Successful API authentication for request: {request.method} {request.url}")

        return HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token
        )

# Create an instance of the auth middleware
api_key_auth = APIKeyAuth()