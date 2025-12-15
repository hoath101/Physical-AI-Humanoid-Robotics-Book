from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import logging
from collections import defaultdict, deque
from typing import Dict

# Set up logging
logger = logging.getLogger(__name__)

class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Middleware to implement rate limiting for API endpoints.
    """
    def __init__(self, app, requests_limit: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        # Dictionary to store request timestamps for each client
        self.requests: Dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        # Get client IP address
        client_ip = request.client.host if request.client else "unknown"

        # Get current timestamp
        current_time = time.time()

        # Remove old requests outside the time window
        while (self.requests[client_ip] and
               current_time - self.requests[client_ip][0] > self.window_seconds):
            self.requests[client_ip].popleft()

        # Check if the client has exceeded the rate limit
        if len(self.requests[client_ip]) >= self.requests_limit:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "status": "error",
                    "message": "Rate limit exceeded",
                    "retry_after": self.window_seconds
                }
            )

        # Add current request timestamp
        self.requests[client_ip].append(current_time)

        # Process the request
        response = await call_next(request)
        return response

# Create a default rate limiter instance
rate_limiter = RateLimiterMiddleware