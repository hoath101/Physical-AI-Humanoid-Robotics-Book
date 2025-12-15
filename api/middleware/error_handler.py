from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback
from typing import Callable, Awaitable

# Set up logging
logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware:
    """
    Middleware to handle errors globally and return consistent error responses.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope)
        response = None

        try:
            # Call the next middleware or route handler
            await self.app(scope, receive, send)
        except RequestValidationError as e:
            # Handle validation errors
            logger.error(f"Validation error: {e}")
            response = JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "message": "Validation error",
                    "details": e.errors()
                }
            )
        except StarletteHTTPException as e:
            # Handle HTTP exceptions
            logger.error(f"HTTP exception: {e.status_code} - {e.detail}")
            response = JSONResponse(
                status_code=e.status_code,
                content={
                    "status": "error",
                    "message": str(e.detail),
                    "status_code": e.status_code
                }
            )
        except Exception as e:
            # Handle all other exceptions
            logger.error(f"Unhandled exception: {str(e)}")
            logger.error(traceback.format_exc())
            response = JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Internal server error",
                    "error_id": f"err_{hash(str(e)) % 1000000}"
                }
            )

        if response:
            await response(scope, receive, send)