from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
import time

from api.api.v1 import router as v1_router
from api.config.settings import settings
from api.config.database import lifespan
from api.middleware.error_handler import ErrorHandlerMiddleware
from api.middleware.rate_limiter import RateLimiterMiddleware
from api.utils.logger import log_api_call, app_logger
from starlette.middleware.security import SecurityMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    logger.info("Starting up RAG Chatbot API...")
    # Startup logic can go here
    yield
    # Shutdown logic can go here
    logger.info("Shutting down RAG Chatbot API...")

# Create FastAPI app with lifespan and settings
app = FastAPI(
    title="RAG Chatbot API",
    description="API for querying book content with Retrieval-Augmented Generation",
    version="0.1.0",
    lifespan=lifespan
)

# Add security middleware
app.add_middleware(
    SecurityMiddleware,
    hsts_max_age=31536000,  # 1 year
    hsts_include_subdomains=True,
    hsts_preload=True,
    content_security_policy="default-src 'self'; object-src 'none'; frame-ancestors 'none';",
)

# Add trusted host middleware for additional security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts.split(",")  # Use allowed hosts from settings
)

# Add custom middleware
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RateLimiterMiddleware, requests_limit=100, window_seconds=60)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),  # Use allowed origins from settings
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(v1_router, prefix="/api/v1")

@app.middleware("http")
async def add_process_time_header(request, call_next):
    """
    Middleware to add processing time header and log API calls.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Add processing time to response headers
    response.headers["X-Process-Time"] = str(process_time)

    # Log the API call with timing information
    user_id = request.headers.get("X-User-ID", "anonymous")
    log_api_call(
        endpoint=request.url.path,
        method=request.method,
        duration=process_time,
        user_id=user_id
    )

    return response

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)