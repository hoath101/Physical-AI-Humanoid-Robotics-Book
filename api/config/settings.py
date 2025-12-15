from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # API Configuration
    api_key: str = os.getenv("API_KEY", "your_backend_api_key")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")

    # Database Configuration
    neon_database_url: str = os.getenv("NEON_DATABASE_URL", "")

    # Application Settings
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "info")

    # RAG Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    overlap_size: int = int(os.getenv("OVERLAP_SIZE", "200"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1000"))

    # Server Configuration
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Security Configuration
    allowed_origins: str = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://localhost:3000,http://127.0.0.1,http://127.0.0.1:3000")
    allowed_hosts: str = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1,0.0.0.0,[::1]")

    class Config:
        env_file = ".env"
        case_sensitive = True

# Create a singleton instance of settings
settings = Settings()