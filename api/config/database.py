from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
import logging

from .settings import settings

# Set up logging
logger = logging.getLogger(__name__)

# Neon Postgres database setup
SQLALCHEMY_DATABASE_URL = settings.neon_database_url

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,    # Recycle connections every 5 minutes
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session
async def get_db() -> AsyncGenerator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for database connections.
    """
    logger.info("Initializing database connections...")
    # Startup logic can go here
    yield
    # Shutdown logic can go here
    logger.info("Closing database connections...")