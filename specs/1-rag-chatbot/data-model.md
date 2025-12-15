# Data Model: Integrated RAG Chatbot for Published Book

## Overview
This document defines the data models for the RAG chatbot system based on the entities identified in the feature specification.

## Core Entities

### Book Content
- **Fields**:
  - `id`: Unique identifier for the book content segment
  - `text`: The actual content text
  - `chapter`: Chapter identifier/name
  - `page`: Page number
  - `section`: Section identifier
  - `paragraph_id`: Paragraph identifier within the section
  - `book_id`: Reference to the book this content belongs to
  - `embedding_vector`: Vector representation for similarity search
  - `metadata`: Additional metadata (author, date, etc.)

### Query
- **Fields**:
  - `id`: Unique identifier for the query
  - `question`: The user's question text
  - `query_mode`: Enum (GLOBAL | SELECTION_ONLY)
  - `selected_text`: Text selected by user (for selection-only mode)
  - `user_id`: Reference to the user making the query (optional)
  - `timestamp`: When the query was made
  - `session_id`: Reference to the chat session

### Retrieved Context
- **Fields**:
  - `id`: Unique identifier for the retrieved context
  - `content_segments`: Array of relevant book content segments
  - `relevance_scores`: Array of similarity scores for each segment
  - `query_id`: Reference to the original query
  - `metadata`: Additional retrieval metadata

### Response
- **Fields**:
  - `id`: Unique identifier for the response
  - `answer`: The generated answer text
  - `citations`: Array of source references with chapter/page/section info
  - `query_id`: Reference to the original query
  - `timestamp`: When the response was generated
  - `grounding_confidence`: Confidence level in the grounding (0-1)

### Session
- **Fields**:
  - `id`: Unique identifier for the session
  - `user_id`: Reference to the user (optional for anonymous sessions)
  - `start_time`: When the session started
  - `last_activity`: When the last interaction occurred
  - `active`: Boolean indicating if session is currently active

## API Request/Response Models

### Ingestion Request
- `book_content`: Text content of the book
- `book_metadata`: Metadata about the book (title, author, etc.)
- `chunk_size`: Size of text chunks for processing
- `overlap_size`: Overlap between chunks

### Query Request
- `question`: The user's question
- `query_mode`: "global" or "selection_only"
- `selected_text`: Text selected by user (required for selection_only mode)
- `book_id`: Identifier for the book to query

### Query Response
- `answer`: The answer to the question
- `citations`: Array of source citations with location information
- `query_mode`: The mode used for this query
- `status`: "success" or "no_content_found"

## Database Schema (Neon Postgres)

### books table
- `id` (UUID, primary key)
- `title` (VARCHAR)
- `author` (VARCHAR)
- `content_hash` (VARCHAR) - for change detection
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

### sessions table
- `id` (UUID, primary key)
- `user_id` (VARCHAR, optional)
- `book_id` (UUID, foreign key to books)
- `created_at` (TIMESTAMP)
- `last_activity` (TIMESTAMP)
- `active` (BOOLEAN)

### queries table
- `id` (UUID, primary key)
- `session_id` (UUID, foreign key to sessions)
- `question` (TEXT)
- `query_mode` (VARCHAR - 'global' or 'selection_only')
- `selected_text` (TEXT, optional)
- `timestamp` (TIMESTAMP)

### responses table
- `id` (UUID, primary key)
- `query_id` (UUID, foreign key to queries)
- `answer` (TEXT)
- `citations` (JSONB)
- `grounding_confidence` (FLOAT)
- `timestamp` (TIMESTAMP)

## Vector Database Schema (Qdrant)

### Collection: book_content_chunks
- **Vectors**: Embedding vectors of text chunks
- **Payload**:
  - `chunk_id` (keyword)
  - `book_id` (keyword)
  - `text` (text)
  - `chapter` (keyword)
  - `page` (integer)
  - `section` (keyword)
  - `paragraph_id` (keyword)
  - `metadata` (keyword)