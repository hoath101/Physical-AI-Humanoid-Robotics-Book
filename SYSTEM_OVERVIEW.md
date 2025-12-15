# Physical AI & Humanoid Robotics Book RAG Chatbot - System Overview

## Architecture

The system consists of three main components that work together to provide a RAG-powered chatbot for the Physical AI & Humanoid Robotics book:

### 1. Backend API (FastAPI)
- **Location**: `api/fastapi_app.py` and `main.py`
- **Purpose**: Handles chat requests, manages conversation flow, and coordinates between components
- **Endpoints**:
  - `/chat` - Main endpoint for processing questions
  - `/ingest` - Triggers the book ingestion pipeline
  - `/ingestion-status` - Gets system status
  - `/health` - Health check
  - `/info` - System information

### 2. Ingestion Pipeline (Python)
- **Location**: `rag/ingestion.py`
- **Purpose**: Processes book content, chunks it, generates embeddings, and stores in vector DB
- **Features**:
  - Text chunking with configurable size and overlap
  - OpenAI embedding generation
  - Bulk operations for efficient processing
  - Metadata extraction and storage

### 3. Data Storage Layer
- **Vector Database**: Qdrant (for semantic search)
  - Stores text embeddings for fast similarity search
  - Supports filtering by book sections
  - Enables full-book QA mode
- **Relational Database**: Neon Postgres (for metadata)
  - Stores document metadata, chunk information
  - Tracks question-answer interactions
  - Maintains relational data about the book structure

### 4. Frontend Widget (JavaScript)
- **Location**: `frontend/embed_snippet.js`
- **Purpose**: Provides a user-friendly interface for asking questions
- **Features**:
  - Floating chat interface
  - Text selection detection
  - Selected-text-only and full-book QA modes
  - Responsive design

## End-to-End Flow

### Full-Book QA Mode
1. User asks a question without selecting text
2. API generates embedding for the question
3. Qdrant performs semantic search across book chunks
4. Top-k relevant chunks are retrieved with metadata
5. OpenAI generates response using retrieved context
6. Response and interaction are logged to Neon Postgres

### Selected-Text-Only Mode
1. User selects text on the page and asks a question
2. Selected text is sent to the API along with the question
3. OpenAI generates response using only the selected text
4. No vector search is performed
5. Interaction is logged to Neon Postgres

## Data Flow

```
Book Content (Markdown) → Chunking → Embedding Generation → Qdrant (Vector Storage)
                                                                 ↓
User Question → Embedding → Qdrant Search → Retrieved Chunks → OpenAI Response
                              ↓
                        Neon Postgres (Metadata & Logs)
```

## Configuration

### Environment Variables (.env file)
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-ada-002

# Qdrant Configuration
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=book_chunks

# Neon Postgres Configuration
NEON_DB_URL=your_neon_db_url_here

# Ingestion Configuration
BOOK_DIRECTORY=docs
MAX_CHUNK_SIZE=1000
OVERLAP=100
BATCH_SIZE=50
MAX_WORKERS=5
```

## Error Handling

The system includes comprehensive error handling:

1. **API Level**: HTTP exceptions with appropriate status codes
2. **Database Level**: Connection pooling and retry logic
3. **Embedding Level**: Fallback responses when embeddings fail
4. **UI Level**: Graceful degradation when API is unavailable

## Validation

The system validates:
- Required environment variables are set
- Book directory exists before ingestion
- Question length and format
- API response format
- Database connection status

## Performance Considerations

1. **Caching**: Qdrant provides fast vector similarity search
2. **Batching**: Ingestion pipeline uses bulk operations
3. **Connection Pooling**: Neon Postgres uses connection pooling
4. **Async Processing**: All operations are asynchronous where possible

## Security

1. **API Keys**: Stored in environment variables, not code
2. **CORS**: Configurable CORS settings for web integration
3. **Input Validation**: All user inputs are validated
4. **Rate Limiting**: Can be added at the infrastructure level