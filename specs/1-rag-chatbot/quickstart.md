# Quickstart Guide: Integrated RAG Chatbot

## Overview
This guide provides a quick setup and usage guide for the RAG Chatbot API.

## Prerequisites
- Python 3.11+
- OpenAI API key
- Qdrant Cloud account and API key
- Neon Serverless Postgres account and connection string

## Environment Setup
Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DATABASE_URL=your_neon_postgres_connection_string
API_KEY=your_backend_api_key
```

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables
4. Start the FastAPI server: `uvicorn main:app --reload`

## Basic Usage

### 1. Ingest a Book
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "book_content": "Your book content here...",
    "book_metadata": {
      "title": "Book Title",
      "author": "Author Name"
    },
    "book_id": "unique-book-id"
  }'
```

### 2. Query in Global Mode
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main concepts?",
    "query_mode": "global",
    "book_id": "unique-book-id"
  }'
```

### 3. Query in Selection-Only Mode
```bash
curl -X POST http://localhost:8000/api/v1/query/selection \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain this concept?",
    "query_mode": "selection_only",
    "selected_text": "The concept of embodied intelligence...",
    "book_id": "unique-book-id"
  }'
```

## Frontend Integration
To embed the chatbot in your Docusaurus site:
1. Include the chatbot widget component
2. Configure the API endpoint URL
3. Pass selected text when users highlight content
4. Display responses with proper citations