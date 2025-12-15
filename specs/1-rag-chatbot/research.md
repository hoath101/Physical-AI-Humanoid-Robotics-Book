# Research Summary: Integrated RAG Chatbot for Published Book

## Overview
This research document addresses the technical requirements and implementation approach for the Integrated RAG Chatbot feature, focusing on the requirements from the feature specification.

## Key Decisions Made

### 1. Architecture Pattern
**Decision**: Implement a web application architecture with separate backend API and frontend components
**Rationale**: This approach allows for clear separation of concerns, with the backend handling complex RAG processing and the frontend providing a clean UI that can be embedded in the Docusaurus book reader
**Alternatives considered**:
- Monolithic application: Less maintainable and harder to scale
- Pure client-side solution: Would require exposing sensitive APIs and handling complex processing in browser

### 2. RAG Implementation Strategy
**Decision**: Use a dual-query mode system with global and selection-only modes
**Rationale**: This directly addresses the functional requirement (FR-003) to support both query types, providing flexibility for different user needs
**Implementation**:
- Global mode: Query entire book corpus stored in vector database
- Selection-only mode: Filter retrieved results to only include the user-selected text segments

### 3. Vector Database Strategy
**Decision**: Use Qdrant Cloud for vector storage with metadata filtering
**Rationale**: Supports the required metadata (chapter, page, section, paragraph ID) for proper citations and filtering for selection-only mode
**Implementation**: Store book content chunks with structural metadata as payload in Qdrant vectors

### 4. Grounding and Citation Approach
**Decision**: Implement strict grounding with explicit citation extraction
**Rationale**: Required by FR-004 (no hallucination) and FR-005 (explicit citations)
**Implementation**:
- Use retrieved context as the only source for LLM generation
- Extract and present source metadata (chapter, page, section) as citations
- Implement "not found in source text" response when no relevant context is retrieved

### 5. API Design
**Decision**: Design RESTful FastAPI endpoints following best practices
**Rationale**: FastAPI provides excellent documentation, validation, and async support needed for RAG operations
**Endpoints planned**:
- POST /api/v1/ingest - for book content ingestion
- POST /api/v1/query - for global mode queries
- POST /api/v1/query/selection - for selection-only mode queries

## Technology Research

### FastAPI for Backend
FastAPI chosen for:
- Built-in async support for I/O operations
- Automatic OpenAPI documentation
- Pydantic integration for request/response validation
- Performance comparable to Node.js/Go for I/O-bound operations

### Qdrant Vector Database
Qdrant chosen for:
- Rich filtering capabilities needed for selection-only mode
- Metadata storage and querying
- Cloud tier compatibility with project constraints
- Python SDK with good integration options

### Frontend Integration
Frontend will be implemented as:
- React component that can be embedded in Docusaurus
- State management for chat sessions
- API communication layer
- Text selection detection and context passing

## Risk Mitigation

### Latency Concerns
- Use of efficient embedding models and vector search
- Caching strategies for frequent queries
- Optimized chunking strategy to reduce retrieval time

### Free Tier Limitations
- Plan for data size limits in Qdrant and Neon
- Efficient data structures to maximize free tier usage
- Monitoring and alerting for usage limits

### Hallucination Prevention
- Strict input validation
- Grounding enforcement in the response generation pipeline
- Proper citation of source material