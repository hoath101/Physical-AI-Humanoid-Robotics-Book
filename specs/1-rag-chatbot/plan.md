# Implementation Plan: Integrated RAG Chatbot for Published Book

**Branch**: `1-rag-chatbot` | **Date**: 2025-12-15 | **Spec**: [specs/1-rag-chatbot/spec.md](./spec.md)
**Input**: Feature specification from `/specs/1-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

The Integrated RAG Chatbot will provide a conversational interface for users to ask questions about book content with strict grounding in retrieved text. Based on research, the system will implement a dual-query mode approach (Global and Selection-Only) using FastAPI backend with Qdrant vector storage. The solution will ensure strict grounding to prevent hallucinations, provide proper citations from book content, and respond appropriately when content is unavailable. The frontend will be designed as an embeddable component for the Docusaurus-based book reader.

## Technical Context

**Language/Version**: Python 3.11, JavaScript/TypeScript for frontend integration
**Primary Dependencies**: FastAPI, OpenAI SDK, Qdrant client, Neon Postgres client, Pydantic
**Storage**: Qdrant Cloud (vector database), Neon Serverless Postgres (relational data), local file system (book content)
**Testing**: pytest for backend, Jest for frontend components, integration tests for RAG pipeline
**Target Platform**: Web-based Docusaurus documentation site with embedded chatbot widget
**Project Type**: Web application (backend API + frontend integration)
**Performance Goals**: <5 second response time for queries, 95% accuracy for factually correct responses
**Constraints**: Free tier limitations for Qdrant and Neon, <200ms p95 for API endpoints, offline-capable frontend
**Scale/Scope**: Single book corpus initially, potential for multiple books, 1000+ concurrent users

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Technical Accuracy**: The RAG chatbot must provide factually accurate responses based on the book content, aligning with the constitution's requirement for technical accuracy. The system will ground responses strictly in retrieved text to prevent hallucinations.

**Clarity**: The chatbot interface must be understandable for students with computer science or AI backgrounds, using appropriate technical terminology while remaining accessible.

**Reproducibility**: The implementation must be replicable, with clear documentation for setting up the RAG pipeline, API endpoints, and frontend integration.

**Alignment with Physical AI**: While the chatbot itself is not a physical AI system, it will support the educational mission by helping students understand the book content related to bridging AI systems with physical humanoid robots.

**Rigor**: The RAG implementation will follow best practices for retrieval-augmented generation, using proper citation and grounding techniques.

**Standards Compliance**: The system will use proper citation formats and maintain academic rigor in how it presents information from the book.

**Constraints Check**: The implementation will work within the specified technology stack (FastAPI, Qdrant, Neon Postgres) and meet the performance requirements (response time <5 seconds).

## Project Structure

### Documentation (this feature)

```text
specs/1-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
api/
├── main.py              # FastAPI application entry point
├── config/
│   ├── settings.py      # Application settings and configuration
│   └── database.py      # Database connection setup
├── models/
│   ├── request.py       # Request schemas (Pydantic models)
│   ├── response.py      # Response schemas (Pydantic models)
│   └── document.py      # Document and content models
├── services/
│   ├── ingestion.py     # Document ingestion and processing service
│   ├── retrieval.py     # RAG retrieval service
│   ├── embedding.py     # Embedding generation service
│   └── chat.py          # Chat and response generation service
├── api/
│   ├── v1/
│   │   ├── routes/
│   │   │   ├── ingest.py    # Ingestion endpoints
│   │   │   ├── query.py     # Query endpoints (global and selection modes)
│   │   │   └── health.py    # Health check endpoints
│   │   └── __init__.py
├── utils/
│   ├── validators.py    # Input validation utilities
│   ├── citations.py     # Citation generation utilities
│   └── text_processing.py # Text chunking and processing utilities
└── tests/
    ├── unit/
    ├── integration/
    └── contract/

frontend/
├── src/
│   ├── components/
│   │   ├── ChatbotWidget.tsx    # Main chatbot UI component
│   │   ├── MessageList.tsx      # Chat message display
│   │   └── InputArea.tsx        # User input area
│   ├── services/
│   │   ├── apiClient.ts         # API communication layer
│   │   └── chatService.ts       # Chat business logic
│   ├── types/
│   │   └── chat.ts              # TypeScript type definitions
│   └── hooks/
│       └── useChat.ts           # Chat state management
└── tests/
    └── components/
```

**Structure Decision**: Web application structure with separate backend API and frontend components to support the Docusaurus integration. The backend handles RAG processing, embeddings, and API endpoints, while the frontend provides the chatbot UI that can be embedded in the book reader.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
