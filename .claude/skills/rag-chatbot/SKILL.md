---
name: book-rag-chatbot
description: Design, implement, and embed a Retrieval-Augmented Generation (RAG) chatbot for a published book using OpenAI Agents/ChatKit SDKs, FastAPI, Neon Serverless Postgres, and Qdrant Cloud. Supports full-book QA and user-selected-text-only QA.
allowed-tools: Read, Write, Edit, Grep, Glob, Bash
---

# Integrated Book RAG Chatbot Skill

## Purpose
This skill builds an end-to-end Retrieval-Augmented Generation (RAG) chatbot embedded inside a published book or book website. The chatbot answers reader questions using the book’s content and can optionally restrict answers strictly to user-selected text.

## Core Capabilities
- Book ingestion and chunking
- Vector indexing in Qdrant Cloud (Free Tier)
- Metadata storage in Neon Serverless Postgres
- OpenAI Agents / ChatKit-based reasoning
- FastAPI backend
- Embedded frontend widget
- Context-restricted answering (selected-text-only mode)

## Technology Stack
- OpenAI Agents SDK / ChatKit SDK
- FastAPI (Python)
- Neon Serverless Postgres
- Qdrant Cloud (Free Tier)
- JavaScript embed widget

## Operating Modes
1. **Global Book QA**
   - Retrieve relevant chunks from the full book
   - Answer using retrieved context only

2. **Selected Text QA**
   - User highlights text in the book
   - Answer strictly using the provided selection
   - No external retrieval allowed

## High-Level Workflow
1. Ingest book content and generate embeddings
2. Store embeddings in Qdrant with chapter/page metadata
3. Store relational metadata in Neon Postgres
4. Use an OpenAI Agent to:
   - Decide retrieval strategy
   - Enforce context constraints
5. Serve via FastAPI
6. Embed chatbot into the book frontend

## Behavioral Constraints
- Never hallucinate outside retrieved or user-provided context
- If answer is not found, respond with a clear limitation
- Selected-text mode must ignore vector search entirely

## Files to Use
- Architecture: `architecture/system-overview.md`
- API: `api/fastapi_app.py`
- RAG Logic: `rag/`
- Database Setup: `db/`
- Frontend Embed: `frontend/embed_snippet.js`

## Output Expectations
When asked to implement or modify this system, produce:
- Clean, modular Python code
- Clear API contracts
- Secure database access patterns
- Minimal, embeddable frontend code
