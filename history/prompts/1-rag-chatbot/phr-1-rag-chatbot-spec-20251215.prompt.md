---
id: phr-1-rag-chatbot-spec-20251215
title: Integrated RAG Chatbot Specification
stage: spec
date: 2025-12-15
surface: /sp.specify
model: Claude Sonnet 4.5
feature: rag-chatbot
branch: 1-rag-chatbot
user: Claude Code User
command: /sp.specify
labels: [specification, rag, chatbot, book]
links:
  spec: specs/1-rag-chatbot/spec.md
  ticket:
  adr:
  pr:
files:
- path: specs/1-rag-chatbot/spec.md
  action: created
  summary: Feature specification for integrated RAG chatbot
- path: specs/1-rag-chatbot/checklists/requirements.md
  action: created
  summary: Quality checklist for specification validation
tests:

---

## Prompt

Task Title: Integrated RAG Chatbot for Published Book

Objective:
Design and specify an end-to-end Retrieval-Augmented Generation (RAG) chatbot embedded within a published digital book. The chatbot must answer user questions about the book's content with high factual accuracy and strict grounding in retrieved text.

Scope & Requirements:

Core Functionality

The chatbot must answer questions about the book using a RAG pipeline.

Responses must be grounded strictly in retrieved book content.

The system must support two query modes:

Global Mode: Answer using the entire book corpus.

Selection-Only Mode: Answer using only the text explicitly selected by the user.

If the answer cannot be found in the provided context, the chatbot must respond with a clear "not found in source text" message.

Technology Stack (Mandatory)

LLM & Agent Layer: OpenAI Agents SDK / ChatKit SDK

Backend API: FastAPI (Python)

Vector Database: Qdrant Cloud (Free Tier)

Relational Database: Neon Serverless Postgres

Embeddings: OpenAI embeddings API

Environment Management: .env-based configuration

RAG Architecture

Document ingestion pipeline for book content:

Chunking strategy (token-aware, overlap included)

Metadata enrichment (chapter, page, section, paragraph ID)

Embedding storage in Qdrant with namespace/collection separation.

Retrieval logic with configurable top_k, score thresholding, and optional re-ranking.

Context assembly rules that enforce:

Selection-only isolation when user highlights text

Source citation per response

API Design

FastAPI endpoints including:

/ingest - ingest and index book content

/query - global RAG query

/query/selection - selection-restricted RAG query

Clear request/response schemas using Pydantic.

Proper error handling and logging.

Chatbot Behavior

Deterministic, non-hallucinating answers.

Explicit citation of source passages.

Clear refusal behavior when context is insufficient.

Low-latency response suitable for in-book interaction.

Frontend Integration (Conceptual)

Embedded chatbot UI within the book reader.

Ability to pass:

User question

Selected text (if any)

Book/document identifiers

Streaming or near-real-time responses.

Security & Performance

API key management via environment variables.

Rate-limiting considerations.

Stateless FastAPI design suitable for serverless deployment.

Deliverables

System architecture diagram (described in text)

Detailed RAG workflow

FastAPI endpoint specifications

Data models for Postgres and Qdrant

Example request/response payloads

Pseudocode or reference implementation snippets

Constraints

Must be compatible with free tiers of Neon and Qdrant.

Must not rely on proprietary or unavailable services beyond those listed.

Must be production-oriented, not a toy example.

Success Criteria

The chatbot reliably answers book-specific questions.

Selection-only mode strictly limits answers to highlighted text.

No hallucinated content outside retrieved sources.

Architecture is scalable and maintainable.

## Response snapshot

Created feature specification for integrated RAG chatbot with the following key elements:

- User stories prioritized by importance (P1-P3)
- Functional requirements focusing on user capabilities
- Success criteria with measurable outcomes
- Key entities without implementation details
- Edge cases identified
- Quality checklist completed

Specification addresses both global and selection-only query modes, response grounding requirements, and citation functionality as specified in the original requirements.

## Outcome

- ✅ Impact: Created comprehensive feature specification for RAG chatbot
- 🧪 Tests:
- 📁 Files: specs/1-rag-chatbot/spec.md, specs/1-rag-chatbot/checklists/requirements.md
- 🔁 Next prompts: /sp.plan, /sp.tasks
- 🧠 Reflection: Successfully created technology-agnostic specification that focuses on user value

## Evaluation notes (flywheel)

- Failure modes observed: PowerShell script execution failed due to environment limitations
- Graders run and results (PASS/FAIL): Specification quality checklist completed with all items passing
- Prompt variant (if applicable):
- Next experiment (smallest change to try): Proceed to planning phase