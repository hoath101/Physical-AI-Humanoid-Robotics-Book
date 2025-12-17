---
id: phr-001-chatkit-integration-spec-20251215
title: ChatKit Integration Spec
stage: spec
date: 2025-12-15
surface: /sp.specify
model: Claude Sonnet 4.5
feature: 001-chatkit-integration
branch: 001-chatkit-integration
user: Claude
command: /sp.specify
labels: [specification, frontend, chatkit, react]
links:
  spec: specs/001-chatkit-integration/spec.md
  ticket:
  adr:
  pr:
files:
  - path: specs/001-chatkit-integration/spec.md
    action: created
    summary: Technical specification for ChatKit integration
  - path: specs/001-chatkit-integration/checklists/requirements.md
    action: created
    summary: Quality checklist for specification
tests:

---

## Prompt

You are a senior product engineer and technical spec author.

Purpose

Produce a complete technical specification for updating an existing React frontend to use OpenAI ChatKit for React as the chat UI, integrated with an existing FastAPI backend that already handles OpenAI + RAG logic.

Context

The project already has:

A React-based frontend

A floating chat launcher button

A FastAPI backend endpoint for chat (POST /api/chat)

The frontend currently uses custom chat UI and/or manual fetch/axios logic

The backend must remain unchanged

In Scope

Frontend-only changes

Dependency updates

ChatKit integration

UI wiring and state management

Out of Scope

Backend changes

OpenAI prompt engineering

RAG logic changes

Database or vector store changes

Functional Requirements

Chat UI

Replace existing message list, input, and submit logic with OpenAI ChatKit components

Chat UI must render only when the chat is opened

Messages must persist while the chat is open

Launcher Behavior

Preserve existing floating button

Toggle chat open/close on click

No regression in accessibility (ARIA labels, keyboard focus)

API Communication

All chat requests must go to POST /api/chat

Frontend must not contain any OpenAI API key

Error states must be handled gracefully (network failure, backend errors)

State Management

Chat open/close state handled locally

Chat history managed by ChatKit

No duplicate or parallel message state

Non-Functional Requirements

Use TypeScript

No new UI frameworks

No duplicate chat logic

No unused dependencies

Maintain existing CSS modules / styling approach

Compatible with modern browsers

Dependency Requirements

Add:

@openai/chatkit-react

Do not install:

chatkit

@pusher/chatkit

Remove any unused chat-related dependencies

Technical Constraints

ChatKit must be wrapped in ChatProvider

Backend URL must be configurable (relative path preferred)

Chat component must be isolated and reusable

No global side effects

Deliverables

Updated dependency list

Component-level design

ChatLau

## Response snapshot

# Feature Specification: React Frontend ChatKit Integration

**Feature Branch**: `001-chatkit-integration`
**Created**: 2025-12-15
**Status**: Draft
**Input**: User description: "Update existing React frontend to use OpenAI ChatKit for React as the chat UI, integrated with existing FastAPI backend that already handles OpenAI + RAG logic"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Access Chat Interface via Launcher (Priority: P1)

User clicks the floating chat launcher button to open the chat widget. The ChatKit-based chat interface appears with a clean, modern UI that replaces the existing custom chat components. The launcher button remains accessible to close the chat.

**Why this priority**: This is the foundational interaction that enables all other chat functionality. Without the ability to access the chat interface, no other features are usable.

**Independent Test**: Can be fully tested by clicking the launcher button and verifying the ChatKit interface appears with proper styling and functionality, delivering immediate access to chat capabilities.

**Acceptance Scenarios**:

1. **Given** user is on any page of the application, **When** user clicks the floating chat launcher button, **Then** the ChatKit-based chat interface appears with proper styling and accessibility features
2. **Given** chat interface is open, **When** user clicks the launcher button again, **Then** the chat interface closes and only the launcher button remains visible

### User Story 2 - Send and Receive Messages via ChatKit UI (Priority: P1)

User types a message in the ChatKit input field and submits it. The message is sent to the existing FastAPI backend endpoint (POST /api/chat), and the response is displayed in the ChatKit message list. The conversation history persists while the chat is open.

**Why this priority**: This is the core functionality that provides value to users - the ability to have a conversation with the system using the enhanced ChatKit UI.

**Independent Test**: Can be fully tested by sending messages through the ChatKit interface and verifying they are properly transmitted to the backend and responses are displayed, delivering the core chat experience.

**Acceptance Scenarios**:

1. **Given** chat interface is open and user is on input field, **When** user types a message and submits, **Then** message appears in the chat history and is sent to POST /api/chat endpoint
2. **Given** message was sent to backend, **When** response is received from backend, **Then** response appears in the chat history below the user's message

### User Story 3 - Handle API Communication Errors Gracefully (Priority: P2)

When the chat backend is unavailable or returns an error, the ChatKit interface displays appropriate error messages to the user without breaking the UI. Users can retry sending messages when connectivity is restored.

**Why this priority**: Error handling is crucial for maintaining a professional user experience and preventing confusion when technical issues occur.

**Independent Test**: Can be tested by simulating backend errors and verifying that users see helpful error messages rather than broken UI, delivering resilience in the chat system.

**Acceptance Scenarios**:

1. **Given** backend is unreachable, **When** user attempts to send a message, **Then** user sees a clear error message and can attempt to retry
2. **Given** backend returns error response, **When** error is received, **Then** appropriate error handling occurs without breaking the chat interface

### Edge Cases

- What happens when the user opens the chat, sends messages, closes it, then reopens - conversation history should be preserved during the session?
- How does the system handle very long messages or messages with special characters?
- What occurs when the user has multiple tabs open and interacts with chat in both?
- How does the system handle network timeouts or intermittent connectivity?
- What happens when the user navigates to different pages while chat is open?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST replace existing custom chat UI components with OpenAI ChatKit components for message display, input, and submission
- **FR-002**: System MUST render the ChatKit chat interface only when the chat is opened via the floating launcher button
- **FR-003**: System MUST preserve chat message history while the chat interface remains open during the user session
- **FR-004**: System MUST maintain the existing floating chat launcher button functionality with toggle behavior for opening/closing the chat
- **FR-005**: System MUST send all chat requests to the existing POST /api/chat FastAPI endpoint without modification
- **FR-006**: System MUST handle API communication errors gracefully with appropriate user-facing error messages for network failures and backend errors
- **FR-007**: System MUST maintain accessibility standards with proper ARIA labels and keyboard focus management without regression
- **FR-008**: System MUST manage chat open/close state locally without affecting other application state
- **FR-009**: System MUST ensure no duplicate message states or parallel message processing occurs during chat interactions
- **FR-010**: System MUST integrate ChatKit within a ChatProvider wrapper as required by the library

### Key Entities

- **Chat Session**: Represents an active chat interaction that persists while the chat interface is open, containing message history and UI state
- **Chat Message**: Represents individual user or system messages with content, timestamp, and sender type (user/system)
- **Chat Interface State**: Represents the visibility and open/closed status of the chat widget, managed locally in the frontend

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can access the ChatKit-based chat interface within 1 second of clicking the launcher button
- **SC-002**: 95% of chat messages are successfully transmitted to the backend and responses are displayed without errors
- **SC-003**: Users can maintain an active chat session with message history preserved while navigating the application
- **SC-004**: Error handling prevents UI crashes during backend connectivity issues, with 99% of error scenarios handled gracefully
- **SC-005**: The new ChatKit integration maintains the same accessibility compliance score as the previous implementation (measured by automated accessibility testing tools)
- **SC-006**: The chat feature has zero OpenAI API keys exposed in frontend code, maintaining security compliance
- **SC-007**: All existing functionality of the floating chat launcher is preserved without regression

## Outcome

- ✅ Impact: Created complete technical specification for ChatKit integration feature
- 🧪 Tests:
- 📁 Files: specs/001-chatkit-integration/spec.md, specs/001-chatkit-integration/checklists/requirements.md
- 🔁 Next prompts: /sp.plan, /sp.tasks
- 🧠 Reflection: Specification addresses all requirements from user input with clear user stories, functional requirements, and success criteria

## Evaluation notes (flywheel)

- Failure modes observed:
- Graders run and results (PASS/FAIL):
- Prompt variant (if applicable):
- Next experiment (smallest change to try):