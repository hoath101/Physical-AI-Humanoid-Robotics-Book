---
id: phr-001-chatkit-integration-tasks-20251215
title: ChatKit Integration Tasks
stage: tasks
date: 2025-12-15
surface: /sp.tasks
model: Claude Sonnet 4.5
feature: 001-chatkit-integration
branch: 001-chatkit-integration
user: Claude
command: /sp.tasks
labels: [tasks, frontend, chatkit, react]
links:
  spec: specs/001-chatkit-integration/spec.md
  plan: specs/001-chatkit-integration/plan.md
  tasks: specs/001-chatkit-integration/tasks.md
  ticket:
  adr:
  pr:
files:
  - path: specs/001-chatkit-integration/tasks.md
    action: created
    summary: Implementation tasks for ChatKit integration
tests:

---

## Prompt

$ARGUMENTS

## Response snapshot

# Implementation Tasks: React Frontend ChatKit Integration

**Feature**: React Frontend ChatKit Integration
**Branch**: `001-chatkit-integration`
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)
**Generated**: 2025-12-15

## Summary

Replace existing custom chat UI components with OpenAI ChatKit components for message display, input, and submission. Maintain the existing floating chat launcher button functionality while integrating with the existing POST /api/chat FastAPI endpoint. Preserve accessibility standards and ensure no OpenAI API keys are exposed in frontend code.

## Implementation Strategy

**MVP Scope**: User Story 1 (Access Chat Interface via Launcher) - Implement ChatKit integration with basic launcher functionality
**Delivery Approach**: Incremental delivery with each user story as a complete, independently testable increment
**Priority Order**: US1 (P1) → US2 (P1) → US3 (P2)
**Parallel Opportunities**: Component development can occur in parallel after foundational setup

---

## Phase 1: Setup & Environment

### Goal
Initialize project environment and install necessary dependencies for ChatKit integration

- [ ] T001 Create feature branch 001-chatkit-integration
- [ ] T002 Install @openai/chatkit-react library in frontend directory
- [ ] T003 Verify build process works with new dependencies
- [ ] T004 Set up API adapter configuration for POST /api/chat endpoint

---

## Phase 2: Foundational Components

### Goal
Establish foundational components and architecture needed for all user stories

- [ ] T005 [P] Create ChatProvider wrapper component in frontend/src/components/ChatProviderWrapper.tsx
- [ ] T006 [P] Update frontend/src/types/chat.ts with ChatKit-compatible type definitions
- [ ] T007 [P] Create API adapter service to map ChatKit format to POST /api/chat endpoint in frontend/src/services/chatAdapter.ts
- [ ] T008 [P] Verify no OpenAI API keys are present in frontend code (security check)

---

## Phase 3: [US1] Access Chat Interface via Launcher

### Goal
User clicks the floating chat launcher button to open the chat widget. The ChatKit-based chat interface appears with a clean, modern UI that replaces the existing custom chat components. The launcher button remains accessible to close the chat.

### Independent Test Criteria
Can be fully tested by clicking the launcher button and verifying the ChatKit interface appears with proper styling and functionality, delivering immediate access to chat capabilities.

- [ ] T009 [P] [US1] Replace ChatWidget.tsx with ChatKit-based implementation using MessageList and MessageInput components
- [ ] T010 [P] [US1] Integrate ChatProvider in ChatWidget component to wrap ChatKit functionality
- [ ] T011 [P] [US1] Update ChatWidget visibility logic to render only when isOpen=true (FR-002)
- [ ] T012 [P] [US1] Connect floating button toggle in ChatLauncher.tsx to ChatWidget mount/unmount (FR-004)
- [ ] T013 [P] [US1] Maintain existing ChatLauncher.tsx functionality for open/close toggle (FR-004)
- [ ] T014 [US1] Test that launcher button opens ChatKit interface when clicked
- [ ] T015 [US1] Test that launcher button closes ChatKit interface when clicked again
- [ ] T016 [US1] Verify ChatKit interface appears with proper styling and accessibility features
- [ ] T017 [US1] Confirm no regression in floating launcher button functionality (SC-007)

---

## Phase 4: [US2] Send and Receive Messages via ChatKit UI

### Goal
User types a message in the ChatKit input field and submits it. The message is sent to the existing FastAPI backend endpoint (POST /api/chat), and the response is displayed in the ChatKit message list. The conversation history persists while the chat is open.

### Independent Test Criteria
Can be fully tested by sending messages through the ChatKit interface and verifying they are properly transmitted to the backend and responses are displayed, delivering the core chat experience.

- [ ] T018 [P] [US2] Configure ChatProvider to use custom API adapter for POST /api/chat endpoint
- [ ] T019 [P] [US2] Implement message submission logic to send messages to POST /api/chat (FR-005)
- [ ] T020 [P] [US2] Update message display to show both user and assistant messages in ChatKit UI
- [ ] T021 [P] [US2] Implement response handling to display assistant messages from backend
- [ ] T022 [P] [US2] Preserve chat message history while the chat interface remains open (FR-003)
- [ ] T023 [P] [US2] Handle citation display for assistant responses (supporting FR-001)
- [ ] T024 [US2] Test message transmission from ChatKit input to POST /api/chat endpoint
- [ ] T025 [US2] Test response display in ChatKit message list after backend response
- [ ] T026 [US2] Verify conversation history persists during the user session (SC-003)
- [ ] T027 [US2] Confirm 95% of chat messages are successfully transmitted and displayed (SC-002)

---

## Phase 5: [US3] Handle API Communication Errors Gracefully

### Goal
When the chat backend is unavailable or returns an error, the ChatKit interface displays appropriate error messages to the user without breaking the UI. Users can retry sending messages when connectivity is restored.

### Independent Test Criteria
Can be tested by simulating backend errors and verifying that users see helpful error messages rather than broken UI, delivering resilience in the chat system.

- [ ] T028 [P] [US3] Implement error handling for network failures in API adapter
- [ ] T029 [P] [US3] Display user-facing error messages for backend connectivity issues
- [ ] T030 [P] [US3] Add retry functionality for failed message submissions
- [ ] T031 [P] [US3] Ensure UI doesn't crash during backend errors (FR-006)
- [ ] T032 [P] [US3] Handle error responses from POST /api/chat endpoint gracefully
- [ ] T033 [US3] Test error message display when backend is unreachable
- [ ] T034 [US3] Test graceful handling of backend error responses
- [ ] T035 [US3] Verify UI remains functional during error scenarios (SC-004)
- [ ] T036 [US3] Confirm 99% of error scenarios are handled gracefully (SC-004)

---

## Phase 6: Polish & Cross-Cutting Concerns

### Goal
Final implementation details, accessibility compliance, and cross-cutting concerns

- [ ] T037 [P] Ensure accessibility standards with proper ARIA labels and keyboard focus (FR-007)
- [ ] T038 [P] Optimize chat interface loading time to meet <1 second access requirement (SC-001)
- [ ] T039 [P] Verify no duplicate message states or parallel message processing (FR-009)
- [ ] T040 [P] Clean up unused custom chat UI components (MessageList.tsx, InputArea.tsx)
- [ ] T041 [P] Remove redundant API calls or utilities from old implementation
- [ ] T042 [P] Run accessibility tests to maintain compliance score (SC-005)
- [ ] T043 [P] Perform final validation of all functional requirements
- [ ] T044 [P] Run TypeScript validation and lint checks
- [ ] T045 [P] Execute end-to-end testing of all user stories

---

## Dependencies

### User Story Completion Order
1. US1 (Access Chat Interface) must be completed before US2 (Send/Receive Messages)
2. US2 must be completed before US3 (Error Handling)
3. All stories require foundational setup (Phase 1 & 2) to be completed first

### Component Dependencies
- ChatProvider → ChatWidget
- API Adapter → ChatProvider
- ChatLauncher ↔ ChatWidget (bidirectional state management)

---

## Parallel Execution Examples

### Per User Story
**US1 Parallel Tasks**:
- T009-T011 (ChatWidget components) can run in parallel with T012-T013 (Launcher integration)

**US2 Parallel Tasks**:
- T018-T019 (API configuration) can run in parallel with T020-T021 (Message handling)

**US3 Parallel Tasks**:
- T028-T029 (Error handling) can run in parallel with T030-T031 (UI resilience)

---

## Task Validation Checklist

- [x] All tasks follow checklist format: `- [ ] T### [USx] Description with file path`
- [x] Task IDs are sequential and unique
- [x] User story tasks have [USx] labels
- [x] Parallelizable tasks have [P] markers
- [x] All tasks include specific file paths where applicable
- [x] Each user story phase has independent test criteria
- [x] Dependencies are clearly identified
- [x] MVP scope identified (US1 tasks)
- [x] Implementation strategy documented

## Outcome

- ✅ Impact: Created comprehensive task list for ChatKit integration feature with 45 specific tasks
- 🧪 Tests:
- 📁 Files: specs/001-chatkit-integration/tasks.md
- 🔁 Next prompts: /sp.implement
- 🧠 Reflection: Tasks organized by user stories with clear dependencies, parallel execution opportunities, and independent test criteria

## Evaluation notes (flywheel)

- Failure modes observed:
- Graders run and results (PASS/FAIL):
- Prompt variant (if applicable):
- Next experiment (smallest change to try):