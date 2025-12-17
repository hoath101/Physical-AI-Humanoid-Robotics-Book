---
id: phr-001-chatkit-integration-plan-20251215
title: ChatKit Integration Plan
stage: plan
date: 2025-12-15
surface: /sp.plan
model: Claude Sonnet 4.5
feature: 001-chatkit-integration
branch: 001-chatkit-integration
user: Claude
command: /sp.plan
labels: [planning, frontend, chatkit, react]
links:
  spec: specs/001-chatkit-integration/spec.md
  plan: specs/001-chatkit-integration/plan.md
  ticket:
  adr:
  pr:
files:
  - path: specs/001-chatkit-integration/plan.md
    action: created
    summary: Implementation plan for ChatKit integration
  - path: specs/001-chatkit-integration/research.md
    action: created
    summary: Research findings for ChatKit integration
  - path: specs/001-chatkit-integration/data-model.md
    action: created
    summary: Data model for chat entities
  - path: specs/001-chatkit-integration/quickstart.md
    action: created
    summary: Quickstart guide for ChatKit implementation
  - path: specs/001-chatkit-integration/contracts/chat-api.yaml
    action: created
    summary: API contract for chat integration
tests:

---

## Prompt

Create a step-by-step implementation plan to update the existing React + TypeScript frontend to use OpenAI ChatKit for React, based strictly on the approved /sp.specify document.

Inputs

Approved technical specification from /sp.specify

Existing React frontend with:

Floating chat launcher button

Custom chat UI and/or manual fetch/axios logic

Existing backend endpoint:

POST /api/chat (unchanged)

Planning Rules (Must Follow)

Do not write final implementation code

This is a plan, not a build

Each step must be:

Atomic

Ordered

Reversible

Identify risk points and validation checks per phase

Assume the project uses TypeScript

Frontend-only changes

Required Plan Structure
Phase 0 — Pre-Flight Checks

Verify current frontend framework and build tool

Confirm existing chat components and fetch logic

Identify files to be modified or removed

Confirm backend API contract

Phase 1 — Dependency & Environment Preparation

Add @openai/chatkit-react

Verify openai client dependency

Remove unused chat-related dependencies (if any)

Validate build passes after dependency changes

Phase 2 — Component Architecture Setup

Define responsibility boundaries for:

ChatLauncher (floating button)

ChatWidget (ChatKit wrapper)

Decide where ChatProvider will live

Confirm state ownership (open/close vs chat history)

Phase 3 — ChatKit Integration

Configure ChatProvider to use /api/chat

Ensure no API keys exist in frontend

Replace custom message handling with ChatKit

Handle loading and error states

Phase 4 — UI & Interaction Wiring

Connect floating button toggle to ChatWidget mount/unmount

Ensure focus management and accessibility

Preserve existing styling and layout behavior

Phase 5 — Cleanup & Refactor

Remove unused chat UI components

Remove redundant API calls or utilities

Verify no duplicate logic remains

Phase 6 — Validation & Testing

Manual test cases:

Open/close chat

Send messages

Backend error handling

TypeScript validation

Lint and build checks

Risk Identification (Required)

For e

## Response snapshot

# Implementation Plan: React Frontend ChatKit Integration

**Branch**: `001-chatkit-integration` | **Date**: 2025-12-15 | **Spec**: [React Frontend ChatKit Integration Spec](./spec.md)
**Input**: Feature specification from `/specs/001-chatkit-integration/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Replace existing custom chat UI components with OpenAI ChatKit components for message display, input, and submission. Maintain the existing floating chat launcher button functionality while integrating with the existing POST /api/chat FastAPI endpoint. Preserve accessibility standards and ensure no OpenAI API keys are exposed in frontend code.

## Technical Context

**Language/Version**: TypeScript 5.2.2, React 18.2.0
**Primary Dependencies**: React, styled-components, OpenAI ChatKit React library
**Storage**: N/A (client-side only)
**Testing**: Jest with React Testing Library
**Target Platform**: Web browser (modern browsers)
**Project Type**: Web application frontend
**Performance Goals**: <1 second chat interface access, 95% message transmission success rate
**Constraints**: No OpenAI API keys in frontend, maintain accessibility compliance, preserve existing launcher functionality
**Scale/Scope**: Single chat interface per user session, concurrent message handling support

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Technical Accuracy**: Implementation must follow React best practices and ChatKit integration patterns
- **Clarity**: Implementation should maintain code readability and proper documentation
- **Reproducibility**: All changes must be testable and replicable in the development environment
- **Alignment with Physical AI**: N/A - this is a UI enhancement feature
- **Rigor**: Follow TypeScript best practices and React patterns

## Project Structure

### Documentation (this feature)

```text
specs/001-chatkit-integration/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
frontend/
├── src/
│   ├── components/
│   │   ├── ChatLauncher.tsx
│   │   ├── ChatWidget.tsx
│   │   └── chat.module.css
│   ├── context/
│   │   └── ChatContext.tsx
│   ├── types/
│   │   └── chat.ts
│   ├── App.js
│   └── index.js
└── package.json
```

**Structure Decision**: Web application frontend structure selected. The existing React application with custom chat components will be updated to use OpenAI ChatKit components while maintaining the existing component structure and API integration.

## Phase 0 — Pre-Flight Checks

### Verify current frontend framework and build tool
- **Status**: ✅ Confirmed React 18.2.0 + TypeScript 5.2.2 with Webpack build system
- **Files identified**: frontend/src/App.js, frontend/src/components/ChatLauncher.tsx, frontend/src/components/ChatWidget.tsx
- **Build tool**: Webpack with Babel, development server via webpack-dev-server

### Confirm existing chat components and fetch logic
- **Status**: ✅ Existing components identified:
  - ChatLauncher.tsx: Floating button with open/close toggle
  - ChatWidget.tsx: Full chat interface with message display and input
  - ChatContext.tsx: State management and API communication
  - MessageList.tsx and InputArea.tsx: Sub-components
- **Fetch logic**: Custom fetch implementation in ChatContext.tsx using POST /api/v1/query endpoints

### Identify files to be modified or removed
- **To modify**:
  - ChatWidget.tsx: Replace custom UI with ChatKit components
  - ChatContext.tsx: Update API integration to use POST /api/chat
  - App.js: Potentially add ChatProvider wrapper
- **To remove**:
  - Custom message display components when replaced by ChatKit
  - Custom input components when replaced by ChatKit

### Confirm backend API contract
- **Current endpoints**: /api/v1/query and /api/v1/query/selection
- **Required endpoint**: POST /api/chat (as per specification)
- **Action needed**: Create adapter to map ChatKit format to existing backend

## Phase 1 — Dependency & Environment Preparation

### Add @openai/chatkit-react
- **Task**: Install the OpenAI ChatKit React library
- **Command**: `npm install @openai/chatkit-react`
- **Validation**: Verify installation and compatibility with existing React/TypeScript setup

### Verify openai client dependency
- **Task**: Check if additional OpenAI client libraries are needed
- **Status**: ChatKit should handle client functionality; no additional client needed

### Remove unused chat-related dependencies (if any)
- **Current dependencies**: react, react-dom, styled-components, axios
- **Analysis**: Keep existing dependencies; only add ChatKit library
- **Action**: No removals needed initially

### Validate build passes after dependency changes
- **Validation**: Run `npm run build` to ensure no conflicts with new dependencies
- **Expected**: Build should succeed with new ChatKit library

## Phase 2 — Component Architecture Setup

### Define responsibility boundaries for:
- **ChatLauncher (floating button)**:
  - Keep existing functionality unchanged (FR-004)
  - Continue managing open/close state in App.js
  - Toggle visibility of ChatWidget only

- **ChatWidget (ChatKit wrapper)**:
  - Replace custom UI with ChatKit components
  - Integrate ChatProvider within this component
  - Handle visibility based on open state

### Decide where ChatProvider will live
- **Decision**: Place ChatProvider in ChatWidget component to encapsulate ChatKit functionality
- **Alternative considered**: App.js level, but ChatWidget is more appropriate for scope

### Confirm state ownership (open/close vs chat history)
- **Open/close state**: Continue managing in App.js (local state)
- **Chat history**: Managed by ChatKit (replacing current ChatContext implementation)
- **Integration**: Pass open/close state as prop to control ChatWidget visibility

## Phase 3 — ChatKit Integration

### Configure ChatProvider to use /api/chat
- **Task**: Implement custom adapter to connect ChatKit to POST /api/chat endpoint
- **Challenge**: ChatKit may expect different format than existing backend
- **Solution**: Create API adapter layer that maps between formats

### Ensure no API keys exist in frontend
- **Status**: Verified that no OpenAI API keys are present in current implementation
- **Validation**: Confirm ChatKit integration maintains this security requirement

### Replace custom message handling with ChatKit
- **Components to replace**: MessageList.tsx, InputArea.tsx, custom styling
- **Components to keep**: ChatLauncher.tsx (unchanged functionality)
- **Integration**: Use MessageList, MessageInput, and other ChatKit components

### Handle loading and error states
- **Requirement**: Maintain error handling as specified in FR-006
- **Implementation**: Use ChatKit's built-in error states plus custom error handling

## Phase 4 — UI & Interaction Wiring

### Connect floating button toggle to ChatWidget mount/unmount
- **Current behavior**: ChatWidget renders when isOpen=true
- **Maintain**: Same behavior with updated ChatKit implementation
- **Validation**: Toggle functionality works as before

### Ensure focus management and accessibility
- **Requirement**: Maintain accessibility as specified in FR-007
- **Validation**: Verify keyboard navigation, ARIA labels, and screen reader compatibility

### Preserve existing styling and layout behavior
- **Approach**: Use CSS modules or styled-components to maintain visual consistency
- **Adjustment**: Adapt ChatKit components to match existing design language

## Phase 5 — Cleanup & Refactor

### Remove unused chat UI components
- **To remove**: MessageList.tsx, InputArea.tsx (when fully replaced by ChatKit)
- **To keep**: ChatLauncher.tsx, Widget.tsx (if still needed for structure)

### Remove redundant API calls or utilities
- **To remove**: Custom fetch logic in ChatContext.tsx (if fully replaced)
- **To keep**: General context structure if still needed for state management

### Verify no duplicate logic remains
- **Check**: Ensure no parallel message handling systems exist
- **Validation**: Only ChatKit handles messaging functionality

## Phase 6 — Validation & Testing

### Manual test cases:
- **Open/close chat**: Verify launcher button works as before
- **Send messages**: Test message transmission and response display
- **Backend error handling**: Simulate backend errors and verify graceful handling
- **TypeScript validation**: Ensure all type definitions are correct
- **Lint and build checks**: Confirm code quality and build process

### Automated testing:
- **Unit tests**: Update tests for new component structure
- **Integration tests**: Verify ChatKit integration works properly
- **Accessibility tests**: Run automated accessibility checks

## Risk Identification

### High Risk Items:
1. **API Compatibility**: ChatKit may require different request/response format than existing backend
   - **Mitigation**: Create adapter layer to map between formats
   - **Validation**: Test API integration early in Phase 3

2. **Citation Display**: Current implementation shows citations separately; ChatKit may not support this natively
   - **Mitigation**: Customize ChatKit message rendering to include citations
   - **Validation**: Ensure citation information is properly displayed

3. **Styling Integration**: ChatKit components may conflict with existing styled-components approach
   - **Mitigation**: Use ChatKit theming options and custom CSS
   - **Validation**: Verify visual consistency with existing design

### Medium Risk Items:
1. **State Management**: Integrating ChatKit's internal state with existing local state
   - **Mitigation**: Clear separation of concerns between open/close state and chat history
   - **Validation**: Test state transitions don't conflict

2. **Performance**: ChatKit may introduce performance overhead
   - **Mitigation**: Monitor performance metrics during testing
   - **Validation**: Ensure <1 second chat access (SC-001)

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [N/A] | [No violations identified] | [All constitution requirements met] |

## Outcome

- ✅ Impact: Created comprehensive implementation plan for ChatKit integration feature
- 🧪 Tests:
- 📁 Files: specs/001-chatkit-integration/plan.md, specs/001-chatkit-integration/research.md, specs/001-chatkit-integration/data-model.md, specs/001-chatkit-integration/quickstart.md, specs/001-chatkit-integration/contracts/chat-api.yaml
- 🔁 Next prompts: /sp.tasks, /sp.implement
- 🧠 Reflection: Plan addresses all required phases from user input with detailed tasks, risk identification, and validation checkpoints

## Evaluation notes (flywheel)

- Failure modes observed:
- Graders run and results (PASS/FAIL):
- Prompt variant (if applicable):
- Next experiment (smallest change to try):