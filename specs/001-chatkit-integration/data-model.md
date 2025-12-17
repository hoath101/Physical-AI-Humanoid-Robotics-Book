# Data Model: React Frontend ChatKit Integration

## Entity: Chat Session
**Description**: Represents an active chat interaction that persists while the chat interface is open, containing message history and UI state (from spec FR-003)

**Fields**:
- id: string (unique session identifier)
- createdAt: Date (session start time)
- messages: ChatMessage[] (collection of messages in the session)
- isActive: boolean (whether the session is currently active/open)

**Validation rules**:
- Must have at least one message (the initial system message or first user message)
- Messages must be ordered chronologically
- Session must have a reasonable timeout for cleanup

**State transitions**:
- Created → Active (when chat is opened)
- Active → Inactive (when chat is closed)
- Inactive → Archived (after timeout period)

## Entity: Chat Message
**Description**: Represents individual user or system messages with content, timestamp, and sender type (user/system) (from spec FR-002, FR-003)

**Fields**:
- id: string (unique message identifier)
- content: string (message text content)
- role: 'user' | 'assistant' | 'system' (sender type)
- timestamp: Date (when message was created/sent)
- citations?: Citation[] (optional citation information for assistant responses)
- status: 'sent' | 'sending' | 'error' (message transmission status)

**Validation rules**:
- Content must not be empty or whitespace-only
- Role must be one of the allowed values
- Timestamp must be current or past (not future)
- Citations only applicable for assistant messages

**State transitions**:
- Created → Sending (when message is submitted)
- Sending → Sent (when successfully transmitted)
- Sending → Error (when transmission fails)

## Entity: Citation
**Description**: Represents citation information for assistant responses that reference book content (from existing implementation)

**Fields**:
- chapter?: string (book chapter reference)
- section?: string (book section reference)
- page?: string (book page reference)
- text_snippet?: string (quoted text snippet)
- source_url?: string (optional source URL)

**Validation rules**:
- Must have at least one reference field (chapter, section, page, or text_snippet)
- Text snippet should be reasonably short (less than 200 characters)

## Entity: Chat Interface State
**Description**: Represents the visibility and open/closed status of the chat widget, managed locally in the frontend (from spec FR-008)

**Fields**:
- isOpen: boolean (whether chat widget is visible)
- isMinimized: boolean (whether chat is minimized vs. full size)
- position: { x: number, y: number } (position of chat widget)
- size: { width: number, height: number } (dimensions of chat widget)

**Validation rules**:
- isOpen and isMinimized cannot both be true simultaneously
- Position coordinates must be within viewport bounds
- Size dimensions must be within reasonable limits

**State transitions**:
- Closed → Open (when launcher is clicked while closed)
- Open → Closed (when close button is clicked)
- Open → Minimized (when minimize button is clicked)
- Minimized → Open (when launcher is clicked while minimized)

## API Contract: Chat Communication
**Description**: Defines the interface between ChatKit and the backend API (from spec FR-005)

**Request format** (to POST /api/chat):
```typescript
{
  message: string,           // User's message content
  history?: ChatMessage[],  // Previous conversation history (if needed)
  metadata?: object         // Additional context information
}
```

**Response format** (from POST /api/chat):
```typescript
{
  response: string,         // Assistant's response
  citations?: Citation[],  // Optional citation information
  error?: string           // Optional error message
}
```

**Validation rules**:
- Request message must be provided and non-empty
- Response must include either response text or error
- Citations format must match Citation entity definition