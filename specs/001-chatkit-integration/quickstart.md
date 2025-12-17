# Quickstart: React Frontend ChatKit Integration

## Prerequisites
- Node.js 16+ installed
- npm or yarn package manager
- Existing React + TypeScript project
- Access to backend with POST /api/chat endpoint

## Installation
1. Install OpenAI ChatKit React library:
```bash
npm install @openai/chatkit-react
# or
yarn add @openai/chatkit-react
```

2. Verify TypeScript dependencies are present:
```bash
npm install --save-dev @types/react @types/react-dom
```

## Basic Setup

### 1. Configure ChatProvider
Wrap your application or chat component with the ChatProvider:

```typescript
import { ChatProvider } from '@openai/chatkit-react';

function App() {
  return (
    <ChatProvider /* configuration options */>
      {/* Your chat components */}
    </ChatProvider>
  );
}
```

### 2. Update ChatWidget Component
Replace custom message display and input components with ChatKit equivalents:

```typescript
import {
  MessageList,
  MessageInput,
  useChat
} from '@openai/chatkit-react';

function ChatWidget() {
  // Use ChatKit hooks for chat functionality
  const { messages, sendMessage } = useChat();

  return (
    <div className="chat-container">
      <MessageList messages={messages} />
      <MessageInput onSubmit={sendMessage} />
    </div>
  );
}
```

### 3. Configure Backend Integration
Set up the ChatProvider to use your existing POST /api/chat endpoint by implementing the required adapter functions to map between ChatKit's expected format and your backend API format.

## Environment Variables
Ensure the following environment variables are configured:
- `REACT_APP_API_URL` - Base URL for your backend API
- No OpenAI API keys should be present in frontend

## Running the Application
1. Start the development server:
```bash
npm run dev
# or
yarn dev
```

2. Open your browser to the application
3. Click the floating chat launcher to open the ChatKit-powered chat interface
4. Test sending messages and receiving responses

## Testing
1. Verify the chat launcher opens and closes the chat interface
2. Test sending messages and receiving responses
3. Confirm error handling works when backend is unavailable
4. Check accessibility features (keyboard navigation, ARIA labels)
5. Verify message history persists during the session