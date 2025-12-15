# Physical AI & Humanoid Robotics Book Chatbot Widget Embedding Guide

This guide explains how to embed the RAG-powered chatbot widget into your Docusaurus-based "Physical AI & Humanoid Robotics" book.

## Files Included

- `frontend/embed_snippet.js` - The main chatbot widget JavaScript
- `instructions.md` - This file with embedding instructions

## Embedding in Docusaurus

### Option 1: Adding to All Pages (Recommended)

To add the chatbot to all pages of your Docusaurus site, modify your `docusaurus.config.js` file:

```js
module.exports = {
  // ... other config
  scripts: [
    // Add this line to include the chatbot widget
    {
      src: '/js/chatbot.js', // Path where you'll place the widget
      async: true,
      defer: true,
    },
  ],
  // ... rest of config
};
```

Then place the `embed_snippet.js` file in your static directory:

1. Create a `static/js` directory in your Docusaurus project if it doesn't exist:
   ```
   your-docusaurus-project/static/js/
   ```

2. Copy the `embed_snippet.js` file to:
   ```
   your-docusaurus-project/static/js/chatbot.js
   ```

3. Restart your Docusaurus development server.

### Option 2: Adding to Specific Pages

To add the widget to specific pages only, you can add the script tag directly in the page's markdown frontmatter:

```md
---
title: Your Page Title
scripts:
  - /js/chatbot.js
---
```

### Option 3: Custom React Component

For more control, you can create a custom React component that loads the script:

```jsx
import React, { useEffect } from 'react';

const ChatbotWidget = () => {
  useEffect(() => {
    // Dynamically load the chatbot script
    const script = document.createElement('script');
    script.src = '/js/chatbot.js';
    script.async = true;
    document.body.appendChild(script);

    return () => {
      // Cleanup: remove the script if component unmounts
      document.body.removeChild(script);
    };
  }, []);

  return null; // The widget will appear as a floating element
};

export default ChatbotWidget;
```

## Configuration

You can customize the chatbot behavior by setting configuration options before the script loads. Add this to your Docusaurus configuration:

```js
// In docusaurus.config.js or in a custom HTML file
module.exports = {
  // ... other config
  headTags: [
    {
      tagName: 'script',
      attributes: {
        type: 'text/javascript'
      },
      innerHTML: `
        window.CHATBOT_CONFIG = {
          apiBaseUrl: 'https://your-api-domain.com', // Replace with your FastAPI backend URL
          widgetTitle: 'Physical AI & Robotics Assistant',
          placeholderText: 'Ask about Physical AI, ROS 2, Humanoid Robotics...',
          selectedTextPlaceholder: 'Ask about the selected text...',
          themeColor: '#4f46e5' // Customize the theme color
        };
      `
    }
  ],
  // ... rest of config
};
```

## Environment Configuration

Make sure your FastAPI backend is accessible from your Docusaurus site. If they're on different domains, ensure CORS is properly configured in your FastAPI app:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-docusaurus-site.com"],  # Replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Features

The chatbot widget includes:

1. **Floating Chat Interface**: A circular button in the bottom-right corner that expands to show the chat interface.

2. **Text Selection Mode**:
   - Users can select text on the page
   - When text is selected, the chatbot will only use that selected text to answer questions
   - The input placeholder changes to indicate this mode

3. **Full Book QA Mode**:
   - When no text is selected, the chatbot searches the entire book for relevant information
   - Provides comprehensive answers using the full knowledge base

4. **Responsive Design**: Works well on different screen sizes.

5. **Conversation History**: Maintains a history of the conversation within the session.

## Usage Instructions for Users

1. Click the robot icon (🤖) in the bottom-right corner to open the chatbot.
2. Select text on the page to activate "selected text mode".
3. Type your question in the input field and press Enter or click "Send".
4. The chatbot will respond based on the selected text (if applicable) or the full book content.
5. Close the chatbot by clicking the X button in the top-right corner.

## Customization

You can customize various aspects of the widget:

- `apiBaseUrl`: The URL of your FastAPI backend
- `widgetTitle`: The title displayed in the chatbot header
- `placeholderText`: Placeholder text when no text is selected
- `selectedTextPlaceholder`: Placeholder text when text is selected
- `themeColor`: Primary color for the widget UI

## Troubleshooting

If the chatbot doesn't appear:

1. Check browser console for JavaScript errors
2. Ensure the script is loaded from the correct path
3. Verify that your API endpoint is accessible (check CORS settings)
4. Make sure environment variables are properly set for your backend

## Security Considerations

- Ensure your API endpoint is secured with appropriate authentication if needed
- Consider rate limiting for API requests
- The widget makes requests to your API endpoint, so ensure it's served over HTTPS in production