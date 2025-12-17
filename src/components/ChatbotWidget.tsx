import React, { useState, useEffect } from 'react';
import { useThemeConfig } from '@docusaurus/theme-common';
import { ChatKit, useChatKit } from '@openai/chatkit-react';
import styles from './ChatbotWidget.module.css';

// Define the chatbot config type
interface ChatbotConfig {
  enabled?: boolean;
  title?: string;
  initialOpen?: boolean;
  clientToken?: string;
}

// ChatbotWidget component using ChatKit
const ChatbotWidget: React.FC = () => {
  // Get configuration from Docusaurus theme
  const themeConfig: any = useThemeConfig();
  const chatbotConfig: ChatbotConfig = themeConfig.chatkit || {};

  // Check if we're on the client side to handle SSR
  const [isClient, setIsClient] = useState(false);
  const [hasInitialized, setHasInitialized] = useState(false);

  useEffect(() => {
    setIsClient(true);
    setHasInitialized(true);
  }, []);

  // Check if chatbot is enabled
  const chatbotEnabled = chatbotConfig.enabled !== false; // Default to true if not specified

  // State for widget visibility
  const [isOpen, setIsOpen] = useState(chatbotConfig.initialOpen || false);

  // Initialize ChatKit with proper configuration for RAG backend
  const { control, sendUserMessage, focusComposer, setThreadId } = useChatKit({
    api: {
      async getClientSecret(existing) {
        if (existing) {
          // Refresh expired token
          try {
            const res = await fetch('/api/chatkit/refresh', {
              method: 'POST',
              body: JSON.stringify({ token: existing }),
              headers: { 'Content-Type': 'application/json' },
            });
            const data = await res.json();
            return data.client_secret;
          } catch (error) {
            console.error('Error refreshing ChatKit token:', error);
            throw error;
          }
        }

        // Create new session
        try {
          const res = await fetch('/api/chatkit/session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
          });
          const data = await res.json();
          return data.client_secret;
        } catch (error) {
          console.error('Error getting ChatKit session:', error);
          throw error;
        }
      },
    },
    theme: 'light',
    locale: 'en',
    composer: {
      placeholder: 'Ask anything about Physical AI & Humanoid Robotics...',
      tools: [
        {
          id: 'general',
          label: 'General Chat',
          shortLabel: 'Chat',
          icon: 'sparkle',
          placeholderOverride: 'Ask me anything about the book...',
          pinned: true,
        },
        {
          id: 'search',
          label: 'Search Book Content',
          shortLabel: 'Search',
          icon: 'search',
          placeholderOverride: 'What are you looking for in the book?',
          pinned: true,
        }
      ]
    },
    startScreen: {
      greeting: 'Welcome to the Physical AI & Humanoid Robotics Assistant! Ask me anything about the book content.',
      prompts: [
        {
          label: 'Explain a concept',
          prompt: 'Explain ROS 2 in simple terms',
          icon: 'lightbulb',
        },
        {
          label: 'Digital Twin',
          prompt: 'How do digital twins work in robotics?',
          icon: 'square-code',
        },
        {
          label: 'AI Perception',
          prompt: 'What is AI perception in robotics?',
          icon: 'chart',
        },
        {
          label: 'VLA Systems',
          prompt: 'How do Vision-Language-Action systems work?',
          icon: 'notebook-pencil',
        },
      ],
    },
    onError: ({ error }) => {
      console.error('ChatKit error:', error);
    },
    onThreadChange: ({ threadId }) => {
      localStorage.setItem('lastThreadId', threadId || '');
    },
    onReady: () => {
      console.log('ChatKit is ready');
    },
  });

  // Toggle chat widget visibility
  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  // Close chat when pressing Escape key
  useEffect(() => {
    if (!isClient) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, isClient]);

  // Don't render if chatbot is disabled or not on client
  if (!chatbotEnabled || !isClient) {
    return null;
  }

  return (
    <div className={styles.chatContainer}>
      {/* Chat Launcher Button */}
      {!isOpen && (
        <button
          className={styles.chatLauncher}
          onClick={toggleChat}
          aria-label="Open chat"
          aria-expanded={isOpen}
          data-testid="chat-launcher"
        >
          💬
        </button>
      )}

      {/* Chat Widget */}
      {isOpen && (
        <div className={styles.chatKitWidget} role="dialog" aria-modal="true" aria-label="Chat Assistant">
          {/* Chat Header */}
          <div className={styles.widgetHeader} onClick={toggleChat}>
            <h3 className={styles.widgetTitle} id="chat-title">{chatbotConfig.title || 'Book Assistant'}</h3>
            <button
              className={styles.toggleButton}
              onClick={toggleChat}
              aria-label="Close chat"
              aria-controls="chat-messages"
            >
              ×
            </button>
          </div>

          {/* ChatKit Container */}
          <div className={styles.chatKitContainer}>
            {control ? (
              <ChatKit
                control={control}
                className={styles.chatKitFrame}
              />
            ) : (
              <div className={styles.loadingState}>
                <p>Loading chat interface...</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatbotWidget;