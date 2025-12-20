import React, { useState, useEffect, useRef } from 'react';
import { useThemeConfig } from '@docusaurus/theme-common';
import styles from './ChatbotWidget.module.css';

// Define the chatbot config type
interface ChatbotConfig {
  enabled?: boolean;
  title?: string;
  initialOpen?: boolean;
  apiUrl?: string;
  bookId?: string;
}

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: string;
}

// ChatbotWidget component with direct backend integration
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

  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedMode, setSelectedMode] = useState<'global' | 'selection_only'>('global');

  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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

  // Handle sending a message
  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date().toISOString(),
    };

    // Add user message to chat
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Call the backend API
      const response = await fetch(`${chatbotConfig.apiUrl || 'http://localhost:8000'}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: inputValue,
          query_mode: selectedMode,
          book_id: chatbotConfig.bookId || 'default-book',
          selected_text: selectedMode === 'selection_only' ? 'Use provided context only' : undefined,
          top_k: 5,
          temperature: 0.3
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: `ai-${Date.now()}`,
        text: data.answer,
        sender: 'assistant',
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);

      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        text: 'Sorry, I encountered an error processing your request. Please try again.',
        sender: 'assistant',
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle key press (Enter to send)
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Handle quick prompt click
  const handleQuickPrompt = (prompt: string) => {
    setInputValue(prompt);
  };

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
        <div className={styles.chatWidget} role="dialog" aria-modal="true" aria-label="Chat Assistant">
          {/* Chat Header */}
          <div className={styles.widgetHeader}>
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

          {/* Mode Selection */}
          <div className={styles.modeSelector}>
            <button
              className={`${styles.modeButton} ${selectedMode === 'global' ? styles.activeMode : ''}`}
              onClick={() => setSelectedMode('global')}
            >
              Full Book QA
            </button>
            <button
              className={`${styles.modeButton} ${selectedMode === 'selection_only' ? styles.activeMode : ''}`}
              onClick={() => setSelectedMode('selection_only')}
            >
              Search Book
            </button>
          </div>

          {/* Chat Messages */}
          <div className={styles.chatMessages}>
            {messages.length === 0 && (
              <div className={styles.welcomeMessage}>
                <h4>Welcome to the Physical AI & Humanoid Robotics Assistant!</h4>
                <p>Ask me anything about the book content.</p>

                <div className={styles.quickPrompts}>
                  <button onClick={() => handleQuickPrompt('Explain ROS 2 in simple terms')}>
                    Explain ROS 2
                  </button>
                  <button onClick={() => handleQuickPrompt('How do digital twins work in robotics?')}>
                    Digital Twins
                  </button>
                  <button onClick={() => handleQuickPrompt('What is AI perception in robotics?')}>
                    AI Perception
                  </button>
                  <button onClick={() => handleQuickPrompt('How do Vision-Language-Action systems work?')}>
                    VLA Systems
                  </button>
                </div>
              </div>
            )}

            {messages.map((message) => (
              <div
                key={message.id}
                className={`${styles.message} ${styles[message.sender]}`}
                aria-label={`${message.sender} message: ${message.text}`}
              >
                <div className={styles.messageContent}>
                  {message.text}
                </div>
                <div className={styles.messageTimestamp}>
                  {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className={styles.message} aria-label="Assistant is typing">
                <div className={styles.messageContent}>
                  <div className={styles.typingIndicator}>
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className={styles.inputArea}>
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask anything about Physical AI & Humanoid Robotics..."
              className={styles.inputField}
              rows={2}
              disabled={isLoading}
              aria-label="Type your message"
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className={styles.sendButton}
              aria-label="Send message"
            >
              {isLoading ? 'Sending...' : 'Send'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatbotWidget;