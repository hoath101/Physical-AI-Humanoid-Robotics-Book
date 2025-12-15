import React, { useState, useRef, useEffect } from 'react';
import clsx from 'clsx';
import styles from './ChatbotWidget.module.css';

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
}

const ChatbotWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: 'Hello! I\'m your Physical AI & Humanoid Robotics assistant. Ask me about ROS 2, Digital Twins, AI Perception, or Vision-Language-Action systems!',
      sender: 'bot'
    }
  ]);
  const [inputMessage, setInputMessage] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Add keyboard shortcuts and focus management
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Toggle chat with Ctrl/Cmd + K
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(!isOpen);
        if (!isOpen && inputRef.current) {
          inputRef.current.focus();
        }
      }

      // Close chat with Escape key
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  // Focus management when chat opens/closes
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Call the RAG API to get a response based on book content
      // In production, this would be your actual API endpoint
      const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await fetch(`${API_BASE_URL}/api/rag-query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: inputMessage }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();

      const botResponse: Message = {
        id: Date.now() + 1,
        text: data.response,
        sender: 'bot'
      };

      setMessages(prev => [...prev, botResponse]);
      setIsLoading(false);
    } catch (error) {
      console.error('Error getting response from RAG API:', error);
      const errorMessage: Message = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error processing your question. Please try again.',
        sender: 'bot'
      };
      setMessages(prev => [...prev, errorMessage]);
      setIsLoading(false);
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const quickQuestions = [
    "What is ROS 2?",
    "Explain Digital Twins",
    "How does AI Perception work?",
    "What are VLA systems?"
  ];

  const handleQuickQuestion = (question: string) => {
    setInputMessage(question);
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  return (
    <>
      {/* Simple chatbot button - always visible at bottom right */}
      <button
        className={clsx(styles.chatLauncher, styles.floatingButton)}
        onClick={toggleChat}
        aria-label="Open chatbot assistant"
        title="Open chatbot (Ctrl+K)"
      >
        <span className={styles.chatButtonText}>Chat</span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={styles.chatButtonIcon} aria-hidden="true">
          <path d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H16L14.5 20.5C14.2674 21.0008 13.8335 21.3857 13.3033 21.5721C12.7731 21.7585 12.1871 21.7315 11.6777 21.499C11.1684 21.2665 10.7757 20.8517 10.587 20.342C10.3983 19.8323 10.429 19.2677 10.671 18.758L11.5 17H7C6.46957 17 5.96086 16.7893 5.58579 16.4142C5.21071 16.0391 5 15.5304 5 15V5C5 4.46957 5.21071 3.96086 5.58579 3.58579C5.96086 3.21071 6.46957 3 7 3H17C17.5304 3 18.0391 3.21071 18.4142 3.58579C18.7893 3.96086 19 4.46957 19 5V15Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M9 7H15" stroke="white" strokeWidth="2" strokeLinecap="round"/>
          <path d="M9 11H15" stroke="white" strokeWidth="2" strokeLinecap="round"/>
          <path d="M9 15H13" stroke="white" strokeWidth="2" strokeLinecap="round"/>
        </svg>
      </button>

      {/* Chatbot modal */}
      {isOpen && (
        <div
          className={styles.chatContainer}
          ref={chatContainerRef}
          role="dialog"
          aria-modal="true"
          aria-label="Physical AI Assistant Chat"
          aria-describedby="chat-instructions"
        >
          <div className={styles.chatHeader}>
            <div className={styles.chatHeaderContent}>
              <div className={styles.chatIcon} aria-hidden="true">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H16L14.5 20.5C14.2674 21.0008 13.8335 21.3857 13.3033 21.5721C12.7731 21.7585 12.1871 21.7315 11.6777 21.499C11.1684 21.2665 10.7757 20.8517 10.587 20.342C10.3983 19.8323 10.429 19.2677 10.671 18.758L11.5 17H7C6.46957 17 5.96086 16.7893 5.58579 16.4142C5.21071 16.0391 5 15.5304 5 15V5C5 4.46957 5.21071 3.96086 5.58579 3.58579C5.96086 3.21071 6.46957 3 7 3H17C17.5304 3 18.0391 3.21071 18.4142 3.58579C18.7893 3.96086 19 4.46957 19 5V15Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M9 7H15" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                  <path d="M9 11H15" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                  <path d="M9 15H13" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </div>
              <h3 id="chat-header">Physical AI Assistant</h3>
            </div>
            <button
              className={styles.closeButton}
              onClick={toggleChat}
              aria-label="Close chat assistant"
              title="Close chat (Esc)"
            >
              ×
            </button>
          </div>

          <div
            className={styles.chatMessages}
            aria-live="polite"
            aria-relevant="additions"
            role="log"
          >
            <p id="chat-instructions" className={styles.visuallyHidden}>
              Chat messages between user and assistant. New messages appear at the bottom.
            </p>
            {messages.map((message) => (
              <div
                key={message.id}
                className={clsx(
                  styles.message,
                  styles[`${message.sender}Message`]
                )}
                role="listitem"
                aria-label={`${message.sender === 'user' ? 'You said' : 'Assistant said'}: ${message.text}`}
              >
                <div className={styles.messageText}>{message.text}</div>
              </div>
            ))}
            {isLoading && (
              <div
                className={clsx(styles.message, styles.botMessage)}
                role="status"
                aria-label="Assistant is typing"
              >
                <div className={styles.messageText}>
                  <span className={styles.typingIndicator} aria-hidden="true">
                    <span></span>
                    <span></span>
                    <span></span>
                  </span>
                  <span className={styles.visuallyHidden}>Assistant is typing...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} aria-hidden="true" />
          </div>

          {messages.length <= 1 && (
            <div className={styles.quickQuestions} role="group" aria-label="Suggested questions">
              <p className={styles.visuallyHidden}>Suggested questions to ask the assistant:</p>
              {quickQuestions.map((question, index) => (
                <button
                  key={index}
                  className={styles.quickQuestionButton}
                  onClick={() => handleQuickQuestion(question)}
                  aria-label={`Ask: ${question}`}
                >
                  {question}
                </button>
              ))}
            </div>
          )}

          <form onSubmit={handleSendMessage} className={styles.chatInputForm} role="form" aria-label="Message input form">
            <label htmlFor="chat-input" className={styles.visuallyHidden}>Type your message</label>
            <input
              id="chat-input"
              type="text"
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask about Physical AI, ROS 2, Humanoid Robots... (Press Enter to send, Esc to close)"
              className={styles.chatInput}
              disabled={isLoading}
              aria-describedby="chat-instructions"
              autoComplete="off"
            />
            <button
              type="submit"
              className={styles.sendButton}
              disabled={!inputMessage.trim() || isLoading}
              aria-label="Send message"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </form>
        </div>
      )}
    </>
  );
};

export default ChatbotWidget;