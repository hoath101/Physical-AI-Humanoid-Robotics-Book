import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import useChat from '../hooks/useChat';
import MessageList from './MessageList';
import InputArea from './InputArea';

const WidgetContainer = styled.div<{ isOpen: boolean }>`
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: ${(props) => (props.isOpen ? '400px' : '0')};
  height: ${(props) => (props.isOpen ? '600px' : '0')};
  border: 1px solid #ddd;
  border-radius: 10px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  background-color: white;
  overflow: hidden;
  transition: all 0.3s ease;
  z-index: 1000;
  display: flex;
  flex-direction: column;
`;

const WidgetHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background-color: #007bff;
  color: white;
  cursor: pointer;
`;

const WidgetTitle = styled.h3`
  margin: 0;
  font-size: 1.1rem;
`;

const ToggleButton = styled.button`
  background: none;
  border: none;
  color: white;
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    background-color: rgba(255, 255, 255, 0.2);
    border-radius: 50%;
  }
`;

const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
`;

const FloatingActionButton = styled.button<{ isOpen: boolean }>`
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background-color: #007bff;
  color: white;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 1001;
  display: ${(props) => (props.isOpen ? 'none' : 'flex')};
  align-items: center;
  justify-content: center;

  &:hover {
    background-color: #0056b3;
  }
`;

const LoadingIndicator = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 1rem;
  color: #6c757d;
`;

const ErrorBanner = styled.div`
  background-color: #f8d7da;
  color: #721c24;
  padding: 0.5rem 1rem;
  text-align: center;
  font-size: 0.9rem;
`;

const ChatbotWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedText, setSelectedText] = useState<string | undefined>(undefined);

  // In a real implementation, you would get these from environment or props
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const apiKey = process.env.REACT_APP_API_KEY || 'your_backend_api_key';
  const defaultBookId = process.env.REACT_APP_DEFAULT_BOOK_ID || 'default-book';

  const {
    messages,
    isLoading,
    error,
    sendMessage,
    initializeSession
  } = useChat({
    apiUrl,
    apiKey,
    defaultBookId
  });

  // Function to handle text selection on the page
  const handleTextSelection = () => {
    const selectedText = window.getSelection()?.toString().trim();
    if (selectedText) {
      setSelectedText(selectedText);
    } else {
      setSelectedText(undefined);
    }
  };

  // Add event listener for text selection
  useEffect(() => {
    document.addEventListener('mouseup', handleTextSelection);
    return () => {
      document.removeEventListener('mouseup', handleTextSelection);
    };
  }, []);

  const toggleWidget = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      // Initialize session when opening if not already done
      if (!messages.length) {
        initializeSession();
      }
    }
  };

  const handleSendMessage = (message: string, selectedText?: string) => {
    sendMessage(message, selectedText);
  };

  return (
    <>
      <FloatingActionButton isOpen={isOpen} onClick={toggleWidget}>
        💬
      </FloatingActionButton>

      <WidgetContainer isOpen={isOpen}>
        <WidgetHeader onClick={toggleWidget}>
          <WidgetTitle>Book Assistant</WidgetTitle>
          <ToggleButton onClick={toggleWidget}>
            ×
          </ToggleButton>
        </WidgetHeader>

        <ChatContainer>
          {error && <ErrorBanner>Error: {error}</ErrorBanner>}

          <MessageList messages={messages} />

          <InputArea
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            selectedText={selectedText}
          />

          {isLoading && (
            <LoadingIndicator>
              Thinking...
            </LoadingIndicator>
          )}
        </ChatContainer>
      </WidgetContainer>
    </>
  );
};

export default ChatbotWidget;