import React from 'react';
import styled from 'styled-components';
import { Message } from '../types/chat';

interface MessageListProps {
  messages: Message[];
}

const MessageListContainer = styled.div`
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  padding: 1rem;
  height: 100%;
  background-color: #f9f9f9;
`;

const MessageBubble = styled.div<{ isUser: boolean }>`
  display: flex;
  margin-bottom: 1rem;
  align-items: flex-start;

  justify-content: ${(props) => (props.isUser ? 'flex-end' : 'flex-start')};
`;

const MessageContent = styled.div<{ isUser: boolean }>`
  max-width: 80%;
  padding: 0.75rem 1rem;
  border-radius: 18px;
  background-color: ${(props) => (props.isUser ? '#007bff' : '#e9ecef')};
  color: ${(props) => (props.isUser ? 'white' : 'black')};
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
  word-wrap: break-word;
  white-space: pre-wrap;
`;

const CitationList = styled.div`
  margin-top: 0.5rem;
  padding-top: 0.5rem;
  border-top: 1px solid #ddd;
`;

const CitationItem = styled.div`
  font-size: 0.8rem;
  color: #6c757d;
  margin-top: 0.25rem;

  &:first-child {
    margin-top: 0;
  }
`;

const MessageList: React.FC<MessageListProps> = ({ messages }) => {
  return (
    <MessageListContainer>
      {messages.map((message) => (
        <MessageBubble key={message.id} isUser={message.role === 'user'}>
          <MessageContent isUser={message.role === 'user'}>
            {message.content}
            {message.citations && message.citations.length > 0 && (
              <CitationList>
                <strong>Citations:</strong>
                {message.citations.map((citation, index) => (
                  <CitationItem key={index}>
                    {citation.chapter && `Chapter: ${citation.chapter}`}
                    {citation.section && `, Section: ${citation.section}`}
                    {citation.page && `, Page: ${citation.page}`}
                    {citation.text_snippet && ` - "${citation.text_snippet}"`}
                  </CitationItem>
                ))}
              </CitationList>
            )}
          </MessageContent>
        </MessageBubble>
      ))}
    </MessageListContainer>
  );
};

export default MessageList;