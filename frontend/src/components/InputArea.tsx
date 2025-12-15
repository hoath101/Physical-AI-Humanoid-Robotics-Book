import React, { useState, KeyboardEvent, ChangeEvent } from 'react';
import styled from 'styled-components';

const InputContainer = styled.div`
  display: flex;
  padding: 1rem;
  border-top: 1px solid #ddd;
  background-color: white;
`;

const InputField = styled.textarea`
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #ccc;
  border-radius: 18px;
  resize: none;
  min-height: 50px;
  max-height: 150px;
  font-family: inherit;
  font-size: 1rem;

  &:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
  }
`;

const SendButton = styled.button`
  margin-left: 0.5rem;
  padding: 0.75rem 1.5rem;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 18px;
  cursor: pointer;
  font-size: 1rem;

  &:hover {
    background-color: #0056b3;
  }

  &:disabled {
    background-color: #6c757d;
    cursor: not-allowed;
  }
`;

interface InputAreaProps {
  onSendMessage: (message: string, selectedText?: string) => void;
  isLoading: boolean;
  selectedText?: string;
}

const InputArea: React.FC<InputAreaProps> = ({ onSendMessage, isLoading, selectedText }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSend = () => {
    const message = inputValue.trim();
    if (message && !isLoading) {
      onSendMessage(message, selectedText);
      setInputValue('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Send message on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);
  };

  return (
    <InputContainer>
      <InputField
        value={inputValue}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        placeholder={selectedText ? "Ask about the selected text..." : "Type your question about the book..."}
        disabled={isLoading}
        rows={2}
      />
      <SendButton onClick={handleSend} disabled={isLoading || !inputValue.trim()}>
        {isLoading ? 'Sending...' : 'Send'}
      </SendButton>
    </InputContainer>
  );
};

export default InputArea;