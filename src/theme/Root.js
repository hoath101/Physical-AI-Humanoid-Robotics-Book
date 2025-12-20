import React from 'react';
import ChatbotWidget from '../components/ChatbotWidget';

// Root wrapper component that wraps the entire Docusaurus application
export default function Root({ children }) {
  return (
    <>
      {children}
      <ChatbotWidget />
    </>
  );
}
