import React from 'react';
import ChatbotWidget from './ChatbotWidget';

// Root component that wraps the entire application
const Root = ({ children }) => {
  console.log('Root component rendered - ChatbotWidget should be loaded');
  return (
    <>
      {children}
      <ChatbotWidget />
    </>
  );
};

export default Root;