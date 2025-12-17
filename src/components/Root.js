import React from 'react';
import ChatbotWidget from './ChatbotWidget';

// Root component that wraps the entire application
const Root = ({ children }) => {
  return (
    <>
      {children}
      <ChatbotWidget />
    </>
  );
};

export default Root;