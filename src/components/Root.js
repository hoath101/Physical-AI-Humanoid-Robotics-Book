import React from 'react';
import ChatbotWidget from './ChatbotWidget'

// Root component that wraps the entire application
const Root = ({ children }) => {
  return (
    <React.Fragment>
      {children}
      <ChatbotWidget />
    </React.Fragment>
  );
};

export default Root;