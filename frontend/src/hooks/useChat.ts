import { useState, useCallback, useEffect } from 'react';
import ApiClient from '../services/apiClient';
import {
  Message,
  ChatSession,
  QueryRequest,
  QueryResponse,
  IngestionRequest,
  IngestionResponse,
} from '../types/chat';

interface UseChatOptions {
  apiUrl: string;
  apiKey: string;
  defaultBookId: string;
}

const useChat = ({ apiUrl, apiKey, defaultBookId }: UseChatOptions) => {
  const [client] = useState(() => new ApiClient(apiUrl, apiKey));
  const [session, setSession] = useState<ChatSession | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);

  // Initialize a new session
  const initializeSession = useCallback((bookId: string = defaultBookId) => {
    const newSession: ChatSession = {
      id: `session_${Date.now()}`,
      bookId,
      messages: [],
      createdAt: new Date(),
      lastActive: new Date(),
    };

    setSession(newSession);
    setMessages([]);
    setError(null);
  }, [defaultBookId]);

  // Send a message and get a response
  const sendMessage = useCallback(async (question: string, selectedText?: string) => {
    if (!session) {
      initializeSession();
    }

    setIsLoading(true);
    setError(null);

    try {
      // Add user message to the chat
      const userMessage: Message = {
        id: `msg_${Date.now()}_user`,
        content: question,
        role: 'user',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, userMessage]);

      // Prepare the query request
      const queryRequest: QueryRequest = {
        question,
        query_mode: selectedText ? 'selection_only' : 'global',
        book_id: session?.bookId || defaultBookId,
        selected_text: selectedText,
        top_k: 5,
        temperature: 0.3,
      };

      // Set up headers with session ID if available
      const headers: Record<string, string> = {};
      if (sessionId) {
        headers['X-Session-ID'] = sessionId;
      }

      // Call the appropriate API endpoint based on mode
      let response;
      if (selectedText) {
        response = await client.querySelection(queryRequest, { headers });
      } else {
        response = await client.queryGlobal(queryRequest, { headers });
      }

      const queryResponse: QueryResponse = response.data;

      // Create assistant message
      const assistantMessage: Message = {
        id: `msg_${Date.now()}_assistant`,
        content: queryResponse.answer,
        role: 'assistant',
        timestamp: new Date(),
        citations: queryResponse.citations,
      };

      // Update messages with the assistant's response
      setMessages(prev => [...prev, assistantMessage]);

      // Update session with new messages
      if (session) {
        setSession({
          ...session,
          messages: [...session.messages, userMessage, assistantMessage],
          lastActive: new Date(),
        });
      }

      return queryResponse;
    } catch (err) {
      console.error('Error sending message:', err);
      setError(err instanceof Error ? err.message : 'An error occurred while sending the message');

      // Add error message to chat
      const errorMessage: Message = {
        id: `msg_${Date.now()}_error`,
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [client, session, defaultBookId, initializeSession]);

  // Ingest a book
  const ingestBook = useCallback(async (request: IngestionRequest) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await client.ingestBook(request);
      const result: IngestionResponse = response.data;

      // Initialize a session for the newly ingested book
      initializeSession(request.book_id);

      return result;
    } catch (err) {
      console.error('Error ingesting book:', err);
      setError(err instanceof Error ? err.message : 'An error occurred while ingesting the book');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [client, initializeSession]);

  // Check if API is healthy
  const checkHealth = useCallback(async () => {
    try {
      return await client.isHealthy();
    } catch (err) {
      console.error('Health check failed:', err);
      return false;
    }
  }, [client]);

  // Clear the current session
  const clearSession = useCallback(() => {
    setSession(null);
    setMessages([]);
    setError(null);
  }, []);

  // Effect to initialize a default session
  useEffect(() => {
    if (!session) {
      initializeSession();
    }
  }, [session, initializeSession]);

  return {
    session,
    messages,
    isLoading,
    error,
    initializeSession,
    sendMessage,
    ingestBook,
    checkHealth,
    clearSession,
  };
};

export default useChat;