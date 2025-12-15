// TypeScript types for the RAG Chatbot

export interface Citation {
  chapter: string;
  page?: number;
  section?: string;
  paragraph_id?: string;
  text_snippet: string;
}

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  citations?: Citation[];
}

export interface ChatSession {
  id: string;
  bookId: string;
  messages: Message[];
  createdAt: Date;
  lastActive: Date;
}

export interface QueryRequest {
  question: string;
  query_mode: 'global' | 'selection_only';
  book_id: string;
  selected_text?: string;
  top_k?: number;
  temperature?: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  query_mode: 'global' | 'selection_only';
  retrieved_context_count: number;
  status?: 'success' | 'no_content_found';
  processing_time?: string;
}

export interface IngestionRequest {
  book_content: string;
  book_metadata: {
    title: string;
    author: string;
    isbn?: string;
    edition?: string;
    publication_date?: string;
  };
  chunk_size?: number;
  overlap_size?: number;
  book_id: string;
}

export interface IngestionResponse {
  status: 'success' | 'error';
  message: string;
  book_id: string;
  chunks_processed: number;
  processing_time: string;
}

export interface ErrorResponse {
  status: 'error';
  message: string;
  error_id?: string;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  dependencies: Record<string, string>;
}

export interface ChatConfig {
  apiUrl: string;
  apiKey: string;
  defaultBookId: string;
  maxTokens: number;
  temperature: number;
}