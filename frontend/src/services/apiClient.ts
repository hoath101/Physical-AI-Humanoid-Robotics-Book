import axios, { AxiosInstance, AxiosResponse } from 'axios';
import {
  QueryRequest,
  QueryResponse,
  IngestionRequest,
  IngestionResponse,
  HealthResponse,
  ErrorResponse
} from '../types/chat';

class ApiClient {
  private client: AxiosInstance;
  private apiKey: string;

  constructor(apiUrl: string, apiKey: string) {
    this.client = axios.create({
      baseURL: apiUrl,
      timeout: 30000, // 30 second timeout
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
    });
    this.apiKey = apiKey;

    // Add request interceptor to include auth header
    this.client.interceptors.request.use(
      (config) => {
        config.headers.Authorization = `Bearer ${this.apiKey}`;
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error);
        return Promise.reject(error);
      }
    );
  }

  // Query endpoints
  async queryGlobal(request: QueryRequest, options?: { headers?: Record<string, string> }): Promise<AxiosResponse<QueryResponse>> {
    const config = options ? { headers: options.headers } : {};
    return this.client.post<QueryResponse>('/api/v1/query', request, config);
  }

  async querySelection(request: QueryRequest, options?: { headers?: Record<string, string> }): Promise<AxiosResponse<QueryResponse>> {
    const config = options ? { headers: options.headers } : {};
    return this.client.post<QueryResponse>('/api/v1/query/selection', request, config);
  }

  async queryFlexible(request: QueryRequest, options?: { headers?: Record<string, string> }): Promise<AxiosResponse<QueryResponse>> {
    const config = options ? { headers: options.headers } : {};
    return this.client.post<QueryResponse>('/api/v1/query/mode', request, config);
  }

  // Ingestion endpoints
  async ingestBook(request: IngestionRequest): Promise<AxiosResponse<IngestionResponse>> {
    return this.client.post<IngestionResponse>('/api/v1/ingest', request);
  }

  async updateBook(request: IngestionRequest): Promise<AxiosResponse<IngestionResponse>> {
    return this.client.put<IngestionResponse>('/api/v1/ingest', request);
  }

  async deleteBook(bookId: string): Promise<AxiosResponse<any>> {
    return this.client.delete(`/api/v1/ingest/${bookId}`);
  }

  // Health check endpoint
  async healthCheck(): Promise<AxiosResponse<HealthResponse>> {
    return this.client.get<HealthResponse>('/api/v1/health');
  }

  // Utility method to check if API is available
  async isHealthy(): Promise<boolean> {
    try {
      const response = await this.healthCheck();
      return response.data.status === 'healthy';
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }
}

export default ApiClient;