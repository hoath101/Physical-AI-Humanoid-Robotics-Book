"""
Unified AI Client wrapper that supports both OpenAI and Google Gemini.
This module provides a consistent interface for chat completions and embeddings
regardless of the underlying AI provider.
"""
import os
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

load_dotenv()


class AIClient:
    """
    Unified client for OpenAI and Gemini APIs.
    Automatically selects the provider based on AI_PROVIDER environment variable.
    """

    def __init__(self):
        self.provider = os.getenv("AI_PROVIDER", "openai").lower()
        print(f"[AI] Initializing AI Client with provider: {self.provider.upper()}")

        # Thread pool for running sync operations async
        self.executor = ThreadPoolExecutor(max_workers=5)

        if self.provider == "gemini":
            # Initialize Gemini
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set")
            genai.configure(api_key=api_key)
            self.chat_model = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
            self.embed_model = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
            print(f"[AI] Chat Model: {self.chat_model}")
            print(f"[AI] Embedding Model: {self.embed_model}")
        else:
            # Initialize OpenAI
            from openai import AsyncOpenAI
            import httpx
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.openai_client = AsyncOpenAI(
                api_key=api_key,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                    timeout=60.0,
                )
            )
            self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
            self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
            print(f"[AI] Chat Model: {self.chat_model}")
            print(f"[AI] Embedding Model: {self.embed_model}")

    def _create_gemini_embedding_sync(self, text: str) -> List[float]:
        """Synchronous wrapper for Gemini embedding creation"""
        result = genai.embed_content(
            model=self.embed_model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    async def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for text using the configured AI provider.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if self.provider == "gemini":
            # Run Gemini embedding in thread pool to prevent blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                self._create_gemini_embedding_sync,
                text
            )
            return embedding
        else:
            # OpenAI embeddings are asynchronous
            response = await self.openai_client.embeddings.create(
                model=self.embed_model,
                input=text
            )
            return response.data[0].embedding

    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts in parallel (much faster).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if self.provider == "gemini":
            # Process in parallel using thread pool
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    self.executor,
                    self._create_gemini_embedding_sync,
                    text
                )
                for text in texts
            ]
            embeddings = await asyncio.gather(*tasks)
            return embeddings
        else:
            # OpenAI supports batch embeddings natively
            response = await self.openai_client.embeddings.create(
                model=self.embed_model,
                input=texts
            )
            return [item.embedding for item in response.data]

    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> str:
        """
        Create chat completion using the configured AI provider.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        if self.provider == "gemini":
            model = genai.GenerativeModel(self.chat_model)

            # Convert OpenAI message format to Gemini format
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt_parts.append(f"Instructions: {content}\n\n")
                elif role == "user":
                    prompt_parts.append(content)
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}\n\n")

            prompt = "".join(prompt_parts)

            # Generate response with Gemini
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text
        else:
            # OpenAI chat completion
            response = await self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30
            )
            return response.choices[0].message.content

    @property
    def api_key(self):
        """Get the API key for the current provider (for compatibility checks)"""
        if self.provider == "gemini":
            return os.getenv("GEMINI_API_KEY")
        else:
            return os.getenv("OPENAI_API_KEY")


# Global AI client instance
_ai_client = None


def get_ai_client() -> AIClient:
    """Get or create the global AI client instance"""
    global _ai_client
    if _ai_client is None:
        _ai_client = AIClient()
    return _ai_client


async def create_ai_client() -> AIClient:
    """Create and return a new AI client instance"""
    return AIClient()
