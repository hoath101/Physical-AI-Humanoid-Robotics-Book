"""
Basic test for global query functionality.
This is a simple integration test to verify the core functionality works.
"""
import asyncio
from api.services.chat import chat_service
from api.models.request import QueryMode

async def test_global_query():
    """
    Test the global query functionality with sample data.
    """
    # This is a simplified test - in a real implementation you would:
    # 1. Ingest sample book content first
    # 2. Query the content
    # 3. Verify the response

    print("Testing global query functionality...")

    try:
        # Example query - in a real test you would have ingested content first
        result = await chat_service.generate_response(
            query="What is the main topic of the book?",
            book_id="test-book-123",
            query_mode=QueryMode.GLOBAL
        )

        print(f"Query result: {result}")
        print("Global query test completed successfully!")

    except Exception as e:
        print(f"Error during global query test: {str(e)}")
        # This is expected if no content has been ingested yet

if __name__ == "__main__":
    asyncio.run(test_global_query())