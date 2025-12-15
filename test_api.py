"""
Test script for the Physical AI & Humanoid Robotics Book RAG Chatbot API
This script can be used to verify the API endpoints are working correctly.
"""

import asyncio
import httpx
import json

async def test_chat_endpoint():
    """Test the chat endpoint with sample requests."""

    base_url = "http://localhost:8000"

    # Sample test cases

    # Test 1: Full-book QA mode
    print("Testing Full-book QA mode...")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/chat",
            json={
                "question": "What are the key concepts in ROS 2 for humanoid robotics?",
                "selected_text": None  # This should trigger full-book QA mode
            },
            headers={"Content-Type": "application/json"}
        )

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result.get('answer', '')[:200]}...")
            print(f"Mode used: {result.get('mode_used')}")
        else:
            print(f"Error: {response.text}")

    print("\n" + "="*50 + "\n")

    # Test 2: Selected-text-only mode
    print("Testing Selected-text-only mode...")
    sample_text = """
    ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software.
    It provides a collection of tools, libraries, and conventions that aim to simplify
    the task of creating complex and robust robot behavior across a wide variety of
    robot platforms. Key features include improved support for real-time systems,
    better tools for distributed systems, and enhanced security.
    """

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/chat",
            json={
                "question": "What is ROS 2?",
                "selected_text": sample_text  # This should trigger selected-text-only mode
            },
            headers={"Content-Type": "application/json"}
        )

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result.get('answer', '')[:200]}...")
            print(f"Mode used: {result.get('mode_used')}")
        else:
            print(f"Error: {response.text}")

async def test_other_endpoints():
    """Test other API endpoints."""

    base_url = "http://localhost:8000"

    print("Testing other endpoints...\n")

    # Test health endpoint
    print("Testing /health endpoint...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/health")
        print(f"Health check: Status {response.status_code}, Response: {response.json()}")

    print("\nTesting /info endpoint...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/info")
        print(f"Info: Status {response.status_code}, Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("Starting API tests for Physical AI & Humanoid Robotics Book RAG Chatbot...")

    # Run the tests
    asyncio.run(test_chat_endpoint())
    asyncio.run(test_other_endpoints())

    print("\nTests completed. To run this script, make sure the FastAPI server is running on localhost:8000.")