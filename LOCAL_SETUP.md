# Local Setup and Testing Instructions

This guide will help you set up and test the Physical AI & Humanoid Robotics Book RAG Chatbot locally.

## Prerequisites

Before starting, ensure you have:

1. **Python 3.8+** installed
2. **Node.js** (for Docusaurus development, if needed)
3. **Access to OpenAI API** (with sufficient quota)
4. **Qdrant Cloud account** (free tier available)
5. **Neon Postgres account** (free tier available)

## Step 1: Clone and Set Up the Repository

```bash
# If you haven't already, clone the repository
git clone <your-repo-url>
cd hackathon-book

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
# .env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration
QDRANT_URL=https://your-cluster-url.qdrant.tech  # For cloud version
# OR for local development:
# QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=book_chunks

# Neon Postgres Configuration
NEON_DB_URL=postgresql://username:password@ep-xxx.us-east-1.aws.neon.tech/dbname

# Ingestion Configuration
BOOK_DIRECTORY=docs
MAX_CHUNK_SIZE=1000
OVERLAP=100
BATCH_SIZE=50
MAX_WORKERS=5
```

## Step 3: Prepare Book Content

Ensure your book content is in the `docs/` directory in Markdown format (`.md` or `.mdx` files). The ingestion pipeline will automatically detect and process these files.

Example structure:
```
docs/
├── intro.md
├── module-1-ros2/
│   ├── index.md
│   ├── concepts.md
│   └── examples.md
├── module-2-digital-twin/
│   ├── index.md
│   └── simulation.md
└── ...
```

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a requirements.txt file, install the required packages:

```bash
pip install fastapi uvicorn openai qdrant-client asyncpg python-dotenv pydantic httpx aiofiles
```

## Step 5: Run the System

### Option A: Development Mode (Recommended)

```bash
# Start the FastAPI server
python main.py
```

The server will start on `http://localhost:8000` with auto-reload enabled.

### Option B: Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Step 6: Ingest Book Content

Once the server is running, you can trigger the ingestion pipeline:

### Using the API directly:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "book_directory": "docs",
    "max_chunk_size": 1000,
    "overlap": 100
  }'
```

### Using the Python client:

```python
import asyncio
from openai import OpenAI
from rag.ingestion import run_ingestion_pipeline

# Initialize OpenAI client
openai_client = OpenAI(api_key="your-api-key")

# Run ingestion
asyncio.run(run_ingestion_pipeline(
    book_directory="docs",
    openai_client=openai_client,
    max_chunk_size=1000,
    overlap=100
))
```

## Step 7: Test the Chat Functionality

### Test Full-Book QA Mode:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain ROS 2 node communication",
    "selected_text": null
  }'
```

### Test Selected-Text-Only Mode:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is this about?",
    "selected_text": "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms."
  }'
```

## Step 8: Test with Sample Queries

Try these sample queries to test the system:

### ROS 2 Queries:
- "Explain ROS 2 node communication"
- "What are the key differences between ROS 1 and ROS 2?"
- "How do I create a ROS 2 package?"

### Digital Twin Queries:
- "Describe digital twin applications"
- "How does Gazebo integrate with ROS 2?"
- "What are the benefits of digital twin simulation?"

### AI Perception Queries:
- "Explain computer vision in robotics"
- "How do robots perceive their environment?"
- "What are common sensors used in humanoid robots?"

## Step 9: Embed in Docusaurus Site

To test the frontend widget with your Docusaurus site:

1. Copy `frontend/embed_snippet.js` to your Docusaurus `static/js/` directory
2. Add the script to your Docusaurus config:

```js
// docusaurus.config.js
module.exports = {
  // ... other config
  scripts: [
    {
      src: '/js/embed_snippet.js',
      async: true,
      defer: true,
    },
  ],
  // ... rest of config
};
```

3. Configure the widget:

```js
// In your Docusaurus config or a script tag
window.CHATBOT_CONFIG = {
  apiBaseUrl: 'http://localhost:8000',  // Your backend URL
  widgetTitle: 'Physical AI & Robotics Assistant',
  placeholderText: 'Ask about Physical AI, ROS 2, Humanoid Robotics...',
  themeColor: '#4f46e5'
};
```

## Step 10: Monitor System Status

Check the system status:

```bash
curl http://localhost:8000/ingestion-status
```

Check health:

```bash
curl http://localhost:8000/health
```

Get system info:

```bash
curl http://localhost:8000/info
```

## Troubleshooting

### Common Issues:

1. **Environment Variables Not Set**: Ensure all required environment variables are in your `.env` file
2. **API Key Issues**: Verify your OpenAI, Qdrant, and Neon credentials are correct
3. **Database Connection**: Check that your Neon DB URL is properly formatted
4. **Qdrant Connection**: Verify your Qdrant URL and API key are correct

### Enable Logging:

Add these to your `.env` file for more detailed logs:

```bash
# Enable detailed logging
LOG_LEVEL=INFO
```

### Test Individual Components:

```bash
# Test OpenAI connection
python -c "from openai import OpenAI; client = OpenAI(); print('OpenAI connected successfully')"

# Test Qdrant connection
python -c "from qdrant_client import QdrantClient; client = QdrantClient(url='your_url', api_key='your_key'); print('Qdrant connected successfully')"
```

## Performance Testing

To test with larger datasets:

1. Use a larger book directory
2. Adjust `MAX_CHUNK_SIZE` and `OVERLAP` in your configuration
3. Monitor response times using the `/chat` endpoint
4. Check Qdrant dashboard for vector search performance

## Cleanup

To reset the system:

```bash
# Clear Qdrant collection (use with caution!)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Clear all data",
    "selected_text": "CLEAR_COLLECTION"
  }'
```

Or directly call the RAG retriever's clear method in Python.

## Next Steps

1. Integrate with your production Docusaurus site
2. Set up proper CORS for web integration
3. Configure rate limiting for production
4. Set up monitoring and logging for production use