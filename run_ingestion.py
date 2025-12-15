#!/usr/bin/env python3
"""
Script to run the Physical AI & Humanoid Robotics Book ingestion pipeline.
This script processes the book content, chunks it, generates embeddings,
and stores them in Qdrant and Neon Postgres.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from openai import OpenAI
from rag.ingestion import run_ingestion_pipeline, BookIngestionPipeline
from config.ingestion_config import get_config_value, get_all_book_sections
from db.database import db_manager


async def main():
    parser = argparse.ArgumentParser(description="Run the Physical AI & Humanoid Robotics Book ingestion pipeline")
    parser.add_argument(
        "--book-directory",
        type=str,
        default=get_config_value("book_directory", "docs"),
        help="Path to the book directory containing markdown files (default: docs)"
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=get_config_value("max_chunk_size", 1000),
        help="Maximum size of each chunk in characters (default: 1000)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=get_config_value("overlap", 100),
        help="Number of overlapping characters between chunks (default: 100)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actually ingesting data"
    )
    parser.add_argument(
        "--recreate-db",
        action="store_true",
        help="Recreate database tables before ingestion"
    )

    args = parser.parse_args()

    print("🚀 Starting Physical AI & Humanoid Robotics Book ingestion pipeline...")
    print(f"📚 Book directory: {args.book_directory}")
    print(f"📏 Max chunk size: {args.max_chunk_size}")
    print(f"🔄 Overlap: {args.overlap}")
    print(f"🧪 Dry run: {args.dry_run}")
    print("-" * 50)

    # Validate inputs
    if not os.path.exists(args.book_directory):
        print(f"❌ Error: Book directory does not exist: {args.book_directory}")
        sys.exit(1)

    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    # Initialize OpenAI client
    openai_client = OpenAI(api_key=openai_api_key)

    # Check required environment variables
    required_env_vars = ["QDRANT_URL", "QDRANT_API_KEY", "NEON_DB_URL"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        sys.exit(1)

    if args.dry_run:
        print("⚠️  DRY RUN MODE - No data will be ingested")
        print("📋 This would process and chunk the book content, but not store it anywhere")

        # Show what sections would be processed
        print("\n📖 Book sections that would be processed:")
        sections = get_all_book_sections()
        for section_key, section_info in sections.items():
            print(f"  - {section_key}: {section_info['title']}")

        print("\n✅ Dry run completed successfully")
        return

    try:
        # Initialize the ingestion pipeline
        ingestion_pipeline = BookIngestionPipeline(openai_client)

        # Initialize the system components
        await ingestion_pipeline.initialize_system()

        # Run the ingestion pipeline
        await run_ingestion_pipeline(
            book_directory=args.book_directory,
            openai_client=openai_client,
            max_chunk_size=args.max_chunk_size,
            overlap=args.overlap
        )

        print("\n✅ Book ingestion completed successfully!")

        # Print summary
        from rag.retriever import RAGRetriever
        rag_retriever = RAGRetriever(openai_client)
        chunk_count = await rag_retriever.get_chunk_count()
        sections = await rag_retriever.get_all_sections()

        print(f"📊 Summary:")
        print(f"   Total chunks ingested: {chunk_count}")
        print(f"   Sections processed: {len(sections)}")
        print(f"   Unique sections: {', '.join(sections[:5])}{'...' if len(sections) > 5 else ''}")

    except KeyboardInterrupt:
        print("\n⚠️  Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during ingestion: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Close database connection
        try:
            await db_manager.close()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())