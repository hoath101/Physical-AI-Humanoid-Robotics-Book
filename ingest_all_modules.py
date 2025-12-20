"""
Script to ingest all modules at once.
"""
import asyncio
import os
from pathlib import Path
from services.ai_client import AIClient
from rag.ingestion import BookIngestionPipeline

async def main():
    """Run the ingestion pipeline for all modules"""

    print("="*60)
    print("Starting Document Ingestion (All Modules)")
    print("="*60)

    # Initialize AI client
    print("\n[INFO] Initializing AI client...")
    ai_client = AIClient()

    # Create pipeline
    print("\n[INFO] Creating ingestion pipeline...")
    pipeline = BookIngestionPipeline(ai_client)
    pipeline.max_chunk_size = 500
    pipeline.overlap = 50
    print(f"[INFO] Using chunk size: {pipeline.max_chunk_size}")

    # Initialize collection (will reset if needed)
    print("[INFO] Initializing Qdrant collection...")
    await pipeline.initialize_collection()

    # Define all modules
    modules = [
        "module-1-ros2",
        "module-2-digital-twin",
        "module-3-ai-perception",
        "module-4-vla"
    ]

    skip_files = ["intro.md"]
    total_chunks_all = 0

    for module_idx, module in enumerate(modules, 1):
        print(f"\n{'='*60}")
        print(f"Processing Module {module_idx}/{len(modules)}: {module}")
        print(f"{'='*60}")

        docs_dir = os.path.join(os.getcwd(), "docs", module)

        if not os.path.exists(docs_dir):
            print(f"[WARNING] Directory not found: {docs_dir}")
            continue

        # Get all markdown files
        all_md_files = list(Path(docs_dir).glob("*.md"))
        md_files = [f for f in all_md_files if f.name not in skip_files]

        print(f"[INFO] Found {len(md_files)} markdown files")

        # Process files
        total_chunks_module = 0
        for idx, file_path in enumerate(md_files, 1):
            try:
                print(f"\n[{idx}/{len(md_files)}] Processing: {file_path.name}")

                # Process file
                chunks = await pipeline.process_file(str(file_path))

                if chunks:
                    print(f"  -> Generated {len(chunks)} chunks")
                    print(f"  -> Storing in Qdrant...")
                    await pipeline.store_chunks_in_qdrant(chunks)
                    total_chunks_module += len(chunks)
                    total_chunks_all += len(chunks)
                    print(f"  -> Success! (Module total: {total_chunks_module} chunks)")
                else:
                    print(f"  -> No chunks generated")

            except Exception as e:
                print(f"  -> ERROR: {e}")
                print(f"  -> Continuing with next file...")
                import traceback
                traceback.print_exc()

        print(f"\n[Module {module_idx} Complete] {total_chunks_module} chunks ingested")

    print("\n" + "="*60)
    print(f"[SUCCESS] All modules ingested!")
    print(f"  Total modules: {len(modules)}")
    print(f"  Total chunks: {total_chunks_all}")
    print("="*60)
    print("\n[INFO] Your chatbot is now ready to answer questions!")

if __name__ == "__main__":
    asyncio.run(main())
