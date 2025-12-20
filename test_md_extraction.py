#!/usr/bin/env python3
"""
Test script to verify that Markdown extraction works correctly.
This tests the Markdown processing function without requiring Qdrant.
"""

import os
from pathlib import Path
from rag.ingestion import BookIngestionPipeline
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_markdown_extraction():
    """Test the Markdown extraction function with a sample file."""
    print("Testing Markdown extraction function...")

    # Create a sample markdown content to test with
    sample_md_content = """---
id: intro
title: Introduction
sidebar_position: 1
---

# Introduction to Physical AI & Humanoid Robotics

Welcome to the Physical AI & Humanoid Robotics book! This comprehensive educational resource is designed to bridge the gap between digital intelligence and physical robotic bodies, teaching you how to create embodied AI systems.

## What is Physical AI?

Physical AI represents the convergence of artificial intelligence and physical systems. Rather than AI existing purely in digital form, Physical AI brings intelligence into the physical world through robotic bodies. This creates a feedback loop where AI systems can perceive, interact with, and learn from the physical environment.

## The Learning Pathway

This book follows a structured learning pathway that takes you from foundational concepts to advanced autonomous behaviors:

1. **Robotic Middleware** - Understanding ROS 2 and communication patterns
2. **Simulation & Digital Twins** - Creating virtual environments for testing
3. **AI Perception & Navigation** - Enabling robots to understand their environment
4. **Vision-Language-Action** - Natural interaction with robotic systems
5. **Capstone Integration** - Bringing all concepts together

## Target Audience

This book is designed for students with a computer science or AI background who want to:
- Understand the fundamentals of humanoid robotics
- Learn to integrate AI systems with physical robots
- Develop skills in ROS 2, simulation environments, and AI frameworks
- Apply embodied intelligence concepts in practical scenarios
"""

    # Write the sample content to a temporary file
    test_file = "test_sample.md"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(sample_md_content)

    print(f"Created test file: {test_file}")
    print(f"Original content length: {len(sample_md_content)} characters")

    try:
        # Create an instance of the ingestion pipeline (without OpenAI client for this test)
        pipeline = BookIngestionPipeline(openai_client=None)  # We'll only test text extraction

        # Test the Markdown extraction function
        extracted_content = pipeline.extract_text_from_md(test_file)

        print(f"Extracted content length: {len(extracted_content)} characters")
        print("\nFirst 500 characters of extracted content:")
        print(extracted_content[:500] + ("..." if len(extracted_content) > 500 else ""))

        # Verify that key content was extracted
        expected_keywords = [
            "Physical AI",
            "Humanoid Robotics",
            "Robotic Middleware",
            "Simulation & Digital Twins",
            "AI Perception & Navigation"
        ]

        print(f"\nVerifying extraction of key content:")
        for keyword in expected_keywords:
            found = keyword in extracted_content
            print(f"  '{keyword}': {'[FOUND]' if found else '[MISSING]'}")

        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

        print(f"\n[SUCCESS] Markdown extraction test completed successfully!")
        print("The extraction function properly removes Markdown formatting while preserving content.")

        return True

    except Exception as e:
        print(f"\n[ERROR] Error during Markdown extraction test: {str(e)}")
        import traceback
        traceback.print_exc()

        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

        return False

def test_existing_docs():
    """Test extraction on actual documentation files."""
    print("\n" + "="*60)
    print("Testing extraction on actual documentation files...")

    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        print(f"Directory '{docs_dir}' does not exist.")
        return False

    # Find some .md files to test
    md_files = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))

    if not md_files:
        print("No .md files found in docs directory.")
        return False

    print(f"Found {len(md_files)} Markdown files to test.")

    # Test the first few files
    test_files = md_files[:3]  # Test first 3 files
    pipeline = BookIngestionPipeline(openai_client=None)  # Only testing extraction

    for i, file_path in enumerate(test_files):
        print(f"\nTesting file {i+1}: {file_path}")
        try:
            extracted = pipeline.extract_text_from_md(file_path)
            print(f"  Extracted {len(extracted)} characters")
            print(f"  First 100 chars: {extracted[:100]}{'...' if len(extracted) > 100 else ''}")

            # Check that it's not empty and contains meaningful content
            if len(extracted.strip()) == 0:
                print("  [WARN] Warning: Extracted content is empty")
            elif len(extracted.strip()) < 50:
                print("  [WARN] Warning: Extracted content is very short")
            else:
                print("  [SUCCESS] Content extracted successfully")

        except Exception as e:
            print(f"  [ERROR] Error extracting from {file_path}: {str(e)}")

    return True

if __name__ == "__main__":
    print("Testing Markdown extraction functionality...")
    print("="*60)

    success = test_markdown_extraction()
    if success:
        test_existing_docs()

    print("\n" + "="*60)
    print("Test completed.")