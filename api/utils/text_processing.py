import re
from typing import List, Tuple
import tiktoken

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks of specified size.
    """
    if not text:
        return []

    # Use tiktoken to count tokens more accurately
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # First, split by paragraphs to maintain semantic boundaries
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # Check if adding this paragraph would exceed chunk size
        test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

        # Count tokens in the test chunk
        token_count = len(encoding.encode(test_chunk))

        if token_count <= chunk_size:
            # If it fits, add to current chunk
            current_chunk = test_chunk
        else:
            # If current chunk is not empty, save it and start a new one
            if current_chunk:
                chunks.append(current_chunk)

            # If the paragraph itself is too large, split it into smaller pieces
            if len(encoding.encode(paragraph)) > chunk_size:
                sub_chunks = split_large_paragraph(paragraph, chunk_size, encoding)
                chunks.extend(sub_chunks[:-1])  # Add all but the last sub-chunk
                current_chunk = sub_chunks[-1] if sub_chunks else ""  # Start new chunk with the remainder
            else:
                current_chunk = paragraph

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk)

    # Apply overlap if specified
    if overlap > 0 and len(chunks) > 1:
        chunks = apply_overlap(chunks, overlap, encoding, chunk_size)

    return chunks

def split_large_paragraph(paragraph: str, chunk_size: int, encoding) -> List[str]:
    """
    Split a large paragraph into smaller chunks.
    """
    sentences = re.split(r'[.!?]+\s+', paragraph)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
        token_count = len(encoding.encode(test_chunk))

        if token_count <= chunk_size:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def apply_overlap(chunks: List[str], overlap: int, encoding, max_chunk_size: int) -> List[str]:
    """
    Apply overlap between consecutive chunks.
    """
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i-1]
        curr_chunk = chunks[i]

        # Get the end portion of the previous chunk for overlap
        prev_tokens = encoding.encode(prev_chunk)
        overlap_tokens = prev_tokens[-overlap:]

        # Decode the overlap portion
        overlap_text = encoding.decode(overlap_tokens)

        # Combine overlap with current chunk
        new_chunk = overlap_text + " " + curr_chunk

        # Ensure the new chunk doesn't exceed the maximum size
        if len(encoding.encode(new_chunk)) <= max_chunk_size:
            result.append(new_chunk)
        else:
            # If it exceeds, keep the original chunk
            result.append(curr_chunk)

    return result

def extract_metadata_from_text(text: str) -> dict:
    """
    Extract basic metadata from text such as chapter titles, section headers, etc.
    """
    metadata = {}

    # Look for potential chapter titles (lines that might be chapter headers)
    lines = text.split('\n')

    # Simple heuristic to identify potential headers
    potential_headers = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line and len(line) < 100:  # Headers are typically short
            # Check if line looks like a header (title case, all caps, or has chapter/section markers)
            if (line.isupper() or
                line.istitle() or
                re.match(r'^(Chapter|Section|Part)\s+\d+[:\-\s]', line, re.IGNORECASE) or
                re.match(r'^#\s+', line) or  # Markdown header
                line.endswith(':')):
                potential_headers.append((i, line))

    metadata['potential_headers'] = potential_headers
    return metadata

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing line breaks.
    """
    # Replace multiple consecutive line breaks with a single one
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text