from typing import List, Dict, Any
from api.models.request import Citation
import re

def extract_citation_info(metadata: Dict[str, Any]) -> Citation:
    """
    Extract citation information from document metadata.
    """
    return Citation(
        chapter=metadata.get('chapter', 'Unknown'),
        page=metadata.get('page'),
        section=metadata.get('section'),
        paragraph_id=metadata.get('paragraph_id'),
        text_snippet=metadata.get('text', '')[:200] + "..." if len(metadata.get('text', '')) > 200 else metadata.get('text', '')
    )

def format_citations(citations_data: List[Dict[str, Any]]) -> List[Citation]:
    """
    Format raw citation data into Citation objects.
    """
    citations = []
    for citation_data in citations_data:
        citation = extract_citation_info(citation_data)
        citations.append(citation)
    return citations

def generate_citation_text(citation: Citation) -> str:
    """
    Generate human-readable citation text from a Citation object.
    """
    parts = []

    if citation.chapter:
        parts.append(f"Chapter: {citation.chapter}")

    if citation.section:
        parts.append(f"Section: {citation.section}")

    if citation.page is not None:
        parts.append(f"Page: {citation.page}")

    if citation.paragraph_id:
        parts.append(f"Paragraph: {citation.paragraph_id}")

    return ", ".join(parts)

def find_context_around_citation(text: str, search_term: str, context_length: int = 100) -> str:
    """
    Find context around a specific term in the text.
    """
    # Find the position of the search term
    pos = text.lower().find(search_term.lower())
    if pos == -1:
        return ""

    # Calculate start and end positions
    start = max(0, pos - context_length)
    end = min(len(text), pos + len(search_term) + context_length)

    # Extract the context
    context = text[start:end]

    # Add ellipses if we truncated
    if start > 0:
        context = "..." + context
    if end < len(text):
        context = context + "..."

    return context

def extract_page_numbers(text: str) -> List[int]:
    """
    Extract page numbers from text using regex patterns.
    """
    # Common patterns for page numbers
    patterns = [
        r'page\s+(\d+)',           # "page 123"
        r'p\.\s*(\d+)',            # "p. 123" or "p 123"
        r'\b(\d+)\s*-\s*(\d+)\b',  # "123-125" (page ranges)
    ]

    page_numbers = []

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                # Handle page ranges like "123-125"
                start, end = int(match[0]), int(match[1])
                page_numbers.extend(range(start, end + 1))
            else:
                page_numbers.append(int(match))

    return list(set(page_numbers))  # Remove duplicates

def extract_section_headers(text: str) -> List[str]:
    """
    Extract section headers from text using common patterns.
    """
    # Patterns for section headers
    patterns = [
        r'^\s*chapter\s+\d+[:\-\s]+([^\n\r]+)',  # Chapter titles
        r'^\s*section\s+\d+[:\-\s]+([^\n\r]+)',  # Section titles
        r'^\s*part\s+\d+[:\-\s]+([^\n\r]+)',    # Part titles
        r'^\s*#\s+([^\n\r]+)',                   # Markdown headers
        r'^\s*##\s+([^\n\r]+)',                  # Markdown subheaders
        r'^\s*###\s+([^\n\r]+)',                 # Markdown subsubheaders
    ]

    headers = []

    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
        headers.extend(matches)

    return list(set(headers))  # Remove duplicates