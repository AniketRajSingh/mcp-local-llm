"""Document parser for RAG system.

This module provides functions to parse documents from directories and prepare them
for ingestion into the RAG pipeline. It supports text files and can be extended for
PDFs, DOCX, etc.

Functions:
- doc_parser(*dirs): Parse documents from given directories and save to data/raw/.
- parse_file(filepath): Parse a single file and return its content.
"""

import os
import glob
from typing import List, Dict, Any


def parse_file(filepath: str) -> str:
    """Parse a single file and return its text content.

    Currently supports:
    - .txt files: read as text
    - .md files: read as text
    - Other extensions: attempt to read as text, skip on error

    Args:
        filepath: Path to the file.

    Returns:
        str: The text content of the file.
    """
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in ['.txt', '.md']:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Try to read as text anyway
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"Warning: Could not parse {filepath}: {e}")
        return ""


def doc_parser(*dirs: str, output_dir: str = "data/raw") -> List[Dict[str, Any]]:
    """Parse documents from given directories and save them to output_dir.

    Args:
        *dirs: Variable number of directory paths to scan for documents.
        output_dir: Directory to save parsed documents (default: "data/raw").

    Returns:
        List[Dict]: List of document metadata with keys: 'filename', 'content', 'filepath'.
    """
    documents = []
    os.makedirs(output_dir, exist_ok=True)

    for dir_path in dirs:
        if not os.path.isdir(dir_path):
            print(f"Warning: {dir_path} is not a directory, skipping.")
            continue

        # Find all files (recursive)
        file_patterns = ['**/*.txt', '**/*.md']  # Add more patterns as needed
        files = []
        for pattern in file_patterns:
            files.extend(glob.glob(os.path.join(dir_path, pattern), recursive=True))

        for filepath in files:
            content = parse_file(filepath)
            if content.strip():  # Only save non-empty documents
                filename = os.path.basename(filepath)
                output_path = os.path.join(output_dir, filename)

                # Save to output_dir
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                documents.append({
                    'filename': filename,
                    'content': content,
                    'filepath': filepath,
                    'output_path': output_path
                })

    print(f"Parsed {len(documents)} documents from {len(dirs)} directories.")
    return documents


if __name__ == "__main__":
    # Example usage
    docs = doc_parser("path/to/docs1", "path/to/docs2")
    print(f"Parsed {len(docs)} documents.")