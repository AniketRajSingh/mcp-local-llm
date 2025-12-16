
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
import PyPDF2
from PIL import Image
import pytesseract


def parse_pdf(filepath: str) -> str:
    """Parse a PDF file and return its text content."""
    print(f"DEBUG: Attempting to parse PDF: {filepath}")
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() or ""
            print(f"DEBUG: PDF content length: {len(text)}")
            return text
    except Exception as e:
        print(f"Warning: Could not parse PDF {filepath}: {e}")
        return ""

def parse_image(filepath: str) -> str:
    """Perform OCR on an image file and return its text content."""
    print(f"DEBUG: Attempting to parse image: {filepath}")
    try:
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)
        print(f"DEBUG: Image OCR content length: {len(text)}")
        return text
    except Exception as e:
        print(f"Warning: Could not perform OCR on image {filepath}: {e}")
        return ""

def parse_file(filepath: str) -> str:
    """Parse a single file and return its text content.

    Currently supports:
    - .txt files: read as text
    - .md files: read as text
    - .pdf files: extract text using PyPDF2
    - .png, .jpg, .jpeg files: perform OCR using pytesseract
    - .py files: read as plain text
    - Other extensions: attempt to read as text, skip on error

    Args:
        filepath: Path to the file.

    Returns:
        str: The text content of the file.
    """
    ext = os.path.splitext(filepath)[1].lower()
    print(f"DEBUG: parse_file called for {filepath} with extension {ext}")
    try:
        if ext in ['.txt', '.md', '.py']:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"DEBUG: Text/Code content length: {len(content)}")
                return content
        elif ext == '.pdf':
            return parse_pdf(filepath)
        elif ext in ['.png', '.jpg', '.jpeg']:
            return parse_image(filepath)
        else:
            # Try to read as text anyway
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"DEBUG: Generic text content length: {len(content)}")
                return content
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
        print(f"DEBUG: Scanning directory: {dir_path}")

        # Find all files (recursive)
        file_patterns = ['**/*.txt', '**/*.md', '**/*.pdf', '**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.py']
        files = []
        for pattern in file_patterns:
            found_files = glob.glob(os.path.join(dir_path, pattern), recursive=True)
            files.extend(found_files)
            print(f"DEBUG: Pattern '{pattern}' found {len(found_files)} files.")

        print(f"DEBUG: Total files found by glob: {len(files)}")

        for filepath in files:
            print(f"DEBUG: Processing file: {filepath}")
            content = parse_file(filepath)
            if content.strip():  # Only save non-empty documents
                filename = os.path.basename(filepath)
                # Check if the file already exists as a .txt and if it's the source.
                # If a .txt file is processed, we don't want to rename it to .txt.txt
                # Only append .txt for non-text source files like PDF, PNG, PY
                if not filename.endswith(".txt"):
                    output_path = os.path.join(output_dir, filename + ".txt")
                else:
                    output_path = os.path.join(output_dir, filename)

                print(f"DEBUG: Saving content from {filepath} to {output_path} (length: {len(content.strip())})")
                # Save to output_dir
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                documents.append({
                    'filename': filename,
                    'content': content,
                    'filepath': filepath,
                    'output_path': output_path
                })
            else:
                print(f"DEBUG: Skipping empty content for file: {filepath}")

    print(f"Parsed {len(documents)} documents from {len(dirs)} directories.")
    return documents


if __name__ == "__main__":
    # Example usage
    # docs = doc_parser("path/to/docs1", "path/to/docs2")
    # print(f"Parsed {len(docs)} documents.")
    pass # To avoid running example usage when main script is executed
