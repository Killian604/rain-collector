"""

"""
import PyPDF2
from typing import Optional
import os
import torch

import warnings

import os



def create_word_bounded_chunks(text, target_chunk_size):
    """
    Split text into chunks at word boundaries close to the target chunk size.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for the space
        if current_length + word_length > target_chunk_size and current_chunk:
            # Join the current chunk and add it to chunks
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks



def extract_text_from_pdf(file_path: str, max_chars: int = 10_000_000) -> Optional[str]:
    if not validate_pdf(file_path):
        return None
    try:
        with open(file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)

            # Get total number of pages
            num_pages = len(pdf_reader.pages)
            print(f"Processing PDF with {num_pages} pages...")

            extracted_text = []
            total_chars = 0

            # Iterate through all pages
            for page_num in range(num_pages):
                # Extract text from page
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                # Check if adding this page's text would exceed the limit
                if total_chars + len(text) > max_chars:
                    # Only add text up to the limit
                    remaining_chars = max_chars - total_chars
                    extracted_text.append(text[:remaining_chars])
                    print(f"Reached {max_chars} character limit at page {page_num + 1}")
                    break

                extracted_text.append(text)
                total_chars += len(text)
                print(f"Processed page {page_num + 1}/{num_pages}")

            final_text = '\n'.join(extracted_text)
            print(f"\nExtraction complete! Total characters: {len(final_text)}")
            return final_text

    except PyPDF2.PdfReadError:
        print("Error: Invalid or corrupted PDF file")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None


# Get PDF metadata
def get_pdf_metadata(file_path: str) -> Optional[dict]:
    if not validate_pdf(file_path):
        return None
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = {
                'num_pages': len(pdf_reader.pages),
                'metadata': pdf_reader.metadata
            }
            return metadata
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return None


def validate_pdf(file_path: str) -> bool:
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return False
    if not file_path.lower().endswith('.pdf'):
        print("Error: File is not a PDF")
        return False
    return True
