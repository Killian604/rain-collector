"""
Step 1: Pre-process PDF: Use Llama-3.2-1B-Instruct to pre-process the PDF and save it in a .txt file.


"""
from tqdm import tqdm
import os
import PyPDF2
from typing import Optional
import os
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm.notebook import tqdm
import warnings
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

SYS_PROMPT = """
You are a world class text pre-processor, here is the raw data from a PDF, please parse and return it in a way that is crispy and usable to send to a podcast writer.

The raw data is messed up with new lines, Latex math and you will see fluff that we can remove completely. Basically take away any details that you think might be useless in a podcast author's transcript.

Remember, the podcast could be on any topic whatsoever so the issues listed above are not exhaustive

Please be smart with what you remove and be creative ok?

Remember DO NOT START SUMMARIZING THIS, YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED

Be very smart and aggressive with removing details, you will get a running portion of the text and keep returning the processed text.

PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPATILISATION ETC LIKES

ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions ok?
Here is the text:
"""

warnings.filterwarnings('ignore')

def validate_pdf(file_path: str) -> bool:
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return False
    if not file_path.lower().endswith('.pdf'):
        print("Error: File is not a PDF")
        return False
    return True

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


def main(pdf_path: str):
    assert os.path.isfile(pdfpath), f'fnf: {pdfpath=}'
    # Extract metadata first
    print("Extracting metadata...")
    metadata = get_pdf_metadata(pdf_path)
    if metadata:
        print("\nPDF Metadata:")
        print(f"Number of pages: {metadata['num_pages']}")
        print("Document info:")
        for key, value in metadata['metadata'].items():
            print(f"{key}: {value}")

    # Extract text
    print("\nExtracting text...")
    extracted_text = extract_text_from_pdf(pdf_path)

    # Display first 500 characters of extracted text as preview
    if extracted_text:
        print("\nPreview of extracted text (first 500 characters):")
        print("-" * 50)
        print(extracted_text[:500])
        print("-" * 50)
        print(f"\nTotal characters extracted: {len(extracted_text)}")

    # Optional: Save the extracted text to a file
    if extracted_text:
        output_file = 'extracted_text.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"\nExtracted text has been saved to {output_file}")

    pass

if __name__ == '__main__':
    pdfpath = '/home/killfm/Downloads/Mathematics_of_finance.pdf'
    main(
        pdfpath,
    )
