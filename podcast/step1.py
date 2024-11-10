"""
Step 1: Pre-process PDF: Use Llama-3.2-1B-Instruct to pre-process the PDF and save it in a .txt file.


"""
import PyPDF2
from typing import Optional
import os
import torch

import warnings

from podcast import  backend

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



def main(pdf_path: str):
    assert os.path.isfile(pdfpath), f'fnf: {pdfpath=}'
    # Extract metadata first
    print("Extracting metadata...")
    metadata = backend.get_pdf_metadata(pdf_path)
    if metadata:
        print("\nPDF Metadata:")
        print(f"Number of pages: {metadata['num_pages']}")
        print("Document info:")
        for key, value in metadata['metadata'].items():
            print(f"{key}: {value}")

    # Extract text
    print("\nExtracting text...")
    extracted_text = backend.extract_text_from_pdf(pdf_path)

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
