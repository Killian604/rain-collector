"""
Step 1: Pre-process PDF: Simplify incoming text.
Use Llama-3.2-1B-Instruct to pre-process the PDF and save it in a .txt file.
"""
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
import PyPDF2
from podcast import backend
from typing import Optional
import os
import re
import torch
import warnings

# DEFAULT_MODEL = os.path.join(os.environ['MYMODELS'], 'Llama-3.2-1B-Instruct')
DEFAULT_MODEL = os.path.join('/home/killfm/projects/text-generation-webui/models', 'Meta-Llama-3.1-8B-Instruct')  # 'Llama-3.2-1B-Instruct'
device = 'cuda'  # "cuda" if torch.cuda.is_available() else "cpu"

SYS_PROMPT = """
You are a world class text pre-processor, here is the raw data from a PDF, please parse and return it in a way that is crispy and usable to send to a podcast writer.

The raw data is messed up with new lines, Latex math and you will see fluff that we can remove completely. Basically take away any details that you think might be useless in a podcast author's transcript.

Remember, the podcast could be on any topic whatsoever so the issues listed above are not exhaustive

Please be smart with what you remove and be creative ok?

Remember DO NOT START SUMMARIZING THIS, YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED

Be very smart and aggressive with removing details, you will get a running portion of the text and keep returning the processed text.

PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALISATION ETC LIKES

ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions ok?
Here is the text:
"""

warnings.filterwarnings('ignore')


def main(pdf_path: str, intermediate_file_path, output_file_path):
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
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"\nExtracted text has been saved to {intermediate_file_path}")

    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, use_safetensors=True)
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Read the file
    CHUNK_SIZE = 2_000  # Adjust chunk size if needed

    with open(intermediate_file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # chunks = backend.create_word_bounded_chunks(text, CHUNK_SIZE, SYS_PROMPT, tokenizer, model, device)
    chunks = backend.create_word_bounded_chunks(text, CHUNK_SIZE)

    # Calculate number of chunks
    num_chunks = (len(text) + CHUNK_SIZE - 1) // CHUNK_SIZE


    processed_text = ''
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for chunk_num, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            # Process chunk and append to complete text
            processed_chunk = backend.process_chunk(chunk, chunk_num, SYS_PROMPT, tokenizer, model, device)
            processed_chunk = re.sub(r'(\n){2,}', '\n', processed_chunk)
            processed_text += processed_chunk + "\n"

            # Write chunk immediately to file
            out_file.write(processed_chunk + "\n")
            out_file.flush()
    num_chunks = len(chunks)

    print(f"\nProcessing complete!")
    print(f"Input file: {intermediate_file_path}")
    print(f"Output file: {output_file}")
    print(f"Total chunks processed: {num_chunks}")

    # Preview the beginning and end of the complete processed text
    print("\nPreview of final processed text:")
    print("\nBEGINNING:")
    print(processed_text[:1000])
    print("\n...\n\nEND:")
    print(processed_text[-1000:])
    pass

if __name__ == '__main__':
    pdfpath = '/home/killfm/Downloads/Mathematics_of_finance.pdf'
    intermediate_file_path = 'extracted_text.txt'
    output_file_path = output_file = f"clean_2{os.path.basename(pdfpath)}"
    main(
        pdfpath,
        intermediate_file_path,
        output_file_path,
    )
