# ingest/test_ingest_local.py
import os
import time
import glob
from pathlib import Path
from dotenv import load_dotenv
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ingest.docling_loader import CustomDocumentLoader

load_dotenv()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDF_DIR = os.path.join(REPO_ROOT, "pdfs")


def find_pdfs(limit: int = 1):
    pattern = os.path.join(PDF_DIR, "**", "*.pdf")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)[:limit]


def extract_page_text_with_ocr(page):
    """
    Extract all text from a Docling page, including OCR output.
    Returns a list of dicts for clarity.
    """
    page_texts = []
    if hasattr(page, "elements"):
        for i, el in enumerate(page.elements):
            el_text = el.text if hasattr(el, "text") and el.text else ""
            el_md = el.markdown if hasattr(el, "markdown") and el.markdown else ""
            el_content = el.content if hasattr(el, "content") and isinstance(el.content, str) else ""

            # Only include non-empty fields
            page_texts.append({
                "element_index": i + 1,
                "text": el_text,
                "markdown": el_md,
                "content": el_content
            })
    return page_texts


def main():
    pdfs = find_pdfs(3)
    if not pdfs:
        print(f"[ERROR] No PDFs found under {PDF_DIR}")
        return

    print(f"Found {len(pdfs)} pdf(s) to process:\n")

    loader = CustomDocumentLoader()
    for pdf_path in pdfs:
        name = Path(pdf_path).name
        print(f"\n--- Processing: {name} ---")
        start = time.time()

        # Convert using Docling (RapidOCR)
        try:
            document_text = loader.process_document(pdf_path=pdf_path)

        except Exception as e:
            print(f"[ERROR] Conversion failed for {name}: {e}")
            continue

        # Output TXT path
        out_dir = os.path.dirname(__file__)
        out_txt_path = os.path.join(out_dir, f"{Path(pdf_path).stem}_OCR_DUMP.txt")

        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(document_text)

        
        elapsed = time.time() - start
        print(f"Finished {name} in {elapsed:.1f}s")

    print("\nTest run complete.")


if __name__ == '__main__':
    main()
