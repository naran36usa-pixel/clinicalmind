import os
import fitz  # PyMuPDF

def load_pdf_document(config):
    """
    Step 1: Extract raw text page-by-page from the target PDF asset.
    Uses robust path anchoring relative to project root.
    """
    file_name = config.get("file_name")
    if not file_name:
        raise KeyError(f"Configuration for '{config.get('drug_name', 'Unknown')}' lacks a 'file_name' field.")

    # Anchor paths to the project root directory
    # Go up two levels from src/ingestion/ to reach the main clinicalmind root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pdf_path = os.path.join(base_dir, "data", "labels", file_name)
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Expected PDF asset missing at resolved path: {pdf_path}")

    print(f"  [Load] Extraction initialized for: {config['drug_name']}")
    doc = fitz.open(pdf_path)
    
    pages_text = []
    for page_num, page in enumerate(doc):
        pages_text.append(page.get_text())
        
    full_text = "\n".join(pages_text)
    print(f"  [Load] Complete. Character count: {len(full_text)}")
    doc.close()
    
    return full_text