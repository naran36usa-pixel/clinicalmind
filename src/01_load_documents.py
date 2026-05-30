from dotenv import load_dotenv
import fitz  # PyMuPDF
import os

load_dotenv()

# Load PDF with PyMuPDF
pdf_path = "data/lecanemab_label.pdf"

print("Loading document with PyMuPDF...")
doc = fitz.open(pdf_path)

# Extract text page by page
full_text = ""
for page_num, page in enumerate(doc):
    text = page.get_text()
    full_text += f"\n--- Page {page_num + 1} ---\n{text}"

print(f"Total pages: {len(doc)}")
print(f"Total characters: {len(full_text)}")
print(f"\nFirst 500 characters:")
print(full_text[:500])

# Save extracted text for review
with open("docs/extracted_text.txt", "w", 
          encoding="utf-8") as f:
    f.write(full_text)

print("\n--- Text saved to docs/extracted_text.txt ---")
doc.close()