from dotenv import load_dotenv
import os
import re
import json
import fitz
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone

load_dotenv()

# ── Constants ──────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
INDEX_NAME = "clinicalmind"
DRUG_NAME = "Leqembi (Lecanemab)"

# Valid FDA sections to extract
VALID_SECTIONS = [
    "INDICATIONS AND USAGE",
    "DOSAGE AND ADMINISTRATION",
    "DOSAGE FORMS AND STRENGTHS",
    "CONTRAINDICATIONS",
    "WARNINGS AND PRECAUTIONS",
    "ADVERSE REACTIONS",
    "USE IN SPECIFIC POPULATIONS",
    "DESCRIPTION",
    "CLINICAL PHARMACOLOGY",
    "NONCLINICAL TOXICOLOGY",
    "CLINICAL STUDIES",
    "HOW SUPPLIED/STORAGE AND HANDLING",
    "PATIENT COUNSELING INFORMATION"
]

# ── Initialize ─────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(INDEX_NAME)

Settings.embed_model = OpenAIEmbedding(
    model=EMBEDDING_MODEL,
    api_key=os.getenv("OPEN_AI_API_KEY")
)

# ── Load Document ──────────────────────────────────────
print("Step 1: Loading document with PyMuPDF...")
PDF_PATH = os.getenv("PDF_PATH", "data/lecanemab_label.pdf")
doc = fitz.open(PDF_PATH)

# Extract text page by page
pages_text = []
for page_num, page in enumerate(doc):
    text = page.get_text()
    pages_text.append({
        "page_num": page_num + 1,
        "text": text
    })

full_text = "\n".join([p["text"] for p in pages_text])
print(f"Loaded {len(pages_text)} pages")
print(f"Total characters: {len(full_text)}")

# ── Section Extraction ─────────────────────────────────
print("\nStep 2: Extracting FDA sections...")

# Split on numbered sections like "1 INDICATIONS AND USAGE"
section_pattern = r'\n(\d{1,2}\s+[A-Z][A-Z\s,/]{5,})\n'
split_content = re.split(section_pattern, full_text)

# Map sections to their content
sections_found = {}
current_section = None

for item in split_content:
    item_stripped = item.strip()

    if not item_stripped:
        continue

    # Check if this is a section header
    if re.match(r'^\d{1,2}\s+[A-Z][A-Z\s,/]{5,}$',
                item_stripped):

        # Extract just the section name without number
        section_name = re.sub(
            r'^\d{1,2}\s+', '', item_stripped
        ).strip()

        # Only keep valid FDA sections
        if section_name in VALID_SECTIONS:
            current_section = section_name
            sections_found[current_section] = ""
            print(f"  ✅ Found section: {current_section}")
        else:
            current_section = None
    else:
        # Add content to current valid section
        if current_section:
            sections_found[current_section] += \
                item_stripped + "\n\n"

print(f"\nTotal valid sections found: {len(sections_found)}")

# ── Build Metadata Nodes ───────────────────────────────
print("\nStep 3: Building metadata nodes...")
nodes = []

for section_name, section_text in sections_found.items():
    if not section_text.strip():
        continue

    # Split section into paragraphs
    paragraphs = section_text.strip().split("\n\n")

    for idx, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()

        # Skip noise — too short or just drug name
        if len(paragraph) < 50:
            continue

        node = TextNode(
            text=paragraph,
            metadata={
                "drug_name": DRUG_NAME,
                "document_type": "FDA_LABEL",
                "section_name": section_name,
                "paragraph_index": idx,
                "pipeline_version": "1.1.0"
            }
        )
        node.excluded_llm_metadata_keys = [
            "pipeline_version",
            "paragraph_index"
        ]
        nodes.append(node)

print(f"Created {len(nodes)} metadata nodes")

# ── Preview ────────────────────────────────────────────
preview = [
    {
        "section": node.metadata["section_name"],
        "text_preview": node.text[:200],
        "char_count": len(node.text)
    }
    for node in nodes[:25]
]

with open("docs/nodes_preview.json", "w",
          encoding="utf-8") as f:
    json.dump(preview, f, indent=2)

print("Preview saved to docs/nodes_preview.json")

# ── Section Summary ────────────────────────────────────
section_summary = {}
for node in nodes:
    section = node.metadata["section_name"]
    section_summary[section] = \
        section_summary.get(section, 0) + 1

print("\nSection node counts:")
for section, count in section_summary.items():
    print(f"  {section}: {count} nodes")

# ── Store in Pinecone ──────────────────────────────────
print("\nStep 4: Storing in Pinecone...")
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index
)
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context
)

print("\n✅ Week 2 Complete")
print(f"Total nodes stored: {len(nodes)}")
print("Sections extracted and metadata indexed")