# ClinicalMind

A governed agentic RAG pipeline for clinical
document intelligence — built on FDA drug labels
with GxP-compliant audit trail.

## What It Does

Answers clinical questions from FDA drug labels
with source citations, confidence scoring, and
full audit trail. Designed for regulated pharma
environments where AI outputs must be trustworthy
and auditable.

## Why It's Different

Most RAG systems answer every query regardless of
confidence. ClinicalMind routes low-confidence
queries to human review rather than generating
uncertain clinical answers.

In pharma — an uncertain AI answer is more
dangerous than no answer.

## Architecture

```
FDA PDF → Section Chunking → Pinecone
                                │
Query → LangGraph Pipeline ─────┘
         │
         ├── Retrieval Agent
         ├── Confidence Check (threshold: 0.5)
         │     ├── High → Response Agent (Claude)
         │     └── Low  → Human Review Flag
         └── Audit Log (append-only, every query)
```

See `docs/architecture.md` for full diagram
and design decisions.

## Stack

- **Orchestration:** LangGraph
- **LLM:** Claude Sonnet (Anthropic)
- **Vector Store:** Pinecone
- **Embeddings:** OpenAI text-embedding-3-small
- **PDF Extraction:** PyMuPDF
- **Drug Label:** Leqembi (Lecanemab) — FDA approved
  2023, Alzheimer's disease

## Key Design Decisions

**Section-aware chunking:**
FDA labels have defined sections. Standard chunking
crosses boundaries — mixing contraindications with
dosage content. This pipeline respects section
structure.

**Confidence threshold gate:**
Queries below 0.5 confidence route to human review.
Modelled on regulated environment design principles.

**Append-only audit trail:**
Every query logged — timestamp, scores, response
preview, human review flag. Inspired by 21 CFR
Part 11 electronic records requirements.

## Project Structure

```
clinicalmind/
├── README.md
├── data/
│   └── leqembi_label.pdf
├── src/
│   ├── 01_load_documents.py
│   ├── 02_chunk_documents.py
│   ├── 03_create_index.py
│   ├── 04_store_embeddings.py
│   ├── 05_basic_query.py
│   ├── 06_architected_ingestion.py
│   ├── 07_filtered_query.py
│   └── 08_langgraph_pipeline.py
├── docs/
│   ├── architecture.md
│   ├── audit_log.jsonl
│   ├── chunks_preview.json
│   └── nodes_preview.json
└── .gitignore
```

## Setup

```bash
# Clone repo
git clone https://github.com/
  YOUR_USERNAME/clinicalmind.git

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add API keys
cp .env.example .env
# Edit .env with your keys

# Run ingestion
python src/06_architected_ingestion.py

# Run pipeline
python src/08_langgraph_pipeline.py
```

## Sample Output

```
Query: What are the indications for Leqembi?
Section: INDICATIONS AND USAGE

[Confidence: 0.7646 → Response]

LEQEMBI is indicated for treatment of
Alzheimer's disease in patients with mild
cognitive impairment or mild dementia.
[Source 1]

---

Query: What is dark matter made of?

[Confidence: 0.1259 → Human Review]

⚠️ HUMAN REVIEW REQUIRED
Confidence too low for reliable clinical answer.
```

## Audit Trail Sample

```json
{
  "timestamp": "2026-05-30T01:33:55",
  "query": "What are the indications?",
  "confidence_score": 0.7646,
  "requires_human_review": false,
  "chunks_retrieved": 1,
  "response_preview": "LEQEMBI is indicated..."
}
```