# ClinicalMind — Architecture

## System Overview

ClinicalMind is a governed agentic RAG pipeline
for clinical document intelligence. Built on FDA
drug labels with GxP-compliant audit trail.

## Architecture Diagram

```
┌─────────────────────────────────────────────┐
│           CLINICALMIND PIPELINE             │
└─────────────────────────────────────────────┘

INGESTION LAYER
───────────────
FDA Drug Label PDF
        │
        ▼
PyMuPDF Text Extraction
        │
        ▼
Section-Aware Chunking
(respects FDA label structure)
        │
        ▼
Metadata Enrichment
{drug_name, section_name, document_type}
        │
        ▼
OpenAI Embeddings
(text-embedding-3-small, 1536 dim)
        │
        ▼
Pinecone Vector Store
(metadata-filtered index)


QUERY LAYER — LangGraph State Machine
──────────────────────────────────────

User Query + Section Filter
        │
        ▼
┌───────────────────┐
│  Retrieval Agent  │
│  - Embed query    │
│  - Filter by      │
│    section +drug  │
│  - Fetch top-k    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Confidence Check  │
│  - Score > 0.5?   │
└────────┬──────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│Response│ │ Human Review │
│ Agent  │ │    Flag      │
│(Claude)│ │  ⚠️ flagged  │
└────┬───┘ └──────┬───────┘
     │             │
     └──────┬──────┘
            │
            ▼
┌───────────────────┐
│    Audit Log      │
│  Append-only      │
│  21 CFR Part 11   │
│  pattern          │
└───────────────────┘
```

## Key Design Decisions

### 1. Section-Aware Chunking
FDA labels have defined sections. Fixed-size chunking
crosses section boundaries — mixing contraindications
with dosage content. Section-aware chunking keeps
chunks within named sections.

**Impact:** Retrieval precision improved significantly.
Indications query correctly returns indications content
not adverse reactions content.

### 2. Confidence Threshold Gate
Queries below 0.5 confidence route to human review
rather than generating uncertain answers.

**Rationale:** In regulated pharma environments an
uncertain AI answer has real consequences. The system
refuses to answer rather than hallucinate.

### 3. Append-Only Audit Trail
Every query — pass or fail — generates an audit log
entry. No overwrites. Full traceability.

**Rationale:** Modelled on 21 CFR Part 11 electronic
records requirements. AI outputs in regulated
environments need the same auditability as data
pipelines.

### 4. Metadata-Filtered Retrieval
Each vector carries section_name and drug_name
metadata. Queries filter before searching — not after.

**Rationale:** Filtering before retrieval guarantees
containment. Adding new drugs doesn't degrade
retrieval quality.

## Stack

| Component | Tool | Reason |
|---|---|---|
| PDF extraction | PyMuPDF | Most reliable for FDA labels |
| Embeddings | OpenAI text-embedding-3-small | Cost efficient, well documented |
| Vector store | Pinecone | Managed, metadata filtering |
| Agent orchestration | LangGraph | Conditional routing, state machine |
| LLM | Claude Sonnet | Strong clinical reasoning |
| Audit trail | JSONL append-only | Simple, GxP-pattern compliant |

## Failure Modes

| Scenario | System Behaviour |
|---|---|
| Low retrieval confidence | Routes to human review |
| Irrelevant query | Routes to human review |
| Section not in index | Empty retrieval, human review |
| API failure | Error captured in state, logged |