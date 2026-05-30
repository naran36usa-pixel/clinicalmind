```markdown
# ClinicalMind — Architecture

## System Overview

ClinicalMind is a governed RAG pipeline for clinical document intelligence built on FDA drug labels. It demonstrates confidence-based routing, structured retrieval, and traceable execution logs.

The system is designed around a key principle:

> In regulated environments, uncertain answers should be escalated rather than generated.

---

## Architecture Diagram


```

┌─────────────────────────────────────────────┐
│              CLINICALMIND PIPELINE          │
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
(preserves FDA label structure)
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
User Query + Optional Filters
│
▼
┌───────────────────┐
│  Retrieval Agent  │
│ - Embed query     │
│ - Apply filters   │
│ - Fetch top-k     │
└────────┬──────────┘
│
▼
┌───────────────────┐
│ Confidence Check  │
│ - Compute score   │
│ - Apply threshold │
└────────┬──────────┘
│
┌────┴────┐
│         │
▼         ▼
Low Score   High Score
│         │
▼         ▼
Human Review    Response Generation
Flagged ⚠️      (Claude Sonnet)
│         │
└──────┬──┘
│
▼
Audit Log

```

---

## Key Design Decisions

### 1. Section-Aware Chunking
FDA labels contain structured sections such as indications, warnings, and dosage information. Instead of using fixed-size chunking, ClinicalMind preserves section boundaries during ingestion. This helps maintain semantic integrity by preventing unrelated clinical content from being mixed within embeddings.

### 2. Confidence-Based Routing
The system evaluates retrieval confidence before invoking the LLM. Queries below a configurable threshold are routed to a human review path instead of generating a response. This design emphasizes controlled failure over forced generation.

### 3. Traceable Execution Logging
Each query execution produces a structured log entry capturing:
- Timestamp
- Query text
- Confidence score
- Routing decision
- Retrieved section metadata
- Response preview

The logging system is append-only during runtime execution to support traceability during testing and debugging.

### 4. Metadata-Filtered Retrieval
Each vector entry includes metadata such as:
- `drug_name`
- `section_name`
- `document_type`

Filtering is applied before retrieval to constrain search scope and improve relevance. This enables scaling to multiple documents without degrading retrieval precision.

---

## Technology Stack

| Component | Tool | Purpose |
| :--- | :--- | :--- |
| **PDF Extraction** | PyMuPDF | FDA label parsing |
| **Embeddings** | OpenAI text-embedding-3-small | Semantic representation |
| **Vector Store** | Pinecone | Similarity search with metadata filtering |
| **Orchestration** | LangGraph | State-based workflow control |
| **LLM** | Claude 3.5 Sonnet | Response generation |
| **Logging** | JSONL | Structured execution traces |

---

## Failure Modes

| Scenario | System Behavior |
| :--- | :--- |
| **Low retrieval confidence** | Routed to human review queue |
| **Out-of-domain query** | Flagged for manual review |
| **Missing section in index** | No response generated; execution stopped |
| **External API failure** | Error captured and logged inside execution state |

---

## Limitations

* Designed for a single FDA label dataset in this version.
* Evaluation is demonstration-scale and not production-validated.
* Human review workflow is simulated via log routing.
* Audit logs are local flat files and not cryptographically secured.
* Not intended for clinical decision-making or regulatory use.

---

## Future Direction

The next planned enhancement is support for multiple FDA drug labels with metadata-aware retrieval and cross-document filtering. The core governance and routing architecture will remain unchanged, enabling scalable expansion while preserving controlled failure behavior.

```