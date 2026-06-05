# ClinicalMind — Architecture

## System Overview

ClinicalMind is a governed, enterprise-grade RAG pipeline for clinical document intelligence built on multi-drug FDA labels. It demonstrates deterministic hybrid routing, structured isolated retrieval, query hydration, and traceable execution logs.

The system is designed around a core GxP engineering principle:

> In regulated environments, uncertain answers must be escalated rather than generated.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────┐
│           CLINICALMIND PIPELINE             │
└─────────────────────────────────────────────┘

INGESTION LAYER (Config-Driven)
───────────────────────────────
Multi-Drug FDA Label PDFs + JSON Configs
│
▼
PyMuPDF Text Extraction & Section-Aware Chunking
│
▼
Metadata Enrichment
{drug_name, section_name, document_type}
│
▼
OpenAI Embeddings (text-embedding-3-small, 1536 dim) ──► Pinecone Vector Store

QUERY LAYER — LangGraph State Machine (Hybrid Flow)
───────────────────────────────────────────────────

Streamlit UI User Query & Sidebar Token Constraints
│
▼
┌────────────────────────────────────────────────────────┐
│ Hybrid Intent Router Node                              │
│ - Check for Explicit UI Sidebar Parameters             │
│   ├─► IF PRESENT: Enforce Boundaries & Bypass LLM      │
│   └─► IF ABSENT: Fallback to LLM Extraction (Claude)  │
└────────────────────────┬───────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│ Isolated Retrieval Agent Node                          │
│ - Context Hydration Engine (Injects Target Context)    │
│ - Calculate Embeddings on Hydrated Payload             │
│ - Apply Compound Metadata Filters Layer                │
└────────────────────────┬───────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────┐
│ GxP Confidence Check Gate                              │
│ - Evaluate Max Vector Match Score against Threshold    │
└────────────────────────┬───────────────────────────────┘
│
┌────────────┴────────────┐
│ Score < 0.5000          │ Score >= 0.5000
▼                         ▼
┌───────────────────────┐ ┌───────────────────────┐
│   Human Escrow Node   │ │Grounded Response Node │
│ (Safety Isolation ⚠️) │ │ (Claude 4.5 Sonnet)   │
└───────────┬───────────┘ └───────────┬───────────┘
│                         │
└────────────┬────────────┘
             ▼
┌────────────────────────────────────────────────────────┐
│ 21 CFR Part 11 Audit Trail Logger                      │
│ - Append-Only Transaction Ledger (docs/audit_log.jsonl)│
└────────────────────────┬───────────────────────────────┘
│
▼
Render Streamlit UI
```

---

## Key Design Decisions

### 1. Section-Aware Chunking
FDA labels contain structured, highly explicit sections (e.g., `INDICATIONS AND USAGE`, `WARNINGS AND PRECAUTIONS`). ClinicalMind abandons arbitrary token-length splitting in favor of strict section boundary preservation during ingestion. This maintains semantic integrity and blocks toxic cross-contamination of unrelated clinical variables within individual vector embeddings.

### 2. Hybrid Intent-Driven Routing
To eliminate the architectural disconnect between user interfaces and backend routing, the pipeline uses a hybrid evaluation pattern. If a user defines targeted constraints using the Streamlit dashboard dropdowns, the entry node intercepts execution, enforces those boundaries, and bypasses LLM inference entirely. If the query arrives unstructured, the system falls back to a semantic entity extraction parser to map the target domain.

### 3. Context-Hydrated Isolated Retrieval
Raw, shorthand user questions (e.g., *"give me the adverse reactions"*) naturally yield poor proximity scores in raw vector spaces due to keyword scarcity. The retrieval engine solves this by dynamically hydrating the query payload with explicit structured metadata context before text vectorization. This optimizes semantic proximity math at the database layer while maintaining sharp pre-retrieval metadata isolation inside Pinecone.

### 4. Confidence-Based Governance Gate
The system enforces a strict mathematical safety boundary ($0.5000$). Every retrieved payload is checked at runtime. If the maximum vector proximity score drops below the threshold, the system triggers an emergency brake, blocking downstream generation to eliminate hallucination risks, and diverts the execution state to a human escrow node.

### 5. Config-Driven Ingestion Engine
The ingestion layer supports multi-drug scaling out-of-the-box. Adding a new pharmaceutical product requires zero pipeline code modifications. The ingestion engine dynamically discovers newly added PDF files and corresponding JSON configuration mappings at runtime, automating chunk parsing, indexing, and vector isolation seamlessly.

### 6. Traceable 21 CFR Part 11 Audit Trail
Every single transaction pipeline traversal is recorded immutably in an append-only JSONL log ledger. Each log entry captures the complete deterministic lifecycle of the execution state: timestamp, query, hydrated input, metadata filters, array of raw mathematical match scores, compliance routing result, and a generation payload preview.

---

## Technology Stack

| Component | Tool | Purpose |
|---|---|---|
| PDF Extraction | PyMuPDF | FDA structured label parsing |
| Embeddings | OpenAI text-embedding-3-small | Semantic vector representation |
| Vector Store | Pinecone | Similarity search with native metadata filtering |
| Orchestration | LangGraph | State-based deterministic workflow control |
| LLM Engine | Claude 4.5 Sonnet | Entity parsing fallback & Grounded response generation |
| Frontend | Streamlit | Control dashboard & verification UI |
| Logging | JSONL | Append-only GxP traceable transaction execution ledger |

---

## Failure Modes & Mitigations

| Scenario | System Behavior | Mitigation Strategy |
|---|---|---|
| Low retrieval confidence ($< 0.5000$) | Intercepted by Gate Node | Block generation; Route to simulated human escrow node. |
| Missing keyword / Vague pronoun | Intercepted by Retrieval Node | Context Hydration Engine appends structured drug tokens to vector inputs. |
| Cross-contamination risk | Filtered at DB layer | Multi-drug scoping isolated explicitly via pre-retrieval metadata filters. |
| External API Downstream Failure | Exception State Caught | State machine captures error stack and seals transaction directly inside the audit ledger. |

---

## Limitations

- Evaluation is scaled for multi-drug demonstration and not yet production-validated.
- The human review dashboard resolution workflow is simulated within the state graph.
- Local JSONL audit logs lack cryptographic hashing or distributed security protections.
- The platform functions as a document intelligence utility and is not intended for direct clinical decision-making.

---

## Future Direction

1. **Cryptographic Trail Verification:** Upgrade the append-only audit ledger to utilize SHA-256 block hashing or ledger database endpoints to achieve absolute non-repudiation tracking.
2. **Production GxP Validation:** Formally expand testing suites to meet strict compliance validation requirements for healthcare and life sciences cloud environments.
3. **Cross-Section Synthesis:** Refactor the retrieval agent to support compound list evaluation (e.g., scanning `ADVERSE REACTIONS` and `BOXED WARNINGS` concurrently) when answering highly complex safety queries.
