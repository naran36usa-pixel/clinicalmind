# ClinicalMind

A governed clinical RAG prototype built on FDA drug labels, demonstrating confidence-based response routing, source-grounded retrieval, and audit trail logging.

---

## 📋 Problem Statement

Clinical and regulatory teams frequently need to locate critical information from lengthy FDA drug labels, including indications, contraindications, warnings, and dosing guidance.

* **The Traditional Gap:** Conventional keyword search lacks semantic understanding and document awareness.
* **The LLM Risk:** Generic AI assistants may generate answers without sufficient supporting evidence or operate outside the scope of approved source material.

ClinicalMind explores a **governance-first retrieval architecture** where low-confidence queries are routed for human review instead of forcing an AI-generated response.

> **Core Philosophy:** In regulated environments, an uncertain answer can be more dangerous than no answer.

---

## ✨ Key Features

* **Section-aware FDA label chunking** – Preserves structural document boundaries during indexing.
* **Retrieval-Augmented Generation (RAG)** – Restricts generation to retrieved source context.
* **Confidence-based response routing** – Evaluates retrieval quality before generating a response.
* **Human review escalation** – Handles low-confidence and out-of-domain queries through a controlled failure path.
* **Source-cited responses** – Maps generated answers back to FDA label sections.
* **Query-level audit logging** – Records execution metadata and routing decisions.
* **LangGraph workflow orchestration** – Coordinates retrieval, evaluation, routing, and response generation.

---

## 📐 Architecture Overview

```text
FDA Label PDF
      │
      ▼
Section Parser
      │
      ▼
Chunk Generator
      │
      ▼
Embeddings
      │
      ▼
Pinecone Vector Store
      │
      ▼
LangGraph Workflow
 ├── Retrieval Node
 ├── Confidence Evaluation
 ├── Response Generation
 └── Audit Logger
```

### Query Flow

```text
User Query
     │
     ▼
Retrieve Relevant Sections
     │
     ▼
Calculate Confidence Score
     │
 ┌───┴───────────┐
 │               │
 ▼               ▼
Low Score    High Score
 │               │
 ▼               ▼
Human Review   Generate Response
 │               │
 └───────┬───────┘
         ▼
      Audit Log
```

---

## 🧠 Design Decisions

### Section-Aware Chunking

FDA labels follow a well-defined structure. Instead of splitting text purely by token or character count, ClinicalMind preserves section boundaries during chunking.

This reduces the risk of mixing unrelated clinical content across embeddings and improves retrieval relevance.

### Confidence-Based Routing

The system evaluates retrieval confidence before generating a response.

Queries below a configurable threshold are routed for review rather than automatically answered. The objective is to demonstrate a controlled failure path rather than maximize answer coverage at the expense of reliability.

### Audit Logging

Each query execution records key execution metadata, including:

* Timestamp
* Query text
* Confidence score
* Routing decision
* Retrieved section metadata
* Response preview

This provides traceability for system behavior during testing and evaluation.

---

## 📊 Confidence Methodology

The current implementation derives confidence from retrieval relevance scores returned by the vector search process.

A threshold of **0.50** is used to determine whether a response should be generated or routed for review.

> ⚠️ **Note:** This threshold was selected for demonstration purposes and has not been formally calibrated against a validation dataset.

Future iterations may incorporate additional confidence signals such as reranker scores and model-based evaluation.

---

## 🛠️ Technology Stack

| Component              | Technology                         |
| ---------------------- | ---------------------------------- |
| Workflow Orchestration | LangGraph                          |
| LLM                    | Claude 4 Sonnet                  |
| Embeddings             | OpenAI `text-embedding-3-small`    |
| Vector Database        | Pinecone                           |
| PDF Processing         | PyMuPDF                            |
| Source Document        | FDA Leqembi (Lecanemab) Drug Label |

---

## 🗂️ Repository Structure

```text
clinicalmind/
├── data/
│   └── leqembi_label.pdf
│
├── src/
│   ├── ingestion/
│   ├── indexer/
│   ├── pipeline/
│   └── query/
│
├── docs/
│   ├── architecture.md
│   ├── audit_log.jsonl
│   └── sample_outputs/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### Clone Repository

```bash
git clone https://github.com/naran36usa-pixel/clinicalmind.git
cd clinicalmind
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate

# Windows
# .\venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_key
OPEN_AI_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=your_environment
```

---

## 🚀 Running the Project

### Run Ingestion

```bash
python src/ingestion/ingestion_pipeline.py
```

### Load & Chunk Documents

```bash
python src/ingestion/load_documents.py
python src/ingestion/chunk_documents.py
```

### Build Index & Store Embeddings

```bash
python src/indexer/create_index.py
python src/indexer/store_embeddings.py
```

### Launch Pipeline

```bash
python src/pipeline/langgraph_pipeline.py
```

### Run Queries

```bash
python src/query/basic_query.py
python src/query/filtered_query.py
```

---

## 🧪 Example Execution

### Scenario A: High-Confidence Query

**Query**

```text
What are the indications for Leqembi?
```

**Output**

```text
Confidence Score: 0.76

LEQEMBI is indicated for the treatment of Alzheimer's disease in
patients with mild cognitive impairment or mild dementia.

Source: INDICATIONS AND USAGE
```

---

### Scenario B: Low-Confidence Query

**Query**

```text
What is dark matter made of?
```

**Output**

```text
Confidence Score: 0.12

⚠️ HUMAN REVIEW REQUIRED

Query does not appear to be supported by the indexed clinical source material.
```

---

## 📜 Sample Audit Log Entry

```json
{
  "timestamp": "2026-05-30T01:33:55Z",
  "query": "What are the indications for Leqembi?",
  "confidence_score": 0.7646,
  "requires_human_review": false,
  "retrieved_sections": [
    "INDICATIONS_AND_USAGE"
  ],
  "response_preview": "LEQEMBI is indicated..."
}
```

---

## 🛑 Limitations

* Single FDA label dataset
* Demonstration-scale evaluation
* Simulated human review workflow
* Local audit logs are mutable and not cryptographically protected
* Not validated for GxP or regulatory compliance
* Not intended for clinical decision-making

---

## 🚀 Next Enhancement

* Support multiple FDA drug labels with metadata-aware retrieval and filtering

---

## ⚖️ Disclaimer

This project is a technical demonstration of governance-oriented RAG architecture patterns.

It is not a validated clinical system and should not be used for patient care, medical advice, or regulatory decision-making.
