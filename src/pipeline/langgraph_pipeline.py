from dotenv import load_dotenv
import os
import json
from datetime import datetime, timezone
# change to:
"timestamp": datetime.now(timezone.utc).isoformat()
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from pinecone import Pinecone
from openai import OpenAI
import anthropic

load_dotenv()

# ── Initialize ─────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("clinicalmind")
openai_client = OpenAI(
    api_key=os.getenv("OPEN_AI_API_KEY")
)
claude_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# ── Constants ──────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5
AUDIT_LOG_PATH = "docs/audit_log.jsonl"

# ── State Definition ───────────────────────────────────
class ClinicalQueryState(TypedDict):
    query: str
    section_filter: str
    retrieved_chunks: List[str]
    retrieval_scores: List[float]
    confidence_score: float
    requires_human_review: bool
    response: str
    error: str


# ── Agent 1 — Retrieval Agent ──────────────────────────
def retrieval_agent(
    state: ClinicalQueryState
) -> ClinicalQueryState:

    print(f"\n[Retrieval Agent] Query: {state['query']}")

    try:
        # Embed query
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=state["query"]
        )
        query_embedding = response.data[0].embedding

        # Build filter
        filter_dict = None
        if state.get("section_filter"):
            filter_dict = {
                "section_name": {
                    "$eq": state["section_filter"]
                }
            }
            print(f"[Retrieval Agent] "
                  f"Filtering to: "
                  f"{state['section_filter']}")

        # Query Pinecone
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            filter=filter_dict
        )

        # Extract chunks and scores
        chunks = []
        scores = []

        for match in results.matches:
            node_content = match.metadata.get(
                "_node_content", "{}"
            )
            try:
                node_data = json.loads(node_content)
                text = node_data.get("text", "")
            except:
                text = match.metadata.get("text", "")

            chunks.append(text)
            scores.append(round(match.score, 4))
            print(f"[Retrieval Agent] "
                  f"Score: {match.score:.4f} | "
                  f"Section: "
                  f"{match.metadata.get('section_name')}")

        return {
            **state,
            "retrieved_chunks": chunks,
            "retrieval_scores": scores,
            "error": ""
        }

    except Exception as e:
        print(f"[Retrieval Agent] Error: {e}")
        return {
            **state,
            "retrieved_chunks": [],
            "retrieval_scores": [],
            "error": str(e)
        }


# ── Agent 2 — Confidence Check ─────────────────────────
def confidence_check(
    state: ClinicalQueryState
) -> ClinicalQueryState:
    """
    Evaluates retrieval quality.
    Sets confidence score and human review flag.
    This is the GxP governance gate.
    """
    print(f"\n[Confidence Check] Evaluating...")

    scores = state.get("retrieval_scores", [])

    if not scores:
        confidence = 0.0
    else:
        # Use top score as confidence signal
        confidence = max(scores)

    requires_review = confidence < CONFIDENCE_THRESHOLD

    print(f"[Confidence Check] "
          f"Score: {confidence:.4f} | "
          f"Threshold: {CONFIDENCE_THRESHOLD} | "
          f"Human review: {requires_review}")

    return {
        **state,
        "confidence_score": confidence,
        "requires_human_review": requires_review
    }


# ── Routing Function ───────────────────────────────────
def route_by_confidence(
    state: ClinicalQueryState
) -> str:
    """
    Conditional routing based on confidence.
    High confidence → generate response
    Low confidence → flag for human review
    """
    if state["requires_human_review"]:
        print("[Router] Low confidence → Human review")
        return "human_review"
    else:
        print("[Router] High confidence → Response")
        return "response"


# ── Agent 3 — Response Agent ───────────────────────────
def response_agent(
    state: ClinicalQueryState
) -> ClinicalQueryState:

    print(f"\n[Response Agent] Generating answer...")

    if not state["retrieved_chunks"]:
        return {
            **state,
            "response": "No relevant content found."
        }

    context_parts = []
    for i, chunk in enumerate(
        state["retrieved_chunks"]
    ):
        context_parts.append(
            f"[Source {i+1}]: {chunk}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a clinical document assistant
analyzing FDA drug label information.

Answer using ONLY the provided context.
If answer not in context say so clearly.
Cite which source you used.

Context:
{context}

Question: {state['query']}

Answer:"""

    try:
        claude_response = claude_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        answer = claude_response.content[0].text
        print(f"[Response Agent] Answer generated")

        return {**state, "response": answer}

    except Exception as e:
        return {
            **state,
            "response": f"Response failed: {e}"
        }


# ── Agent 4 — Human Review ─────────────────────────────
def human_review_agent(
    state: ClinicalQueryState
) -> ClinicalQueryState:
    """
    Handles low confidence queries.
    Flags for human review instead of
    returning uncertain answer.
    This is the regulated environment design decision —
    uncertain AI answer in pharma is dangerous.
    """
    print(f"\n[Human Review] Flagging query...")

    review_message = (
        f"⚠️ HUMAN REVIEW REQUIRED\n"
        f"Query: {state['query']}\n"
        f"Confidence score: "
        f"{state['confidence_score']:.4f} "
        f"(below threshold {CONFIDENCE_THRESHOLD})\n"
        f"Reason: Retrieval confidence too low to "
        f"generate a reliable clinical answer.\n"
        f"Action: Please review source documents "
        f"manually or rephrase the query."
    )

    print(f"[Human Review] {review_message}")

    return {**state, "response": review_message}


# ── Audit Log ──────────────────────────────────────────
def audit_log(
    state: ClinicalQueryState
) -> ClinicalQueryState:
    """
    Append-only audit trail for every query.
    Modelled on GxP audit trail principles —
    21 CFR Part 11 compliance pattern.
    No overwrites. Full traceability.
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": state["query"],
        "section_filter": state.get(
            "section_filter", ""
        ),
        "confidence_score": state.get(
            "confidence_score", 0.0
        ),
        "requires_human_review": state.get(
            "requires_human_review", False
        ),
        "retrieval_scores": state.get(
            "retrieval_scores", []
        ),
        "chunks_retrieved": len(
            state.get("retrieved_chunks", [])
        ),
        "response_preview": state.get(
            "response", ""
        )[:200],
        "error": state.get("error", "")
    }

    # Append-only — never overwrite
    with open(AUDIT_LOG_PATH, "a",
              encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"\n[Audit Log] Entry written — "
          f"timestamp: {log_entry['timestamp']}")

    return state


# ── Build LangGraph ────────────────────────────────────
def build_pipeline():

    workflow = StateGraph(ClinicalQueryState)

    # Add all nodes
    workflow.add_node("retrieval", retrieval_agent)
    workflow.add_node(
        "confidence_check", confidence_check
    )
    workflow.add_node("response", response_agent)
    workflow.add_node("human_review", human_review_agent)
    workflow.add_node("audit", audit_log)

    # Entry point
    workflow.set_entry_point("retrieval")

    # Linear edges
    workflow.add_edge("retrieval", "confidence_check")

    # Conditional routing — the key Stop B addition
    workflow.add_conditional_edges(
        "confidence_check",
        route_by_confidence,
        {
            "response": "response",
            "human_review": "human_review"
        }
    )

    # Both paths converge at audit
    workflow.add_edge("response", "audit")
    workflow.add_edge("human_review", "audit")
    workflow.add_edge("audit", END)

    return workflow.compile()


# ── Run Tests ──────────────────────────────────────────
if __name__ == "__main__":

    pipeline = build_pipeline()
    print("Pipeline compiled successfully")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    test_queries = [
        {
            "query": "What are the indications "
                     "for Leqembi?",
            "section_filter": "INDICATIONS AND USAGE"
        },
        {
            "query": "What are the contraindications?",
            "section_filter": "CONTRAINDICATIONS"
        },
        {
            "query": "What are the warnings?",
            "section_filter": "WARNINGS AND PRECAUTIONS"
        },
        {
            "query": "What is the chemical composition "
                     "of dark matter?",
            "section_filter": ""
        }
    ]

    for test in test_queries:
        print(f"\n{'='*60}")

        initial_state = ClinicalQueryState(
            query=test["query"],
            section_filter=test["section_filter"],
            retrieved_chunks=[],
            retrieval_scores=[],
            confidence_score=0.0,
            requires_human_review=False,
            response="",
            error=""
        )

        result = pipeline.invoke(initial_state)

        print(f"\nFINAL ANSWER:")
        print(result["response"])
        print(f"Confidence: {result['confidence_score']}")
        print(
            f"Human review: "
            f"{result['requires_human_review']}"
        )

    print("\n✅ Stop B Complete")
    print("Check docs/audit_log.jsonl for audit trail")