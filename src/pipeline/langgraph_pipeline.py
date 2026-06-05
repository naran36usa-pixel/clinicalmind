import os
import json
from datetime import datetime, timezone
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

# Path anchoring to ensure environment keys load properly from root level
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(dotenv_path=os.path.join(base_dir, ".env"))

from langgraph.graph import StateGraph, END
from pinecone import Pinecone
from openai import OpenAI
import anthropic

# ── Initialize Production Infrastructure Clients ───────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("clinicalmind")
openai_client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Governance Constants ─────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5
AUDIT_LOG_PATH = os.path.join(base_dir, "docs", "audit_log.jsonl")

# ── State Definition ───────────────────────────────────
class ClinicalQueryState(TypedDict):
    query: str
    detected_drugs: List[str]       # Structured extraction target
    section_filter: str             # Standardized target section 
    retrieved_chunks: List[str]
    retrieval_scores: List[float]
    confidence_score: float
    requires_human_review: bool
    response: str
    error: str

# ── Node 1: Hybrid Intent Extraction Router ────────────
def intent_router_agent(state: ClinicalQueryState) -> ClinicalQueryState:
    """
    Supervisor Node: Analyzes query boundaries. Establishes a hybrid model
    that respects explicit UI/API inputs before falling back to LLM extraction.
    """
    print(f"\n🧠 [Intent Router] Evaluating query boundaries: '{state['query']}'")
    
    # ── HYBRID MODE ENFORCEMENT ──────────────────────────────────────────────────────
    # If the user selected a drug explicitly in the Streamlit UI sidebar, use it directly.
    # This prevents generic phrasing like "give me indications" from triggering short-circuits.
    if state.get("detected_drugs") and len(state["detected_drugs"]) > 0 and state["detected_drugs"][0] != "":
        print(f"🧬 [Intent Router] Enforcing explicit UI-driven boundaries: {state['detected_drugs']} | Section: {state['section_filter']}")
        return {
            **state,
            "error": ""
        }
    # ──────────────────────────────────────────────────────────────────────────────────

    print("[Intent Router] No UI boundary constraints detected. Running LLM semantic entity parser...")
    prompt = f"""You are a clinical supervisor routing system analyzing an FDA drug label database query.
Extract the target pharmaceutical product(s) and the specific fda label section requested.

Available Drugs in System: Keytruda, Leqembi, Trodelvy.
Available Sections in System: INDICATIONS AND USAGE, CONTRAINDICATIONS, WARNINGS AND PRECAUTIONS, ADVERSE REACTIONS.

Your response must be strict raw JSON matching this format exactly, with no additional conversational text:
{{
    "detected_drugs": ["Keytruda"],
    "section_filter": "WARNINGS AND PRECAUTIONS"
}}

If no matching drug or section is found, return them as empty lists/strings.
Query: {state['query']}
Response:"""

    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw_text = response.content[0].text.strip()
        
        # Robust Markdown block sanitization to handle model text leaking
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()
        raw_text = raw_text.strip("`").strip()
        
        extraction = json.loads(raw_text)
        
        # Standardize colloquial extractions to match our exact database metadata strings
        drug_map = {
            "keytruda": "Keytruda (Pembrolizumab)",
            "leqembi": "Leqembi (Lecanemab)",
            "trodelvy": "Trodelvy (Sacituzumab govitecan-hziy)"
        }
        
        raw_drugs = extraction.get("detected_drugs", [])
        if isinstance(raw_drugs, str):  
            raw_drugs = [raw_drugs]
            
        normalized_drugs = [drug_map[d.lower().strip()] for d in raw_drugs if d.lower().strip() in drug_map]
        
        print(f"[Intent Router] Extracted Entities -> Drugs: {normalized_drugs} | Section: {extraction.get('section_filter')}")
        
        return {
            **state,
            "detected_drugs": normalized_drugs,
            "section_filter": extraction.get("section_filter", ""),
            "error": ""
        }
    except Exception as e:
        print(f"[Intent Router] Error parsing intent: {e}")
        return {**state, "error": f"Intent extraction failure: {str(e)}"}

# ── Node 2: Isolated Retrieval Agent ────────────────────
def retrieval_agent(state: ClinicalQueryState) -> ClinicalQueryState:
    """
    Retrieval Node: Executes dynamic vector space lookups.
    Forces strict drug-level isolation to prevent cross-contamination and
    hydrates generic queries with structured metadata context for optimized search math.
    """
    if state.get("error"):
        return state

    if not state["detected_drugs"] or state["detected_drugs"] == [""]:
        print("[Retrieval Agent] Short-circuiting execution: No recognized drug domain specified.")
        return {
            **state,
            "retrieved_chunks": [],
            "retrieval_scores": [],
            "error": ""  
        }

    print(f"⚙️ [Retrieval Agent] Executing isolated database query loop for {state['detected_drugs']}...")

    # ── METADATA QUERY HYDRATION ENGINE ──────────────────────────────────────────────
    # Forces the embedding model to recognize domain context even if the user typed vague pronouns.
    embedding_input = state["query"]
    target_drug = state["detected_drugs"][0]
    clean_brand = target_drug.split("(")[0].strip().lower()
    
    # Hydrate if query uses pronouns or lacks the concrete brand name keyword
    if "this" in embedding_input.lower() or "the drug" in embedding_input.lower() or clean_brand not in embedding_input.lower():
        embedding_input = f"{embedding_input} (Context: Target pharmaceutical product is {target_drug})"
        print(f"🔧 [Retrieval Agent] Context Hydration Injection -> Optimized String: '{embedding_input}'")
    # ──────────────────────────────────────────────────────────────────────────────────

    try:
        # Generate embedding using the optimized context-aware string
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=embedding_input
        )
        query_embedding = response.data[0].embedding

        # Build compound metadata filter conditions matching Pinecone's exact syntax
        filter_dict = {
            "drug_name": {"$in": state["detected_drugs"]}
        }
        
        if state.get("section_filter"):
            filter_dict["section_name"] = {"$eq": state["section_filter"]}
            
        print(f"[Retrieval Agent] Applying Pinecone Metadata Filters: {filter_dict}")

        results = pinecone_index.query(
            vector=query_embedding,
            top_k=4,
            include_metadata=True,
            filter=filter_dict
        )

        chunks = []
        scores = []

        for match in results.matches:
            text = match.metadata.get("text", "")
            if not text and "_node_content" in match.metadata:
                try:
                    text = json.loads(match.metadata["_node_content"]).get("text", "")
                except:
                    pass

            if text:
                chunks.append(text)
                scores.append(round(match.score, 4))
                print(f"  📑 [Match Score: {match.score:.4f}] Product: {match.metadata.get('drug_name')} | Sec: {match.metadata.get('section_name')}")

        return {
            **state,
            "retrieved_chunks": chunks,
            "retrieval_scores": scores
        }

    except Exception as e:
        print(f"[Retrieval Agent] Critical DB Failure: {e}")
        return {**state, "error": str(e)}

# ── Node 3: GxP Confidence Check Gate ───────────────────
def confidence_check(state: ClinicalQueryState) -> ClinicalQueryState:
    """
    Governance Gate: Enforces clinical threshold limits before allowing LLM inference.
    """
    print(f"🛡️ [Confidence Check] Evaluating mathematical validation boundaries...")
    scores = state.get("retrieval_scores", [])
    
    if state.get("error"):
        return {**state, "confidence_score": 0.0, "requires_human_review": True}

    confidence = max(scores) if scores else 0.0
    requires_review = confidence < CONFIDENCE_THRESHOLD

    print(f"[Confidence Check] Max Vector Score: {confidence:.4f} | Gate Limit: {CONFIDENCE_THRESHOLD} | Escalation Status: {requires_review}")

    return {
        **state,
        "confidence_score": confidence,
        "requires_human_review": requires_review
    }

# ── Routing Decision Determinant ─────────────────────
def route_by_confidence(state: ClinicalQueryState) -> str:
    if state["requires_human_review"]:
        print("🔀 [Router Path] Diverting to human escrow node.")
        return "human_review"
    else:
        print("🔀 [Router Path] Advancing to final response generation node.")
        return "response"

# ── Node 4: Grounded Response Agent ─────────────────────
def response_agent(state: ClinicalQueryState) -> ClinicalQueryState:
    print(f"🤖 [Response Agent] Generating model inference answer...")

    if not state["retrieved_chunks"]:
        return {**state, "response": "Verification error: Empty isolated data context returned."}

    context_parts = [f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(state["retrieved_chunks"])]
    context = "\n\n".join(context_parts)

    prompt = f"""You are a clinical document assistant analyzing verified FDA drug label information.

Answer the query using ONLY the provided metadata context. 
If the explicit facts are missing or unverified inside the context block, clearly state that you cannot answer.
Always explicitly cite the specific product name and your contextual sources ([Source 1], etc.) in your writeup.

Context:
{context}

Question: {state['query']}

Answer:"""

    try:
        claude_response = claude_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=600,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return {**state, "response": claude_response.content[0].text}
    except Exception as e:
        return {**state, "response": f"Downstream Generation Failure: {e}"}

# ── Node 5: Human Review Agent ─────────────────────────
def human_review_agent(state: ClinicalQueryState) -> ClinicalQueryState:
    print(f"⚠️ [Human Review Node] Flagging trace logs for escrow...")
    
    review_message = (
        f"⚠️ CLINICAL COMPLIANCE GUARD FLAGGED THIS QUERY\n"
        f"User Query: '{state['query']}'\n"
        f"Calculated Score: {state['confidence_score']:.4f} (Below Safety Threshold: {CONFIDENCE_THRESHOLD})\n"
        f"System Message: Semantic lookup failed to locate authoritative chunks inside the specified drug domain.\n"
        f"Action Requirement: Manual review of root regulatory documentation recommended."
    )
    return {**state, "response": review_message}

# ── Node 6: 21 CFR Part 11 Audit Trail Logger ─────────
def audit_log(state: ClinicalQueryState) -> ClinicalQueryState:
    """
    Maintains an append-only, trace-tracked transaction ledger for GxP validation.
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": state["query"],
        "detected_drugs": state.get("detected_drugs", []),
        "section_filter": state.get("section_filter", ""),
        "confidence_score": state.get("confidence_score", 0.0),
        "requires_human_review": state.get("requires_human_review", False),
        "retrieval_scores": state.get("retrieval_scores", []),
        "chunks_retrieved": len(state.get("retrieved_chunks", [])),
        "response_preview": state.get("response", "")[:180],
        "error": state.get("error", "")
    }

    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"📝 [Audit Log] Transaction sealed under timestamp hash: {log_entry['timestamp']}")
    return state

# ── Orchestration Compiler ─────────────────────────────
def build_pipeline():
    workflow = StateGraph(ClinicalQueryState)

    # Register Nodes
    workflow.add_node("intent_router", intent_router_agent)
    workflow.add_node("retrieval", retrieval_agent)
    workflow.add_node("confidence_check", confidence_check)
    workflow.add_node("response", response_agent)
    workflow.add_node("human_review", human_review_agent)
    workflow.add_node("audit", audit_log)

    # Direct Progression Map
    workflow.set_entry_point("intent_router")
    workflow.add_edge("intent_router", "retrieval")
    workflow.add_edge("retrieval", "confidence_check")

    # Conditional Routing Gate
    workflow.add_conditional_edges(
        "confidence_check",
        route_by_confidence,
        {
            "response": "response",
            "human_review": "human_review"
        }
    )

    # Convergence Links to Audit Step
    workflow.add_edge("response", "audit")
    workflow.add_edge("human_review", "audit")
    workflow.add_edge("audit", END)

    return workflow.compile()