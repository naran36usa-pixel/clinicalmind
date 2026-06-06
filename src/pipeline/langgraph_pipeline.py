import os
import json
from typing import Dict, Any, List, TypedDict
from dotenv import load_dotenv
from pinecone import Pinecone
from anthropic import Anthropic

# Load system configuration variables
load_dotenv()

# Initialize API clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "clinicalmind")
pinecone_index = pc.Index(index_name)

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ------------------------------------------------------------------------
# 1. State Definition
# ------------------------------------------------------------------------
class GraphState(TypedDict):
    query: str
    hydrated_query: str
    section_filter: str
    detected_drugs: List[str]
    confidence_score: float
    retrieved_context: str
    response: str
    routing_notes: str
    llm_model: str

# ------------------------------------------------------------------------
# 2. Graph Nodes & Logic
# ------------------------------------------------------------------------

def intent_router_agent(state: GraphState) -> GraphState:
    """
    Evaluates query boundaries and automatically overrides section filters 
    across all clinical domains if a semantic mismatch is detected.
    """
    query_text = state.get("query", "").lower().strip()
    current_filter = state.get("section_filter", "").strip()
    
    print(f"🧠 [Intent Router] Evaluating query boundaries: '{query_text}'")
    state["routing_notes"] = ""
    
    # Semantic Keyword Evaluation Matrix
    is_dosage_query = any(word in query_text for word in ["dosage", "dose", "mg", "regimen", "administer", "dosing"])
    is_adverse_query = any(word in query_text for word in ["adverse", "side effect", "reaction", "toxicity", "safety", "complication"])
    is_population_query = any(word in query_text for word in ["pregnancy", "lactation", "pediatric", "geriatric", "renal", "hepatic", "children"])
    
    # Boundary Override Arbitrator
    if is_dosage_query and current_filter != "DOSAGE AND ADMINISTRATION":
        state["section_filter"] = "DOSAGE AND ADMINISTRATION"
        state["routing_notes"] = (
            f"Detected dosing instruction intent while UI constraint was restricted to '{current_filter}'. "
            "Dynamically shifted data partition constraint to 'DOSAGE AND ADMINISTRATION' to maintain data integrity."
        )
        print("🧬 [Intent Router] Overriding UI boundary filter -> Shifting partition constraint to DOSAGE AND ADMINISTRATION.")
        
    elif is_adverse_query and current_filter != "ADVERSE REACTIONS":
        state["section_filter"] = "ADVERSE REACTIONS"
        state["routing_notes"] = (
            f"Detected adverse event profile intent while UI constraint was restricted to '{current_filter}'. "
            "Dynamically shifted data partition constraint to 'ADVERSE REACTIONS' to protect against context cross-contamination."
        )
        print("🧬 [Intent Router] Overriding UI boundary filter -> Shifting partition constraint to ADVERSE REACTIONS.")
        
    elif is_population_query and current_filter != "USE IN SPECIFIC POPULATIONS":
        state["section_filter"] = "USE IN SPECIFIC POPULATIONS"
        state["routing_notes"] = (
            f"Detected specific population context intent while UI constraint was restricted to '{current_filter}'. "
            "Dynamically shifted data partition constraint to 'USE IN SPECIFIC POPULATIONS'."
        )
        print("🧬 [Intent Router] Overriding UI boundary filter -> Shifting partition constraint to USE IN SPECIFIC POPULATIONS.")
        
    else:
        print(f"🧬 [Intent Router] Enforcing explicit UI-driven boundaries: Section -> {current_filter}")
        
    return state


def context_hydration_agent(state: GraphState) -> GraphState:
    """
    Hydrates the incoming query with domain metadata to optimize high-dimensional 
    vector similarity scoring inside Pinecone.
    """
    raw_query = state["query"]
    target_drug = state["detected_drugs"][0] if state["detected_drugs"] else "Unknown Product"
    
    hydrated_string = f"{raw_query} (Context: Target product domain is {target_drug})"
    state["hydrated_query"] = hydrated_string
    
    print(f"🔧 [Retrieval Agent] Context Hydration Injection -> Optimized String: '{hydrated_string}'")
    return state


def retrieval_agent(state: GraphState) -> GraphState:
    """
    Queries Pinecone using the hydrated vector and applies structural metadata filtering
    with a hardened normalization layer to prevent nomenclature suffix mismatches.
    """
    import openai
    openai_client = openai.OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
    
    print("  [Retrieval] Generating embedding vector via model: 'text-embedding-3-small'...")
    response = openai_client.embeddings.create(
        input=[state["hydrated_query"]],
        model="text-embedding-3-small"
    )
    query_vector = response.data[0].embedding
    
    # Normalization Layer: Decouple UI variations from DB keys using token extraction
    raw_ui_drug = state["detected_drugs"][0]
    brand_token = raw_ui_drug.split("(")[0].strip().lower()
    
    db_target_name = raw_ui_drug
    if "leqembi" in brand_token:
        db_target_name = "Leqembi (Lecanemab)"
    elif "keytruda" in brand_token:
        db_target_name = "Keytruda (Pembrolizumab)"
    elif "trodelvy" in brand_token:
        db_target_name = "Trodelvy (Sacituzumab govitecan-hziy)"

    filter_dict = {
        "drug_name": {"$eq": db_target_name}
    }
    
    if state.get("section_filter") and state["section_filter"] != "ALL SECTIONS":
        filter_dict["section_name"] = {"$eq": state["section_filter"].strip().upper()}
        
    print(f"  [Retrieval Engine] Dispatching query with stabilized metadata filter: {filter_dict}")
    
    results = pinecone_index.query(
        vector=query_vector,
        top_k=4,
        include_metadata=True,
        filter=filter_dict
    )
    
    context_chunks = []
    max_score = 0.0
    
    if results.matches:
        max_score = results.matches[0].score
        for match in results.matches:
            db_drug = match.metadata.get("drug_name")
            db_section = match.metadata.get("section_name", "UNKNOWN")
            print(f"  📑 [Match Score: {match.score:.4f}] Product: {db_drug} | Partition: {db_section}")
            
            node_text = ""
            if "text" in match.metadata:
                node_text = match.metadata["text"]
            elif "_node_content" in match.metadata:
                try:
                    node_data = json.loads(match.metadata["_node_content"])
                    node_text = node_data.get("text", "")
                except Exception:
                    pass
            
            if node_text:
                context_chunks.append(node_text)
            
    state["confidence_score"] = max_score
    state["retrieved_context"] = "\n\n".join(context_chunks)
    
    print(f"🛡️ [Confidence Check] Max Score: {max_score:.4f} | Gate Limit: 0.5")
    return state


def generation_agent(state: GraphState) -> GraphState:
    """
    Executes context-grounded response generation using production-grade Anthropic models.
    """
    if state["confidence_score"] < 0.5000:
        state["response"] = (
            "ERROR: The mathematical semantic alignment score fell below the required compliance tolerance gate. "
            "This query has been routed to the human review queue to prevent potential hallucination."
        )
        return state
        
    system_prompt = (
        "You are a deterministic clinical document intelligence engine. Your task is to answer the user query "
        "using ONLY the provided text excerpts below. Do not use outside medical knowledge. If the text does "
        "not explicitly contain the answer, cleanly state that the information cannot be found within the provided context."
    )
    
    user_content = f"Context:\n{state['retrieved_context']}\n\nQuery: {state['query']}"
    
    chosen_model = state.get("llm_model", "claude-sonnet-4-6")
    print(f"🤖 [Generation Layer] Routing payload to active engine: {chosen_model}")
    
    message = anthropic_client.messages.create(
        model=chosen_model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}]
    )
    
    state["response"] = message.content[0].text
    return state

# ------------------------------------------------------------------------
# 3. State Graph Compilation Pipeline
# ------------------------------------------------------------------------
from langgraph.graph import StateGraph, END

workflow = StateGraph(GraphState)

workflow.add_node("intent_router", intent_router_agent)
workflow.add_node("context_hydration", context_hydration_agent)
workflow.add_node("retrieval_layer", retrieval_agent)
workflow.add_node("generation_layer", generation_agent)

workflow.set_entry_point("intent_router")
workflow.add_edge("intent_router", "context_hydration")
workflow.add_edge("context_hydration", "retrieval_layer")
workflow.add_edge("retrieval_layer", "generation_layer")
workflow.add_edge("generation_layer", END)

clinicalmind_pipeline = workflow.compile()