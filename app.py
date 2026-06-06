import streamlit as st
import os
import glob
import json
from src.pipeline.langgraph_pipeline import clinicalmind_pipeline

# Configure application shell metadata
st.set_page_config(
    page_title="ClinicalMind — Governed RAG Framework",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_ingested_drugs():
    """Dynamically parses local config profiles to populate UI options automatically."""
    base_dir = os.path.dirname(__file__)
    config_pattern = os.path.join(base_dir, "data", "configs", "*_config.json")
    drug_list = []
    
    for path in glob.glob(config_pattern):
        if "default" in path:
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if data.get("drug_name"):
                    drug_list.append(data["drug_name"])
        except Exception:
            pass
            
    if not drug_list:
        # Structured fallback if local directories vary in target environment
        return ["Keytruda (Pembrolizumab)", "Leqembi (Lecanemab-irmb)", "Trodelvy (Sacituzumab govitecan-hziy)"]
    return sorted(list(set(drug_list)))

# ------------------------------------------------------------------------
# Sidebar Query Control Interface
# ------------------------------------------------------------------------
with st.sidebar:
    st.header("Query Configuration")
    
    drug_options = get_ingested_drugs()
    selected_drug = st.selectbox("Select Drug Target", drug_options)
    
    selected_section = st.selectbox(
        "Select Section Layer",
        [
            "ADVERSE REACTIONS", 
            "DOSAGE AND ADMINISTRATION", 
            "USE IN SPECIFIC POPULATIONS",
            "INDICATIONS AND USAGE",
            "ALL SECTIONS"
        ]
    )
    
    st.markdown("---")
    st.subheader("Model Infrastructure")
    
    model_choice = st.selectbox(
        "Select LLM Engine Tier",
        options=["Claude Sonnet 4.6 (Balanced Speed/Intelligence)", "Claude Opus 4.8 (Highest Reasoning Tier)"]
    )
    
    model_map = {
        "Claude Sonnet 4.6 (Balanced Speed/Intelligence)": "claude-sonnet-4-6",
        "Claude Opus 4.8 (Highest Reasoning Tier)": "claude-opus-4-8"
    }
    selected_model_id = model_map[model_choice]
    
    st.markdown("---")
    st.caption("**Compliance Gate Threshold:** 0.5")

# ------------------------------------------------------------------------
# Main Application Workspace Canvas
# ------------------------------------------------------------------------
st.title("🧬 ClinicalMind")
st.caption("Governed Clinical Document Intelligence — FDA Drug Label RAG Pipeline")
st.markdown("---")

clinical_query = st.text_area(
    "Enter your clinical query:", 
    placeholder="e.g., what are the most common adverse reactions reported during clinical trials?"
)

if st.button("Run Query Analysis Pipeline", type="primary"):
    if not clinical_query.strip():
        st.warning("Please specify a valid query sequence before running verification steps.")
    else:
        with st.spinner("Executing multi-agent state execution graph..."):
            try:
                initial_state = {
                    "query": clinical_query,
                    "hydrated_query": "",
                    "section_filter": selected_section,
                    "detected_drugs": [selected_drug],
                    "confidence_score": 0.0,
                    "retrieved_context": "",
                    "response": "",
                    "routing_notes": "",
                    "llm_model": selected_model_id
                }
                
                output_state = clinicalmind_pipeline.invoke(initial_state)
                
                if output_state.get("routing_notes"):
                    st.info(f"💡 **Boundary Optimization Node Rule:** {output_state['routing_notes']}")
                
                st.subheader("Verified Clinical Response")
                response_text = output_state.get("response", "")
                
                if "ERROR:" in response_text or "fell below the required compliance tolerance gate" in response_text:
                    st.error(response_text)
                else:
                    st.write(response_text)
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    score = output_state.get("confidence_score", 0.0)
                    if score < 0.5:
                        st.metric(label="Confidence Score", value=f"{score:.4f}", delta=f"{score - 0.5:.4f}", delta_color="inverse")
                    else:
                        st.metric(label="Confidence Score", value=f"{score:.4f}", delta=f"{score - 0.5:.4f}")
                with col2:
                    st.metric(label="Target Domain", value=output_state.get("detected_drugs")[0])
                with col3:
                    st.metric(label="Label Partition", value=output_state.get("section_filter"))
                
                st.markdown("---")
                
                with st.expander("🔍 Sub-System Retrieval Metrics"):
                    st.json({
                        "raw_query": output_state.get("query"),
                        "hydrated_query": output_state.get("hydrated_query"),
                        "active_llm_engine": selected_model_id,
                        "applied_filters": {
                            "drug_name": output_state.get("detected_drugs")[0],
                            "section_name": output_state.get("section_filter")
                        }
                    })
                    
            except Exception as e:
                st.error(f"Pipeline Execution Interrupted: {str(e)}")