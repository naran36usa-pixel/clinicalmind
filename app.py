import streamlit as st
import sys
from datetime import datetime, timezone

# Ensure the app can locate modules in the src directory
sys.path.append("src")

from pipeline.langgraph_pipeline import (
    build_pipeline,
    ClinicalQueryState
)

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="ClinicalMind",
    page_icon="🧬",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────
st.title("🧬 ClinicalMind")
st.caption(
    "Governed Clinical Document Intelligence — "
    "FDA Drug Label RAG Pipeline"
)
st.divider()

# ── Sidebar — Drug Selection ───────────────────────────
st.sidebar.header("Query Configuration")

drug_options = {
    "Leqembi (Lecanemab)": "Leqembi (Lecanemab)",
    "Trodelvy (Sacituzumab govitecan-hziy)": 
        "Trodelvy (Sacituzumab govitecan-hziy)",
    "Keytruda (Pembrolizumab)": 
        "Keytruda (Pembrolizumab)"
}

selected_drug = st.sidebar.selectbox(
    "Select Drug",
    options=list(drug_options.keys())
)

section_options = [
    "INDICATIONS AND USAGE",
    "CONTRAINDICATIONS",
    "WARNINGS AND PRECAUTIONS",
    "ADVERSE REACTIONS",
    "DOSAGE AND ADMINISTRATION",
    "USE IN SPECIFIC POPULATIONS",
    "CLINICAL STUDIES"
]

selected_section = st.sidebar.selectbox(
    "Select Section",
    options=section_options
)

st.sidebar.divider()
st.sidebar.caption(
    "Confidence threshold: 0.5\n\n"
    "Queries below threshold route to "
    "human review automatically."
)

# ── Main Query Interface ───────────────────────────────
query = st.text_area(
    "Enter your clinical query:",
    placeholder=(
        "e.g. What are the contraindications "
        "for this drug?"
    ),
    height=100
)

col1, col2 = st.columns([1, 4])
with col1:
    run_button = st.button(
        "🔍 Run Query",
        type="primary",
        use_container_width=True
    )

# ── Run Pipeline ───────────────────────────────────────
if run_button and query:

    with st.spinner("Running ClinicalMind pipeline..."):
        pipeline = build_pipeline()

        # Instantiate state with UI parameter overrides mapped correctly
        initial_state = ClinicalQueryState(
            query=query,
            detected_drugs=[selected_drug],    
            section_filter=selected_section,   
            retrieved_chunks=[],
            retrieval_scores=[],
            confidence_score=0.0,
            requires_human_review=False,
            response="",
            error=""
        )

        # Invoke the state machine workflow
        result = pipeline.invoke(initial_state)

    # ── Results Display ────────────────────────────────
    st.divider()

    # Extract state variables safely using safe defaults
    confidence = result.get("confidence_score", 0.0)
    review_required = result.get("requires_human_review", False)
    final_response = result.get("response", "")

    # Topline Metric Ribbon
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Confidence Score",
            f"{confidence:.4f}"
        )

    with col2:
        st.metric(
            "Target Domain",
            selected_drug.split("(")[0].strip()
        )

    with col3:
        display_section = selected_section.replace(" AND ", " & ")
        st.metric(
            "Label Partition",
            display_section[:22] if len(display_section) > 22 else display_section
        )

    st.divider()

    # ── Smart Routing Display Logic ────────────────────
    # Escrow state triggers if confidence is low or if the guard output string is flagged
    if review_required or "CLINICAL COMPLIANCE GUARD FLAGGED" in final_response:
        st.error("🛑 GxP Safety Isolation Triggered")
        st.warning(
            f"**Manual Escalation Required**\n\n"
            f"The retrieval confidence score ({confidence:.4f}) fell below the mandatory "
            f"regulatory threshold of **0.5000**.\n\n"
            f"To prevent hallucinated data contamination, downstream model inference has been blocked."
        )
        with st.expander("⚠️ View Compliance Guard System Message", expanded=True):
            st.code(final_response, language="text")
    else:
        st.success("✅ Verification Complete — Context Grounded")
        st.markdown("### Verified Clinical Response")
        st.markdown(final_response)

    # ── Diagnostics & Audit Logs ───────────────────────
    with st.expander("🔍 Sub-System Retrieval Metrics"):
        st.write(f"**Individual Match Scores:** {result.get('retrieval_scores', [])}")
        st.write(f"**Isolated Chunks Parsed:** {len(result.get('retrieved_chunks', []))}")
        st.write(f"**Pipeline Error Exception State:** '{result.get('error', '')}'")

    with st.expander("📋 21 CFR Part 11 Immutable Audit Token"):
        st.json({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_query": query,
            "governed_boundaries": {
                "drug": selected_drug,
                "section": selected_section
            },
            "runtime_metrics": {
                "confidence_score": confidence,
                "human_review_escalated": review_required,
                "vector_match_count": len(result.get("retrieved_chunks", []))
            }
        })

elif run_button and not query:
    st.warning("Please enter a query first.")

# ── Footer ─────────────────────────────────────────────
st.divider()
st.caption(
    "ClinicalMind — Governed AI for regulated "
    "pharma environments | "
    "Audit trail active | "
    "21 CFR Part 11 inspired"
)