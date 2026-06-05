import os
from dotenv import load_dotenv

# Absolute path environment variable injection
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(dotenv_path=os.path.join(base_dir, ".env"))

from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

# ── Global System Core Configurations ─────────────────
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPEN_AI_API_KEY")
)
Settings.llm = Anthropic(
    model="claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1000
)

# Initialize vector store interface
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("clinicalmind")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store)

# ── Core Executive Query API ───────────────────────────
def query_clinicalmind(question: str, drug_name: str, section_name: str = None, top_k: int = 3):
    """
    Executes a clinical query using strict multi-attribute enterprise metadata filtering.
    Guarantees isolation of drug contexts at the database vector-lookup layer.
    """
    print(f"\n🧠 PIPELINE RETRIEVAL: Targeting Context [Drug: {drug_name} | Section: {section_name or 'GLOBAL'}]")
    
    # 1. Dynamically assemble the database isolation filter checklist
    filter_list = [
        ExactMatchFilter(key="drug_name", value=drug_name)
    ]
    
    # Optional section slicing optimization
    if section_name:
        filter_list.append(ExactMatchFilter(key="section_name", value=section_name))
        
    compound_filters = MetadataFilters(filters=filter_list)

    # 2. Compile query engine with parameter boundaries
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        filters=compound_filters
    )

    try:
        # 3. Request downstream model inference over isolated context nodes
        response = query_engine.query(question)
        return response
    except Exception as e:
        print(f"❌ Core Retrieval Failure: {str(e)}")
        return None

# ── Automated Validation Suite ────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("🧪 CLINICALMIND METADATA RETRIEVAL INTEGRATION TESTS")
    print("="*60)

    # Definitive test cases tracking target execution
    cross_drug_matrix = [
        {
            "question": "What is the target patient population?",
            "drug": "Keytruda (Pembrolizumab)",
            "section": "INDICATIONS AND USAGE"
        },
        {
            "question": "What are the specific contraindications for this product?",
            "drug": "Leqembi (Lecanemab)",
            "section": "CONTRAINDICATIONS"
        },
        {
            "question": "What are the warnings and precautions regarding severe adverse reactions?",
            "drug": "Trodelvy (Sacituzumab govitecan-hziy)",
            "section": "WARNINGS AND PRECAUTIONS"
        }
    ]

    for run in cross_drug_matrix:
        res = query_clinicalmind(
            question=run["question"],
            drug_name=run["drug"],
            section_name=run["section"]
        )
        
        if res:
            print(f"\nResponse:\n{str(res).strip()}")
            print("-" * 40)
            print("Verified Source Provenance Nodes:")
            for source in res.source_nodes:
                meta = source.node.metadata
                print(f" 📑 [Score: {round(source.score, 4)}] -> Drug: {meta['drug_name']} | Sec: {meta['section_name']} | Para Index: {meta['paragraph_index']}")
        print("="*60)