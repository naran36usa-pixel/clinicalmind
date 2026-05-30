from dotenv import load_dotenv
import os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    ExactMatchFilter
)
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from pinecone import Pinecone

load_dotenv("clinical.env")

# ── Initialize ─────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("clinicalmind")

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPEN_AI_API_KEY")
)
Settings.llm = Anthropic(
    model="claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1000
)

# ── Load Index ─────────────────────────────────────────
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index
)
index = VectorStoreIndex.from_vector_store(vector_store)

# ── Test Queries ───────────────────────────────────────
# Update section_name value based on what
# your PDF actually detected in Script 06

test_queries = [
    {
        "question": "Who is the target patient population?",
        "section": "INDICATIONS AND USAGE"
    },
    {
        "question": "What are the contraindications?",
        "section": "CONTRAINDICATIONS"
    },
    {
        "question": "What are the warnings and precautions?",
        "section": "WARNINGS AND PRECAUTIONS"
    }
]

for test in test_queries:
    print(f"\n{'='*50}")
    print(f"Query: {test['question']}")
    print(f"Filtering to section: {test['section']}")
    print('='*50)

    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="section_name",
                value=test["section"]
            )
        ]
    )

    query_engine = index.as_query_engine(
        similarity_top_k=3,
        filters=filters
    )

    try:
        response = query_engine.query(test["question"])
        print(f"\nResponse:\n{response}")
        print(f"\nSource section: "
              f"{response.source_nodes[0].node.metadata['section_name']}")
    except Exception as e:
        print(f"Query failed: {e}")
        print("Check section name matches exactly what was detected")

print("\n✅ Week 2 Query Testing Complete")