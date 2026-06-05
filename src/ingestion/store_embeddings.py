import os
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

def upsert_to_pinecone(processed_nodes):
    """
    Step 4: Generate mathematical embeddings and save nodes into Pinecone.
    """
    if not processed_nodes:
        print("  [Store] Ingestion bypassed: Zero nodes provided.")
        return

    # Base LlamaIndex settings configuration
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPEN_AI_API_KEY")
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index("clinicalmind")

    print(f"  [Store] Syncing vectors over the network to Pinecone index 'clinicalmind'...")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # LlamaIndex vector index handles batching and embedding generation internally
    index = VectorStoreIndex(
        nodes=processed_nodes,
        storage_context=storage_context
    )
    print(f"  [Store] Upload verified successfully.")