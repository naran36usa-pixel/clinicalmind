from dotenv import load_dotenv
import os
import json
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

# ── Initialize ─────────────────────────────────────────
print("Initializing connections...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("clinicalmind")
openai_client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

# ── Load Clean Chunks ──────────────────────────────────
print("Loading chunks...")
with open("docs/chunks_full.json", "r",
          encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Chunks to embed: {len(chunks)}")

# ── Generate Embeddings And Store ─────────────────────
print("Generating embeddings and storing in Pinecone...")

vectors = []
batch_size = 10

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]

    # Generate embeddings for batch
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=batch
    )

    # Build vectors for Pinecone
    for j, embedding_data in enumerate(response.data):
        chunk_id = i + j
        vectors.append({
            "id": f"chunk_{chunk_id}",
            "values": embedding_data.embedding,
            "metadata": {
                "text": batch[j][:500],
                "chunk_id": chunk_id,
                "drug_name": "Leqembi",
                "document_type": "FDA_LABEL"
            }
        })

    print(f"Processed chunks {i+1} to "
          f"{min(i+batch_size, len(chunks))}"
          f" of {len(chunks)}")

# ── Upsert To Pinecone ─────────────────────────────────
print("\nUploading to Pinecone...")
pinecone_index.upsert(vectors=vectors)

# ── Verify ─────────────────────────────────────────────
stats = pinecone_index.describe_index_stats()
print(f"\nPinecone stats after upload:")
print(f"Total vectors: {stats.total_vector_count}")
print("✅ Script 04 Complete")