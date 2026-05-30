from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "clinicalmind"
EMBEDDING_DIMENSION = 1536

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created index: {INDEX_NAME}")
else:
    print(f"Index already exists: {INDEX_NAME}")

# Confirm
index = pc.Index(INDEX_NAME)
stats = index.describe_index_stats()
print(f"Index stats: {stats}")
print("✅ Script 03 Complete")