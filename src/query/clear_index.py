import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "clinicalmind")
index = pc.Index(index_name)

print(f"🧹 Initiating complete purge of Pinecone index: '{index_name}'...")

# Delete all vectors in the index across all namespaces
index.delete(delete_all=True)

print("✅ Index successfully wiped clean.")