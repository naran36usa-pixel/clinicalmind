from dotenv import load_dotenv
import os
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

# ── Initialize ─────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("clinicalmind")
openai_client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

import anthropic
claude_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# ── Query Function ─────────────────────────────────────
def query_clinical_docs(question: str,
                        top_k: int = 3) -> dict:
    """
    Takes a clinical question.
    Returns answer with sources.
    """
    print(f"\nQuery: {question}")
    print("-" * 50)

    # Step 1 — Embed the question
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    query_embedding = response.data[0].embedding

    # Step 2 — Search Pinecone
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Step 3 — Build context from results
    context_chunks = []
    sources = []

    for i, match in enumerate(results.matches):
        text = match.metadata.get("text", "")
        score = match.score
        context_chunks.append(f"[Source {i+1}]: {text}")
        sources.append({
            "source_id": i + 1,
            "chunk_id": match.metadata.get("chunk_id"),
            "relevance_score": round(score, 4),
            "text_preview": text[:150]
        })

    context = "\n\n".join(context_chunks)

    # Step 4 — Ask Claude
    prompt = f"""You are a clinical document assistant 
analyzing FDA drug label information.

Answer the question using ONLY the provided context.
If the answer is not in the context, say so clearly.
Always cite which source you used.

Context:
{context}

Question: {question}

Answer:"""

    claude_response = claude_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = claude_response.content[0].text

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }


# ── Run Test Queries ───────────────────────────────────
test_questions = [
    "What are the indications for Leqembi?",
    "What are the contraindications?",
    "What are the warnings and precautions?"
]

for question in test_questions:
    result = query_clinical_docs(question)

    print(f"Answer:\n{result['answer']}")
    print(f"\nSources used:")
    for source in result["sources"]:
        print(f"  Source {source['source_id']}: "
              f"score={source['relevance_score']} "
              f"| {source['text_preview'][:100]}")
    print("\n" + "="*60)

print("✅ Script 05 Complete — Phase 1 Done")