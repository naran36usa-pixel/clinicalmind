from dotenv import load_dotenv
import fitz
import json

load_dotenv()

print("Loading document...")
doc = fitz.open("data/lecanemab_label.pdf")
full_text = ""
for page in doc:
    full_text += page.get_text()
print(f"Total characters: {len(full_text)}")

# Merge lines into chunks
lines = full_text.split("\n")
chunks_raw = []
current_chunk = ""

for line in lines:
    line = line.strip()
    if not line:
        if len(current_chunk.strip()) > 50:
            chunks_raw.append(current_chunk.strip())
        current_chunk = ""
    else:
        current_chunk += " " + line

if len(current_chunk.strip()) > 50:
    chunks_raw.append(current_chunk.strip())

print(f"Raw chunks: {len(chunks_raw)}")

# Noise filter — relaxed
def is_noise(chunk):
    words = chunk.split()
    if len(words) < 8:
        return True

    # Contains chart symbol
    if '■' in chunk or '▲' in chunk:
        return True

    # High density of single quotes = garbled figure
    single_quotes = chunk.count("'")
    if single_quotes > 10:
        return True

    # High ratio of single character words
    single = sum(1 for w in words if len(w) == 1)
    if len(words) > 15 and single / len(words) > 0.4:
        return True

    return False

clean_chunks = [c for c in chunks_raw if not is_noise(c)]

print(f"Noise removed: {len(chunks_raw) - len(clean_chunks)}")
print(f"Clean chunks: {len(clean_chunks)}")

# Stats
avg = sum(len(c) for c in clean_chunks) // len(clean_chunks)
print(f"Average length: {avg} chars")

# Samples
print(f"\nChunk 1:\n{clean_chunks[0][:300]}")
print(f"\nChunk 10:\n{clean_chunks[9][:300]}")
print(f"\nChunk 20:\n{clean_chunks[19][:300]}")

# Save
preview = [{"chunk_id": i, "preview": c[:150],
            "chars": len(c)}
           for i, c in enumerate(clean_chunks[:25])]

with open("docs/chunks_preview.json", "w",
          encoding="utf-8") as f:
    json.dump(preview, f, indent=2)

with open("docs/chunks_full.json", "w",
          encoding="utf-8") as f:
    json.dump(clean_chunks, f, indent=2)

print("\n✅ Script 02 Complete")