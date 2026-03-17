import os
import json

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

raw_dir = "data/raw"
chunks = []

for filename in os.listdir(raw_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(raw_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        text_chunks = chunk_text(text)
        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "chunk_id": f"{filename}_{i}",
                "text": chunk,
                "source": filename,
                "title": filename.replace(".txt", "")
            })

# Save chunks to a JSON file
with open("data/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"Created {len(chunks)} chunks from {len(os.listdir(raw_dir))} files.")