from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Load model
model = SentenceTransformer('sentence-transformers/LaBSE')

# Load chunks
with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Extract texts
texts = [chunk["text"] for chunk in chunks]

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save index and chunks
faiss.write_index(index, "data/kb.index")
with open("data/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("Index built and saved.")