from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# Load model and index
model = SentenceTransformer('sentence-transformers/LaBSE')
index = faiss.read_index("data/kb.index")
with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

def retrieve(query, k=3):
    query_emb = model.encode([query])
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk": chunks[idx]["text"],
            "score": float(scores[0][i]),
            "source": chunks[idx]["source"]
        })
    return results

if __name__ == "__main__":
    query = input("Enter a Telugu claim: ")
    results = retrieve(query)
    print("\nTop relevant chunks:")
    for r in results:
        print(f"Score: {r['score']:.4f}")
        print(f"Source: {r['source']}")
        print(f"Text: {r['chunk'][:200]}...")
        print("-" * 50)