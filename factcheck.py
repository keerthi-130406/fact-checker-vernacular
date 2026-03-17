from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
torch.manual_seed(42)  # Ensures reproducible results
from torch.quantization import quantize_dynamic
import re
def extract_claim(post):
    # Remove URLs, hashtags, mentions
    post = re.sub(r'http\S+', '', post)
    post = re.sub(r'@\w+', '', post)
    post = re.sub(r'#\w+', '', post)
    # Remove common fluff phrases
    fluff = ["BREAKING:", "NEWS:", "OMG", "Just in:"]
    for f in fluff:
        post = post.replace(f, '')
    # Take first sentence (split by . ! ?)
    sentences = re.split(r'[.!?]', post)
    return sentences[0].strip() if sentences else post
def has_conflict(evidence_texts):
    """Return True if evidence contains both supportive and contradictory signals."""
    pos = False
    neg = False
    for text in evidence_texts:
        text_lower = text.lower()
        if "true" in text_lower:
            pos = True
        if "false" in text_lower or "misleading" in text_lower:
            neg = True
    return pos and neg

# Load retrieval components
embed_model = SentenceTransformer('sentence-transformers/LaBSE')
index = faiss.read_index("data/kb.index")
with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load translation model (NLLB for Telugu->English)
print("Loading translation model (NLLB)...")
model_name = "facebook/nllb-200-distilled-600M"
trans_tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="tel_Telu")
trans_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# Quantize translation model
trans_model = quantize_dynamic(
    trans_model, {torch.nn.Linear}, dtype=torch.qint8
)
print("Translation model quantized.")

# Load LLM for fact-checking (Flan-T5-base)
print("Loading fact-check LLM...")
llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
# Quantize fact-check LLM
# llm_model = quantize_dynamic( llm_model, {torch.nn.Linear}, dtype=torch.qint8)
# print("Fact-check LLM quantized.")
# Cache for fact-check results
cache = {}


def translate_te_to_en(text):
    inputs = trans_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # Force target language to English
    forced_bos_token_id = trans_tokenizer.convert_tokens_to_ids("eng_Latn")
    outputs = trans_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=200
    )
    return trans_tokenizer.decode(outputs[0], skip_special_tokens=True)

def retrieve(query, k=3):
    query_emb = embed_model.encode([query])
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

def generate_verdict(claim_en, evidence_texts):
    evidence = "\n\n".join(evidence_texts)
    prompt = f"""You are a strict fact-checker. Your task is to determine if the claim is True, False, or Misleading based **only** on the evidence. The evidence contains a fact-check analysis with a verdict (like "తప్పుదారి పట్టించే వాదన" or "False"). Use that to decide.

Rules:
- If the evidence directly supports the claim, answer "True".
- If the evidence directly contradicts the claim (e.g., says it's false or misleading), answer "False".
- If the evidence is mixed, insufficient, or the claim is only partially true, answer "Misleading".

Output only the verdict followed by a short explanation.

Claim: {claim_en}

Evidence:
{evidence}

Verdict (True/False/Misleading):"""
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = llm_model.generate(**inputs, max_new_tokens=100)
    return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

def fact_check(claim_te):
    # Check cache first
    if claim_te in cache:
        print("✔ Returning cached result")
        return cache[claim_te]
    
    # Translate claim to English
    claim_en = translate_te_to_en(claim_te)
    print(f"Translated claim: {claim_en}")
    
    # Retrieve relevant chunks
    retrieved = retrieve(claim_te)
    evidence_texts = [r["chunk"] for r in retrieved]
    
    # Generate verdict
    verdict = generate_verdict(claim_en, evidence_texts)
    
    # Store in cache
    cache[claim_te] = (verdict, retrieved)
    return verdict, retrieved

def translate_batch(texts):
    """Translate a list of Telugu texts to English using the NLLB model."""
    # Tokenize with padding to create a batch
    inputs = trans_tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    forced_bos_token_id = trans_tokenizer.convert_tokens_to_ids("eng_Latn")
    outputs = trans_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=200
    )
    return trans_tokenizer.batch_decode(outputs, skip_special_tokens=True)


def fact_check_batch(claims_list):
    results = []
    # Step 1: strip fluff from all claims
    cleaned = [extract_claim(c) for c in claims_list]
    
    # Step 2: translate all claims in one batch
    translated = translate_batch(cleaned)
    
    # Step 3: encode all queries at once for retrieval
    all_embs = embed_model.encode(cleaned)
    faiss.normalize_L2(all_embs)
    
    # Step 4: retrieve and check conflict for each query
    for i, query_emb in enumerate(all_embs):
        scores, indices = index.search(query_emb.reshape(1, -1), k=3)
        retrieved_chunks = [chunks[idx] for idx in indices[0]]
        evidence_texts = [r["text"] for r in retrieved_chunks]
        
        if has_conflict(evidence_texts):
            verdict = "Conflicting evidence – manual review"
        else:
            verdict = generate_verdict(translated[i], evidence_texts)
        results.append((verdict, retrieved_chunks))
    
    return results
if __name__ == "__main__":
    import time
    # Create a list of 10 identical claims (you can also read from a file)
    test_claims = ["చైనాకు చెందిన మహిళా SWAT జట్టు భారత పురుష జట్టును ఓడించింది?"] * 10
    
    start = time.time()
    results = fact_check_batch(test_claims)
    elapsed = time.time() - start
    
    print(f"Processed {len(test_claims)} claims in {elapsed:.2f} seconds")
    print(f"Throughput: {len(test_claims)/elapsed:.2f} claims/sec")
    print(f"Equivalent per minute: {len(test_claims)/elapsed * 60:.0f} claims/min")