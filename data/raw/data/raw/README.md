# Automated Fact-Checker for Telugu News

A RAG pipeline that fact-checks Telugu news/social posts using a verified knowledge base.

## Optimizations
- Quantized translation model (NLLB)
- Caching for repeated queries
- Fluff stripping to focus on claims
- Batch translation for speed
- Conflict detection

## Performance
- Single claim: 2.35 seconds
- Batch: 126 claims/minute on CPU

## How to Run
```bash
pip install -r requirements.txt
python factcheck.py
   
