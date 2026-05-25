# Final Report Outline

## 1. Introduction
- Turkish legal QA problem
- Why grounded answers and citation consistency matter
- Project objective

## 2. Dataset
- Corpus: 25 Turkish laws, article-level chunks
- Benchmark: 290 verified QA pairs
- Dataset files are used as read-only inputs

## 3. Methodology
### 3.1 Baseline RAG
- Embedding model
- FAISS vector search
- Prompt structure
- Same LLM across experiments

### 3.2 Hybrid Retrieval
- Dense retrieval + BM25
- Score normalization
- Weighted combination

### 3.3 Reranker
- Cross-encoder reranking setup
- Top-k candidate reranking

### 3.4 Fine-tuned Component
- Fine-tuned embedding or reranker
- Training data construction
- Hard negative mining if used

## 4. Evaluation
### 4.1 Retrieval Metrics
- Recall@5
- Recall@10
- MRR
- nDCG@10

### 4.2 Answer Metrics
- EM/F1 or LLM judge
- Semantic similarity

### 4.3 Grounding Metrics
- Faithfulness
- Citation accuracy
- Hallucination analysis

## 5. Experiments and Results
- Base RAG
- Hybrid RAG
- Reranked RAG
- Fine-tuned RAG
- Fully optimized system

## 6. Ablation Study
- Contribution of embedding
- Contribution of BM25 hybridization
- Contribution of reranker
- Contribution of fine-tuned component

## 7. Error Analysis
- Retrieval misses
- Correct retrieval but wrong answer
- Hallucinated answers
- Citation mismatch

## 8. Conclusion
- Main findings
- Limitations
- Future work
