# Turkish Legal RAG — Clean Baseline Project

Bu proje, Türk hukuku soru-cevapları için kaynak destekli bir RAG sistemi kurmak ve aynı benchmark üzerinde farklı retrieval/RAG varyantlarını ölçmek için hazırlanmıştır.

## Dataset politikası

`data/raw/` içindeki dosyalar **read-only kaynak** kabul edilir. Scriptler bu dosyaları değiştirmez. Üretilen index, tahmin ve metrik dosyaları yalnızca `outputs/` altına yazılır.

Gerekli raw dosyalar:

```text
data/raw/turk_rag_corpus.csv
data/raw/qa_benchmark_gold.csv
```

## Kurulum

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 0) Dataset'i sadece okuyarak kontrol et

```bash
python scripts/00_dataset_readonly_check.py
```

Bu komut dataset'i değiştirmez. Sadece satır/sütun özetini `outputs/metrics/dataset_readonly_summary.json` içine yazar.

## 1) Dense FAISS index oluştur

```bash
python scripts/01_build_dense_index.py
```

Varsayılan embedding modeli:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Daha güçlü ama daha ağır bir alternatif:

```bash
python scripts/01_build_dense_index.py --model BAAI/bge-m3
```

## 2) Retrieval evaluation çalıştır

Dense baseline:

```bash
python scripts/02_evaluate_retrieval.py --retriever dense
```

BM25 lexical baseline:

```bash
python scripts/02_evaluate_retrieval.py --retriever bm25
```

Hybrid retrieval:

```bash
python scripts/02_evaluate_retrieval.py --retriever hybrid --dense-weight 0.65 --bm25-weight 0.35
```

Çıktılar:

```text
outputs/metrics/retrieval_dense.json
outputs/metrics/retrieval_dense.csv
outputs/metrics/retrieval_bm25.json
outputs/metrics/retrieval_bm25.csv
outputs/metrics/retrieval_hybrid.json
outputs/metrics/retrieval_hybrid.csv
```

## 3) Tek soru dene

```bash
python scripts/03_create_benchmark_splits.py --question "Kişisel verilerin silinmesi ne zaman istenebilir?" --retriever hybrid
```

Bu komut top contextleri ve LLM'e gönderilecek prompt taslağını gösterir. Henüz API çağrısı yapmaz.

## Ölçülen retrieval metrikleri

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- MRR
- nDCG@10

Gold eşleşme mantığı dataset'i değiştirmeden yapılır: sistemin getirdiği `kaynak + madde_no`, benchmark satırındaki gold `kaynak + madde_no / madde_nolari_context` ile karşılaştırılır.

## Sonraki aşamalar

1. Base RAG cevap üretimi
2. LLM Judge ile answer quality / faithfulness ölçümü
3. Hybrid retrieval tuning
4. Reranker ablation
5. Fine-tuned embedding veya reranker ablation
6. Final rapor ve sunum tabloları

---

## Custom Dataset Usage

This project supports running the RAG pipeline on external/custom document collections.  
A custom dataset can be converted into the standard project corpus format and evaluated with the same retrieval and answer generation scripts.

### Supported Custom Dataset Formats

The system supports the following formats:

1. A standard corpus CSV file
2. A JSONL corpus file with optional benchmark JSON files
3. A folder of raw `.txt` or `.md` documents

---

### 1. Standard Corpus CSV Format

If the custom corpus is already prepared as a CSV file, it should contain the following columns:

```text
kaynak
madde_no
context_key
context
retrieval_text
chunk_strategy
kanun_no
url
```

The most important columns are:

* `context_key`: unique ID of each chunk
* `context`: original text chunk
* `retrieval_text`: text used by the retriever
* `kaynak`: source/document name
* `madde_no`: article or chunk identifier

The benchmark file should contain at least:

```text
row_id
soru
cevap
context_key
kaynak
madde_no
score_valid
```

If `context_key`, `kaynak`, and `madde_no` are provided, retrieval metrics such as Recall@k, MRR, and nDCG can be calculated. If only `soru` and `cevap` are provided, answer-level evaluation can still be performed, but retrieval metrics cannot be fully computed.

---

### 2. Converting an External JSONL Dataset

If the external corpus is provided as a JSONL file with fields such as `id`, `text`, `title`, and `metadata`, it can be converted into the standard project format.

Expected files:

```text
data/custom/teacher_dataset/raw/corpus.jsonl
data/custom/teacher_dataset/raw/rag_eval.json
```

Conversion command:

```bash
python scripts/17_convert_external_dataset.py \
  --corpus-jsonl data/custom/teacher_dataset/raw/corpus.jsonl \
  --rag-eval-json data/custom/teacher_dataset/raw/rag_eval.json \
  --output-dir outputs/custom/teacher_dataset
```

This creates:

```text
outputs/custom/teacher_dataset/corpus.csv
outputs/custom/teacher_dataset/benchmark_rag_eval.csv
```

---

### 3. Converting Raw Text Documents

If the external dataset is a folder of `.txt` or `.md` files, it can be converted into the standard corpus format.

Expected folder:

```text
data/custom/teacher_dataset/raw_docs/
```

Conversion command:

```bash
python scripts/14_prepare_custom_corpus.py \
  --input-dir data/custom/teacher_dataset/raw_docs \
  --output-csv outputs/custom/teacher_dataset/corpus.csv
```

---

### Retrieval Evaluation on a Custom Dataset

After converting the external dataset, retrieval can be evaluated with BM25:

```bash
python scripts/02_evaluate_retrieval.py \
  --corpus outputs/custom/teacher_dataset/corpus.csv \
  --benchmark outputs/custom/teacher_dataset/benchmark_rag_eval.csv \
  --retriever bm25 \
  --output-prefix outputs/custom/teacher_dataset/retrieval_bm25
```

Dense retrieval can also be evaluated:

```bash
python scripts/02_evaluate_retrieval.py \
  --corpus outputs/custom/teacher_dataset/corpus.csv \
  --benchmark outputs/custom/teacher_dataset/benchmark_rag_eval.csv \
  --retriever dense \
  --index-dir outputs/custom/teacher_dataset/indexes/dense_faiss \
  --output-prefix outputs/custom/teacher_dataset/retrieval_dense
```

Hybrid retrieval can be evaluated as follows:

```bash
python scripts/02_evaluate_retrieval.py \
  --corpus outputs/custom/teacher_dataset/corpus.csv \
  --benchmark outputs/custom/teacher_dataset/benchmark_rag_eval.csv \
  --retriever hybrid \
  --dense-weight 0.30 \
  --bm25-weight 0.70 \
  --index-dir outputs/custom/teacher_dataset/indexes/dense_faiss \
  --output-prefix outputs/custom/teacher_dataset/retrieval_hybrid_dense30_bm2570
```

---

### Answer Generation on a Custom Dataset

Using Ollama:

```bash
python scripts/08_generate_answers_ollama.py \
  --corpus outputs/custom/teacher_dataset/corpus.csv \
  --benchmark outputs/custom/teacher_dataset/benchmark_rag_eval.csv \
  --retriever bm25 \
  --top-k 5 \
  --prompt-type base \
  --model qwen2.5:7b \
  --output-path outputs/custom/teacher_dataset/predictions_bm25_qwen.csv
```

Using Hugging Face Qwen:

```bash
python scripts/13_generate_answers_hf_qlora.py \
  --corpus outputs/custom/teacher_dataset/corpus.csv \
  --benchmark outputs/custom/teacher_dataset/benchmark_rag_eval.csv \
  --retriever bm25 \
  --top-k 5 \
  --prompt-type base \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --output-path outputs/custom/teacher_dataset/predictions_hf_qwen25_base.csv
```

Using a QLoRA adapter:

```bash
python scripts/13_generate_answers_hf_qlora.py \
  --corpus outputs/custom/teacher_dataset/corpus.csv \
  --benchmark outputs/custom/teacher_dataset/benchmark_rag_eval.csv \
  --retriever bm25 \
  --top-k 5 \
  --prompt-type base \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-path outputs/custom/teacher_dataset/models/qwen25_qlora/adapter \
  --output-path outputs/custom/teacher_dataset/predictions_hf_qwen25_qlora.csv
```

---

### Important Evaluation Rule

External benchmark or hidden teacher test data must not be used for fine-tuning. The correct workflow is:

```text
External documents → retrieval corpus
External benchmark questions → evaluation only
Fine-tuning on hidden benchmark → not allowed
```

This prevents data leakage and ensures that the evaluation reflects generalization to unseen data.

## Quick Custom Dataset Run

For an external JSONL dataset:

```bash
python scripts/17_convert_external_dataset.py \
  --corpus-jsonl data/custom/teacher_dataset/raw/corpus.jsonl \
  --rag-eval-json data/custom/teacher_dataset/raw/rag_eval.json \
  --output-dir outputs/custom/teacher_dataset
```

Evaluate retrieval:

```bash
python scripts/02_evaluate_retrieval.py \
  --corpus outputs/custom/teacher_dataset/corpus.csv \
  --benchmark outputs/custom/teacher_dataset/benchmark_rag_eval.csv \
  --retriever bm25 \
  --output-prefix outputs/custom/teacher_dataset/retrieval_bm25
```

Generate answers:

```bash
python scripts/08_generate_answers_ollama.py \
  --corpus outputs/custom/teacher_dataset/corpus.csv \
  --benchmark outputs/custom/teacher_dataset/benchmark_rag_eval.csv \
  --retriever bm25 \
  --top-k 5 \
  --prompt-type base \
  --model qwen2.5:7b \
  --output-path outputs/custom/teacher_dataset/predictions_bm25_qwen.csv
```

If the teacher provides only raw text files:

```bash
python scripts/14_prepare_custom_corpus.py \
  --input-dir data/custom/teacher_dataset/raw_docs \
  --output-csv outputs/custom/teacher_dataset/corpus.csv
```

Then the generated `corpus.csv` can be used with the same retrieval and answer generation scripts.
