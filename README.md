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
