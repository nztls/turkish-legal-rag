import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any
import re

import pandas as pd
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ProjectPaths, RetrievalConfig
from data import load_benchmark, load_corpus
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever
from retrievers.hybrid import HybridRetriever


def build_retriever(args):
    corpus = load_corpus(args.corpus)

    if args.retriever == "bm25":
        return BM25Retriever(text_col=args.text_col).fit(corpus)

    if args.retriever == "dense":
        if args.index_dir.exists():
            return DenseRetriever.load(args.index_dir, device=args.device)

        retriever = DenseRetriever(
            model_name=args.embedding_model,
            text_col=args.text_col,
            device=args.device,
        ).fit(corpus)
        retriever.save(args.index_dir)
        return retriever

    if args.retriever == "hybrid":
        if args.index_dir.exists():
            dense = DenseRetriever.load(args.index_dir, device=args.device)
        else:
            dense = DenseRetriever(
                model_name=args.embedding_model,
                text_col=args.text_col,
                device=args.device,
            ).fit(corpus)
            dense.save(args.index_dir)

        bm25 = BM25Retriever(text_col=args.text_col).fit(corpus)

        return HybridRetriever(
            dense,
            bm25,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
        )

    raise ValueError(f"Unknown retriever: {args.retriever}")


def retrieve_candidates(retriever, query: str, top_k: int) -> list[dict[str, Any]]:
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query, top_k=top_k)

    if hasattr(retriever, "search"):
        return retriever.search(query, top_k=top_k)

    raise AttributeError("Retriever must have retrieve() or search().")


def format_contexts(hits: list[dict[str, Any]]) -> str:
    blocks = []

    for i, hit in enumerate(hits, start=1):
        kaynak = hit.get("kaynak", "")
        madde_no = hit.get("madde_no", "")
        context = hit.get("context", "")

        if not context:
            context = hit.get("retrieval_text", "")

        blocks.append(
            f"[{i}] Kaynak: {kaynak}\n"
            f"Madde No: {madde_no}\n"
            f"Metin:\n{context}"
        )

    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, contexts_text: str, prompt_type: str) -> str:
    if prompt_type == "base":
        return f"""
Aşağıdaki hukuki kaynak metinlerine dayanarak soruya Türkçe cevap ver.

Soru:
{question}

Kaynak metinler:
{contexts_text}

Cevap:
""".strip()

    if prompt_type == "strict":
        return f"""
    Sen Türk hukuku alanında kaynaklara bağlı çalışan bir RAG cevaplama sistemisin.

    Zorunlu kurallar:
    1. Sadece verilen kaynak metinlerde açıkça bulunan bilgiye dayanarak cevap ver.
    2. Kaynaklarda açıkça desteklenmeyen hiçbir bilgi ekleme.
    3. Soru belirli bir madde, geçici madde, süre, şart veya istisna soruyorsa; bu bilgi kaynaklarda açıkça yoksa cevap üretme.
    4. Kaynaklar soruyu doğrudan cevaplamıyorsa aynen şu cümleyi yaz:
    "Verilen kaynaklarda bu soruyu cevaplamak için yeterli bilgi bulunmamaktadır."
    5. Genel hukuk bilgisi, tahmin veya dış bilgi kullanma.
    6. Cevabı kısa, açık ve Türkçe yaz.
    7. Cevabın sonunda kullandığın kaynakları [1], [2] biçiminde belirt.

    Soru:
    {question}

    Kaynak metinler:
    {contexts_text}

    Cevap:
    """.strip()

    raise ValueError(f"Unknown prompt_type: {prompt_type}")


def call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
    temperature: float,
    max_tokens: int,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> str:
    endpoint = f"{ollama_url.rstrip('/')}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(endpoint, json=payload, timeout=180)

            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                time.sleep(sleep_seconds * attempt)
                continue

            data = response.json()
            return str(data.get("response", "")).strip()

        except Exception as exc:
            last_error = str(exc)
            time.sleep(sleep_seconds * attempt)

    raise RuntimeError(f"Ollama request failed after {retries} retries. Last error: {last_error}")


def is_gold_in_hits(row: pd.Series, hits: list[dict[str, Any]]) -> tuple[bool, int | None]:
    gold_context_key = normalize_for_match(row.get("context_key", ""))
    gold_kaynak = normalize_for_match(row.get("kaynak", ""))
    gold_madde = normalize_madde_no(row.get("madde_no", ""))

    for rank, hit in enumerate(hits, start=1):
        hit_context_key = normalize_for_match(hit.get("context_key", ""))
        hit_kaynak = normalize_for_match(hit.get("kaynak", ""))
        hit_madde = normalize_madde_no(hit.get("madde_no", ""))

        if gold_context_key and hit_context_key and gold_context_key == hit_context_key:
            return True, rank

        if gold_kaynak and gold_madde and hit_kaynak == gold_kaynak and hit_madde == gold_madde:
            return True, rank

        # Kaynak adlarında küçük farklar varsa:
        # örn. "Bilgi Edinme Kanunu" vs "Bilgi Edinme Hakkı Kanunu"
        if gold_kaynak and hit_kaynak and gold_madde and hit_madde:
            same_source_like = gold_kaynak in hit_kaynak or hit_kaynak in gold_kaynak
            same_article = gold_madde == hit_madde

            if same_source_like and same_article:
                return True, rank

    return False, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RAG answers using Ollama as the local LLM generator."
    )

    paths = ProjectPaths()
    config = RetrievalConfig()

    parser.add_argument("--corpus", type=Path, default=paths.corpus_csv)
    parser.add_argument("--benchmark", type=Path, required=True)

    parser.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid"],
        default="bm25",
    )

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prompt-type", choices=["base", "strict"], default="strict")

    parser.add_argument("--model", type=str, default="gemma3:4b")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)

    parser.add_argument("--index-dir", type=Path, default=paths.indexes_dir / "dense_faiss")
    parser.add_argument("--embedding-model", type=str, default=config.embedding_model_name)
    parser.add_argument("--text-col", type=str, default=config.text_col)
    parser.add_argument("--dense-weight", type=float, default=config.dense_weight)
    parser.add_argument("--bm25-weight", type=float, default=config.bm25_weight)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)

    parser.add_argument("--output-path", type=Path, required=True)

    args = parser.parse_args()

    print("Benchmark:", args.benchmark)
    print("Retriever:", args.retriever)
    print("Top-k:", args.top_k)
    print("Prompt type:", args.prompt_type)
    print("Ollama model:", args.model)
    print("Ollama URL:", args.ollama_url)
    print("Output path:", args.output_path)

    benchmark = load_benchmark(args.benchmark, only_valid=True)

    if args.limit is not None:
        benchmark = benchmark.head(args.limit).copy()

    retriever = build_retriever(args)

    rows = []

    for _, row in tqdm(
        benchmark.iterrows(),
        total=len(benchmark),
        desc="Generating Ollama answers",
    ):
        question = str(row["soru"])

        hits = retrieve_candidates(retriever, question, top_k=args.top_k)
        contexts_text = format_contexts(hits)

        prompt = build_prompt(
            question=question,
            contexts_text=contexts_text,
            prompt_type=args.prompt_type,
        )

        answer = call_ollama(
            prompt=prompt,
            model=args.model,
            ollama_url=args.ollama_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        gold_found, gold_rank = is_gold_in_hits(row, hits)

        rows.append(
            {
                "row_id": row.get("row_id"),
                "question": question,
                "gold_answer": row.get("cevap"),
                "generated_answer": answer,
                "gold_context_key": row.get("context_key"),
                "gold_kaynak": row.get("kaynak"),
                "gold_madde_no": row.get("madde_no"),
                "retriever": args.retriever,
                "prompt_type": args.prompt_type,
                "model": args.model,
                "top_k": args.top_k,
                "gold_found_in_context": gold_found,
                "gold_rank": gold_rank,
                "retrieved_context_keys": " || ".join(str(hit.get("context_key", "")) for hit in hits),
                "retrieved_sources": " || ".join(
                    f"[{i}] {hit.get('kaynak', '')} madde {hit.get('madde_no', '')}"
                    for i, hit in enumerate(hits, start=1)
                ),
                "prompt": prompt,
            }
        )

        time.sleep(args.sleep_seconds)

    output_df = pd.DataFrame(rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_path, index=False, encoding="utf-8")

    print(f"\nGenerated answers saved to: {args.output_path}")

def normalize_for_match(value) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()

    if text in {"nan", "none", "null"}:
        return ""

    text = text.replace("ı", "i")
    text = text.replace("ğ", "g")
    text = text.replace("ü", "u")
    text = text.replace("ş", "s")
    text = text.replace("ö", "o")
    text = text.replace("ç", "c")

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def normalize_madde_no(value) -> str:
    text = normalize_for_match(value)

    # 8.0 -> 8
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    # "Madde 8" -> "8"
    text = re.sub(r"^madde\s+", "", text)

    # "Geçici Madde 13" gibi ifadeleri normalize et
    text = text.replace("gecici madde", "gecici")
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    main()