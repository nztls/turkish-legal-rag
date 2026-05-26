import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ProjectPaths, RetrievalConfig
from data import load_corpus
from retrievers.bm25 import BM25Retriever


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def is_same_document(hit: dict[str, Any], row: pd.Series) -> bool:
    hit_context_key = str(hit.get("context_key", "")).strip()
    gold_context_key = str(row.get("context_key", "")).strip()

    if hit_context_key and gold_context_key and hit_context_key == gold_context_key:
        return True

    hit_kaynak = str(hit.get("kaynak", "")).strip()
    gold_kaynak = str(row.get("kaynak", "")).strip()

    hit_madde = str(hit.get("madde_no", "")).strip()
    gold_madde = str(row.get("madde_no", "")).strip()

    return bool(
        hit_kaynak
        and gold_kaynak
        and hit_madde
        and gold_madde
        and hit_kaynak == gold_kaynak
        and hit_madde == gold_madde
    )


def get_gold_document_text(row: pd.Series, corpus: pd.DataFrame) -> str:
    """
    Prefer the matching corpus retrieval_text.
    If not found, fall back to benchmark context.
    """

    context_key = str(row.get("context_key", "")).strip()

    if context_key:
        matched = corpus[corpus["context_key"].astype(str).str.strip() == context_key]
        if len(matched) > 0:
            return str(matched.iloc[0].get("retrieval_text", matched.iloc[0].get("context", "")))

    kaynak = str(row.get("kaynak", "")).strip()
    madde_no = str(row.get("madde_no", "")).strip()

    if kaynak and madde_no:
        matched = corpus[
            (corpus["kaynak"].astype(str).str.strip() == kaynak)
            & (corpus["madde_no"].astype(str).str.strip() == madde_no)
        ]
        if len(matched) > 0:
            return str(matched.iloc[0].get("retrieval_text", matched.iloc[0].get("context", "")))

    return str(row.get("context", ""))


def get_hit_text(hit: dict[str, Any]) -> str:
    text = hit.get("retrieval_text")

    if text is None or str(text).strip() == "":
        text = hit.get("context", "")

    return str(text)

def retrieve_candidates(retriever, query: str, top_k: int) -> list[dict[str, Any]]:
    """
    Calls the retriever method safely.

    Some retrievers expose search(query, top_k),
    while others may expose retrieve(query, top_k).
    """

    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query, top_k=top_k)

    if hasattr(retriever, "search"):
        return retriever.search(query, top_k=top_k)

    raise AttributeError("Retriever must have a retrieve() or search() method.")

def build_pairs(
    benchmark: pd.DataFrame,
    corpus: pd.DataFrame,
    retriever: BM25Retriever,
    candidate_k: int,
    negatives_per_question: int,
) -> list[dict[str, Any]]:
    pairs = []

    for _, row in tqdm(
        benchmark.iterrows(),
        total=len(benchmark),
        desc="Preparing reranker pairs",
    ):
        query = str(row["soru"]).strip()

        if not query:
            continue

        gold_text = get_gold_document_text(row, corpus).strip()

        if gold_text:
            pairs.append(
                {
                    "query": query,
                    "document": gold_text,
                    "label": 1.0,
                    "row_id": row.get("row_id"),
                    "source": "positive_gold",
                    "gold_context_key": row.get("context_key"),
                    "gold_kaynak": row.get("kaynak"),
                    "gold_madde_no": row.get("madde_no"),
                }
            )

        hits = retrieve_candidates(retriever, query, candidate_k)

        negative_count = 0

        for hit in hits:
            if is_same_document(hit, row):
                continue

            doc_text = get_hit_text(hit).strip()

            if not doc_text:
                continue

            pairs.append(
                {
                    "query": query,
                    "document": doc_text,
                    "label": 0.0,
                    "row_id": row.get("row_id"),
                    "source": "hard_negative_bm25",
                    "negative_context_key": hit.get("context_key"),
                    "negative_kaynak": hit.get("kaynak"),
                    "negative_madde_no": hit.get("madde_no"),
                    "gold_context_key": row.get("context_key"),
                    "gold_kaynak": row.get("kaynak"),
                    "gold_madde_no": row.get("madde_no"),
                }
            )

            negative_count += 1

            if negative_count >= negatives_per_question:
                break

    return pairs


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare positive and hard-negative pairs for reranker fine-tuning."
    )

    paths = ProjectPaths()
    config = RetrievalConfig()

    parser.add_argument("--corpus", type=Path, default=paths.corpus_csv)
    parser.add_argument("--train", type=Path, default=paths.outputs_dir / "splits" / "benchmark_train.csv")
    parser.add_argument("--val", type=Path, default=paths.outputs_dir / "splits" / "benchmark_val.csv")
    parser.add_argument("--text-col", type=str, default=config.text_col)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--negatives-per-question", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=paths.outputs_dir / "reranker_data")

    args = parser.parse_args()

    print("Corpus:", args.corpus)
    print("Train split:", args.train)
    print("Validation split:", args.val)
    print("Candidate k:", args.candidate_k)
    print("Negatives per question:", args.negatives_per_question)

    corpus = load_corpus(args.corpus)
    train_df = load_csv(args.train)
    val_df = load_csv(args.val)

    bm25 = BM25Retriever(text_col=args.text_col).fit(corpus)

    train_pairs = build_pairs(
        benchmark=train_df,
        corpus=corpus,
        retriever=bm25,
        candidate_k=args.candidate_k,
        negatives_per_question=args.negatives_per_question,
    )

    val_pairs = build_pairs(
        benchmark=val_df,
        corpus=corpus,
        retriever=bm25,
        candidate_k=args.candidate_k,
        negatives_per_question=args.negatives_per_question,
    )

    train_output = args.output_dir / "train_pairs.jsonl"
    val_output = args.output_dir / "val_pairs.jsonl"

    write_jsonl(train_pairs, train_output)
    write_jsonl(val_pairs, val_output)

    print("\nReranker data prepared.")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Saved train pairs to: {train_output}")
    print(f"Saved validation pairs to: {val_output}")


if __name__ == "__main__":
    main()