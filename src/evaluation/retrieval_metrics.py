from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from normalization import normalize_article_no, normalize_source, parse_article_set


def gold_source(row: pd.Series) -> str:
    # Prefer the explicit source column. Fallback to kanun_adi if needed.
    source = row.get("kaynak")
    if pd.isna(source) or str(source).strip() == "":
        source = row.get("kanun_adi")
    return normalize_source(source)


def gold_articles(row: pd.Series) -> set[str]:
    # Some gold rows include multiple supporting articles like '8-|9-'.
    return parse_article_set(row.get("madde_no"), row.get("madde_nolari_context"))


def hit_is_relevant(hit: Dict, gold_row: pd.Series) -> bool:
    hit_source = normalize_source(hit.get("kaynak"))
    hit_article = normalize_article_no(hit.get("madde_no"))
    return hit_source == gold_source(gold_row) and hit_article in gold_articles(gold_row)


def reciprocal_rank(hits: Sequence[Dict], gold_row: pd.Series) -> float:
    for rank, hit in enumerate(hits, start=1):
        if hit_is_relevant(hit, gold_row):
            return 1.0 / rank
    return 0.0


def recall_at_k(hits: Sequence[Dict], gold_row: pd.Series, k: int) -> float:
    return float(any(hit_is_relevant(hit, gold_row) for hit in hits[:k]))


def ndcg_at_k(hits: Sequence[Dict], gold_row: pd.Series, k: int) -> float:
    dcg = 0.0
    for rank, hit in enumerate(hits[:k], start=1):
        rel = 1.0 if hit_is_relevant(hit, gold_row) else 0.0
        dcg += rel / math.log2(rank + 1)
    # With binary relevance and at least one expected relevant document, ideal DCG@k is 1.
    return dcg


def evaluate_retriever(retriever, benchmark: pd.DataFrame, top_k: int = 10) -> Tuple[Dict, pd.DataFrame]:
    rows: List[Dict] = []

    for _, gold in tqdm(benchmark.iterrows(), total=len(benchmark), desc="Evaluating retrieval"):
        question = str(gold["soru"])
        hits = retriever.search(question, top_k=top_k)
        first_relevant_rank = None
        for hit in hits:
            if hit_is_relevant(hit, gold):
                first_relevant_rank = hit["rank"]
                break

        rows.append(
            {
                "row_id": gold.get("row_id"),
                "question": question,
                "gold_source": gold_source(gold),
                "gold_articles": "|".join(sorted(gold_articles(gold))),
                "first_relevant_rank": first_relevant_rank,
                "recall@1": recall_at_k(hits, gold, 1),
                "recall@3": recall_at_k(hits, gold, 3),
                "recall@5": recall_at_k(hits, gold, 5),
                "recall@10": recall_at_k(hits, gold, 10),
                "mrr": reciprocal_rank(hits, gold),
                "ndcg@10": ndcg_at_k(hits, gold, 10),
                "top1_source": hits[0].get("kaynak") if hits else None,
                "top1_article": hits[0].get("madde_no") if hits else None,
                "top1_score": hits[0].get("score") if hits else None,
                "top1_context_key": hits[0].get("context_key") if hits else None,
                "top5_context_keys": " || ".join(str(hit.get("context_key")) for hit in hits[:5]),
            }
        )

    detail_df = pd.DataFrame(rows)
    metrics = {
        "num_questions": int(len(detail_df)),
        "recall@1": float(detail_df["recall@1"].mean()),
        "recall@3": float(detail_df["recall@3"].mean()),
        "recall@5": float(detail_df["recall@5"].mean()),
        "recall@10": float(detail_df["recall@10"].mean()),
        "mrr": float(detail_df["mrr"].mean()),
        "ndcg@10": float(detail_df["ndcg@10"].mean()),
    }
    return metrics, detail_df


def save_retrieval_results(metrics: Dict, details: pd.DataFrame, output_prefix: Path) -> None:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_prefix.with_suffix(".csv"), index=False, encoding="utf-8")
    output_prefix.with_suffix(".json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
