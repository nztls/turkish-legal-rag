import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sentence_transformers import CrossEncoder
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


def build_base_retriever(args):
    corpus = load_corpus(args.corpus)

    if args.base_retriever == "bm25":
        return BM25Retriever(text_col=args.text_col).fit(corpus)

    if args.base_retriever == "dense":
        if args.index_dir.exists():
            return DenseRetriever.load(args.index_dir, device=args.device)

        dense = DenseRetriever(
            model_name=args.embedding_model,
            text_col=args.text_col,
            device=args.device,
        ).fit(corpus)
        dense.save(args.index_dir)
        return dense

    if args.base_retriever == "hybrid":
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

    raise ValueError(f"Unknown base retriever: {args.base_retriever}")


def retrieve_candidates(retriever, query: str, top_k: int) -> list[dict[str, Any]]:
    """
    Calls the retriever method in a slightly defensive way.
    Our retrievers normally expose retrieve(query, top_k).
    """

    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query, top_k=top_k)

    if hasattr(retriever, "search"):
        return retriever.search(query, top_k=top_k)

    raise AttributeError("Retriever must have a retrieve() or search() method.")


def get_hit_text(hit: dict[str, Any]) -> str:
    """
    Text used by the reranker.

    Prefer retrieval_text if available because it includes source name + legal text.
    Fall back to context if needed.
    """

    text = hit.get("retrieval_text")

    if text is None or str(text).strip() == "":
        text = hit.get("context", "")

    return str(text)


def rerank_hits(
    reranker: CrossEncoder,
    query: str,
    hits: list[dict[str, Any]],
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    if not hits:
        return []

    pairs = [(query, get_hit_text(hit)) for hit in hits]
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    reranked = []

    for hit, score in zip(hits, scores):
        new_hit = dict(hit)
        new_hit["base_score"] = hit.get("score")
        new_hit["reranker_score"] = float(score)
        reranked.append(new_hit)

    reranked.sort(key=lambda item: item["reranker_score"], reverse=True)

    for rank, hit in enumerate(reranked, start=1):
        hit["rerank_rank"] = rank

    return reranked


def is_relevant(hit: dict[str, Any], gold_row: pd.Series) -> bool:
    """
    Primary match: context_key.

    Fallback match: kaynak + madde_no.
    This is useful if context_key formatting has small differences.
    """

    hit_context_key = str(hit.get("context_key", "")).strip()
    gold_context_key = str(gold_row.get("context_key", "")).strip()

    if hit_context_key and gold_context_key and hit_context_key == gold_context_key:
        return True

    hit_kaynak = str(hit.get("kaynak", "")).strip()
    gold_kaynak = str(gold_row.get("kaynak", "")).strip()

    hit_madde = str(hit.get("madde_no", "")).strip()
    gold_madde = str(gold_row.get("madde_no", "")).strip()

    return bool(hit_kaynak and gold_kaynak and hit_madde and gold_madde and hit_kaynak == gold_kaynak and hit_madde == gold_madde)


def reciprocal_rank(ranked_hits: list[dict[str, Any]], gold_row: pd.Series, k: int) -> float:
    for index, hit in enumerate(ranked_hits[:k], start=1):
        if is_relevant(hit, gold_row):
            return 1.0 / index
    return 0.0


def recall_at_k(ranked_hits: list[dict[str, Any]], gold_row: pd.Series, k: int) -> float:
    return 1.0 if any(is_relevant(hit, gold_row) for hit in ranked_hits[:k]) else 0.0


def ndcg_at_k(ranked_hits: list[dict[str, Any]], gold_row: pd.Series, k: int) -> float:
    """
    Single relevant document setting.

    If the relevant document is at rank r, DCG = 1 / log2(r + 1).
    Ideal DCG is 1 because the best possible rank is 1.
    """

    import math

    for index, hit in enumerate(ranked_hits[:k], start=1):
        if is_relevant(hit, gold_row):
            return 1.0 / math.log2(index + 1)

    return 0.0


def evaluate_reranker(
    base_retriever,
    reranker: CrossEncoder,
    benchmark: pd.DataFrame,
    candidate_k: int,
    final_k: int,
    batch_size: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    detail_rows = []

    recall_1 = []
    recall_3 = []
    recall_5 = []
    recall_10 = []
    mrr_values = []
    ndcg_10_values = []

    for _, row in tqdm(
        benchmark.iterrows(),
        total=len(benchmark),
        desc="Evaluating reranker",
    ):
        query = str(row["soru"])

        base_hits = retrieve_candidates(base_retriever, query, top_k=candidate_k)
        reranked_hits = rerank_hits(
            reranker=reranker,
            query=query,
            hits=base_hits,
            batch_size=batch_size,
        )

        r1 = recall_at_k(reranked_hits, row, 1)
        r3 = recall_at_k(reranked_hits, row, 3)
        r5 = recall_at_k(reranked_hits, row, 5)
        r10 = recall_at_k(reranked_hits, row, min(10, final_k))

        rr = reciprocal_rank(reranked_hits, row, min(10, final_k))
        ndcg10 = ndcg_at_k(reranked_hits, row, min(10, final_k))

        recall_1.append(r1)
        recall_3.append(r3)
        recall_5.append(r5)
        recall_10.append(r10)
        mrr_values.append(rr)
        ndcg_10_values.append(ndcg10)

        top_hits = reranked_hits[:final_k]

        relevant_rank = None
        relevant_context_key = None

        for rank, hit in enumerate(top_hits, start=1):
            if is_relevant(hit, row):
                relevant_rank = rank
                relevant_context_key = hit.get("context_key")
                break

        detail_rows.append(
            {
                "row_id": row.get("row_id"),
                "soru": row.get("soru"),
                "gold_context_key": row.get("context_key"),
                "gold_kaynak": row.get("kaynak"),
                "gold_madde_no": row.get("madde_no"),
                "relevant_found_in_top_k": relevant_rank is not None,
                "relevant_rank": relevant_rank,
                "matched_context_key": relevant_context_key,
                "top_context_keys": " || ".join(str(hit.get("context_key", "")) for hit in top_hits),
                "top_sources": " || ".join(
                    f"{hit.get('kaynak', '')} madde {hit.get('madde_no', '')}"
                    for hit in top_hits
                ),
                "top_reranker_scores": " || ".join(
                    f"{hit.get('reranker_score', 0.0):.6f}" for hit in top_hits
                ),
            }
        )

    metrics = {
        "num_questions": int(len(benchmark)),
        "candidate_k": int(candidate_k),
        "final_k": int(final_k),
        "recall@1": float(sum(recall_1) / len(recall_1)),
        "recall@3": float(sum(recall_3) / len(recall_3)),
        "recall@5": float(sum(recall_5) / len(recall_5)),
        "recall@10": float(sum(recall_10) / len(recall_10)),
        "mrr": float(sum(mrr_values) / len(mrr_values)),
        "ndcg@10": float(sum(ndcg_10_values) / len(ndcg_10_values)),
    }

    return metrics, pd.DataFrame(detail_rows)


def save_results(metrics: dict[str, float], details: pd.DataFrame, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    details.to_csv(output_prefix.with_suffix(".csv"), index=False, encoding="utf-8")

    output_prefix.with_suffix(".json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25/Dense/Hybrid retrieval followed by cross-encoder reranking."
    )

    paths = ProjectPaths()
    retrieval_config = RetrievalConfig()

    parser.add_argument("--corpus", type=Path, default=paths.corpus_csv)
    parser.add_argument("--benchmark", type=Path, default=paths.benchmark_csv)

    parser.add_argument(
        "--base-retriever",
        choices=["bm25", "dense", "hybrid"],
        default="bm25",
    )

    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )

    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=10)

    parser.add_argument("--index-dir", type=Path, default=paths.indexes_dir / "dense_faiss")
    parser.add_argument("--embedding-model", type=str, default=retrieval_config.embedding_model_name)
    parser.add_argument("--text-col", type=str, default=retrieval_config.text_col)

    parser.add_argument("--dense-weight", type=float, default=retrieval_config.dense_weight)
    parser.add_argument("--bm25-weight", type=float, default=retrieval_config.bm25_weight)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--output-prefix", type=Path, required=True)

    args = parser.parse_args()

    print("Benchmark:", args.benchmark)
    print("Base retriever:", args.base_retriever)
    print("Candidate k:", args.candidate_k)
    print("Final k:", args.final_k)
    print("Reranker model:", args.reranker_model)
    print("Output prefix:", args.output_prefix)

    benchmark = load_benchmark(args.benchmark, only_valid=True)
    base_retriever = build_base_retriever(args)

    reranker = CrossEncoder(
        args.reranker_model,
        device=args.device,
    )

    metrics, details = evaluate_reranker(
        base_retriever=base_retriever,
        reranker=reranker,
        benchmark=benchmark,
        candidate_k=args.candidate_k,
        final_k=args.final_k,
        batch_size=args.batch_size,
    )

    save_results(metrics, details, args.output_prefix)

    print("\nReranker metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nSaved details to: {args.output_prefix.with_suffix('.csv')}")
    print(f"Saved metrics to: {args.output_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()