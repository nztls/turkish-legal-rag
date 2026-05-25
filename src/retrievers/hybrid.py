from __future__ import annotations

from typing import Dict, List

from retrievers.base import BaseRetriever


def _minmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [1.0 for _ in scores]
    return [(s - low) / (high - low) for s in scores]


class HybridRetriever(BaseRetriever):
    """Combines dense and BM25 scores after min-max normalization."""

    def __init__(self, dense: BaseRetriever, bm25: BaseRetriever, dense_weight: float = 0.65, bm25_weight: float = 0.35):
        self.dense = dense
        self.bm25 = bm25
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        pool_k = max(top_k * 5, 50)
        dense_hits = self.dense.search(query, top_k=pool_k)
        bm25_hits = self.bm25.search(query, top_k=pool_k)

        combined: Dict[int, Dict] = {}
        dense_scores = _minmax([hit["score"] for hit in dense_hits])
        bm25_scores = _minmax([hit["score"] for hit in bm25_hits])

        for hit, norm_score in zip(dense_hits, dense_scores):
            idx = hit["row_idx"]
            combined.setdefault(idx, {**hit, "score": 0.0, "dense_score": 0.0, "bm25_score": 0.0})
            combined[idx]["dense_score"] = norm_score
            combined[idx]["score"] += self.dense_weight * norm_score

        for hit, norm_score in zip(bm25_hits, bm25_scores):
            idx = hit["row_idx"]
            combined.setdefault(idx, {**hit, "score": 0.0, "dense_score": 0.0, "bm25_score": 0.0})
            combined[idx]["bm25_score"] = norm_score
            combined[idx]["score"] += self.bm25_weight * norm_score

        reranked = sorted(combined.values(), key=lambda item: item["score"], reverse=True)[:top_k]
        for rank, hit in enumerate(reranked, start=1):
            hit["rank"] = rank
        return reranked
