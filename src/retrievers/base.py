from __future__ import annotations

from typing import Dict, List

import pandas as pd


class BaseRetriever:
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        raise NotImplementedError


def row_to_hit(row: pd.Series, score: float, rank: int, row_idx: int) -> Dict:
    return {
        "rank": rank,
        "score": float(score),
        "row_idx": int(row_idx),
        "kaynak": row.get("kaynak"),
        "madde_no": row.get("madde_no"),
        "context_key": row.get("context_key"),
        "context": row.get("context"),
        "retrieval_text": row.get("retrieval_text"),
        "url": row.get("url"),
    }
