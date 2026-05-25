from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from retrievers.base import BaseRetriever, row_to_hit
from normalization import normalize_text


_TOKEN_RE = re.compile(r"[0-9a-zA-ZçğıöşüÇĞİÖŞÜ]+")


def tokenize_tr(text: object) -> List[str]:
    normalized = normalize_text(text)
    return _TOKEN_RE.findall(normalized)


class BM25Retriever(BaseRetriever):
    def __init__(self, text_col: str = "retrieval_text") -> None:
        self.text_col = text_col
        self.corpus: Optional[pd.DataFrame] = None
        self.bm25: Optional[BM25Okapi] = None

    def fit(self, corpus: pd.DataFrame) -> "BM25Retriever":
        self.corpus = corpus.reset_index(drop=True).copy()
        tokenized = [tokenize_tr(text) for text in self.corpus[self.text_col].fillna("").astype(str).tolist()]
        self.bm25 = BM25Okapi(tokenized)
        return self

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if self.bm25 is None or self.corpus is None:
            raise RuntimeError("BM25Retriever must be fitted before search.")
        scores = self.bm25.get_scores(tokenize_tr(query))
        if len(scores) == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:top_k]
        hits: List[Dict] = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.corpus.iloc[int(idx)]
            hits.append(row_to_hit(row, float(scores[idx]), rank, int(idx)))
        return hits
