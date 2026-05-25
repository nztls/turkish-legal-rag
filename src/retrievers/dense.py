from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from retrievers.base import BaseRetriever, row_to_hit


class DenseRetriever(BaseRetriever):
    """FAISS-based dense retriever.

    Uses cosine similarity by normalizing embeddings and searching with inner product.
    """

    def __init__(
        self,
        model_name: str,
        text_col: str = "retrieval_text",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.text_col = text_col
        self.model = SentenceTransformer(model_name, device=device)
        self.corpus: Optional[pd.DataFrame] = None
        self.index: Optional[faiss.Index] = None

    def fit(self, corpus: pd.DataFrame, batch_size: int = 64, show_progress: bool = True) -> "DenseRetriever":
        self.corpus = corpus.reset_index(drop=True).copy()
        texts = self.corpus[self.text_col].fillna("").astype(str).tolist()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.index = index
        return self

    def save(self, index_dir: Path) -> None:
        if self.index is None or self.corpus is None:
            raise RuntimeError("DenseRetriever must be fitted before saving.")
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "faiss.index"))
        self.corpus.to_parquet(index_dir / "corpus.parquet", index=False)
        metadata = {"model_name": self.model_name, "text_col": self.text_col}
        (index_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path, device: Optional[str] = None) -> "DenseRetriever":
        index_dir = Path(index_dir)
        metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        obj = cls(metadata["model_name"], metadata.get("text_col", "retrieval_text"), device=device)
        obj.index = faiss.read_index(str(index_dir / "faiss.index"))
        obj.corpus = pd.read_parquet(index_dir / "corpus.parquet")
        return obj

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        if self.index is None or self.corpus is None:
            raise RuntimeError("DenseRetriever must be fitted or loaded before search.")
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        hits: List[Dict] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            row = self.corpus.iloc[int(idx)]
            hits.append(row_to_hit(row, float(score), rank, int(idx)))
        return hits
