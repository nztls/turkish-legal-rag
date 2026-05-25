from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_CORPUS_COLUMNS = {
    "kaynak",
    "madde_no",
    "context_key",
    "context",
    "retrieval_text",
    "chunk_strategy",
    "kanun_no",
    "url",
}

REQUIRED_BENCHMARK_COLUMNS = {
    "soru",
    "cevap",
    "context",
    "kaynak",
    "kanun_adi",
    "madde_no",
    "madde_nolari_context",
    "score_valid",
}


def _read_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, encoding="utf-8")


def load_corpus(path: Path) -> pd.DataFrame:
    """Load corpus without modifying the source CSV file."""
    df = _read_csv(Path(path))
    missing = REQUIRED_CORPUS_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")
    return df


def load_benchmark(path: Path, only_valid: bool = True) -> pd.DataFrame:
    """Load benchmark without modifying the source CSV file."""
    df = _read_csv(Path(path))
    missing = REQUIRED_BENCHMARK_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark is missing required columns: {sorted(missing)}")
    if only_valid and "score_valid" in df.columns:
        df = df[df["score_valid"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    return df.reset_index(drop=True)


def dataset_summary(corpus: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None) -> dict:
    summary = {
        "corpus_rows": int(len(corpus)),
        "corpus_columns": list(corpus.columns),
        "unique_context_keys": int(corpus["context_key"].nunique()),
        "unique_sources": int(corpus["kaynak"].nunique()),
    }
    if benchmark is not None:
        summary.update(
            {
                "benchmark_rows": int(len(benchmark)),
                "benchmark_columns": list(benchmark.columns),
                "benchmark_unique_sources": int(benchmark["kaynak"].nunique()),
            }
        )
    return summary
