from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """
    Centralized project paths.

    The raw dataset is treated as read-only.
    Scripts only read from data/raw and write results under outputs/.
    """

    project_root: Path = Path(__file__).resolve().parents[1]

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def corpus_csv(self) -> Path:
        return self.raw_data_dir / "turk_rag_corpus.csv"

    @property
    def benchmark_csv(self) -> Path:
        return self.raw_data_dir / "qa_benchmark_gold.csv"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def indexes_dir(self) -> Path:
        return self.outputs_dir / "indexes"

    @property
    def predictions_dir(self) -> Path:
        return self.outputs_dir / "predictions"

    @property
    def metrics_dir(self) -> Path:
        return self.outputs_dir / "metrics"


@dataclass(frozen=True)
class RetrievalConfig:
    """
    Retrieval configuration for baseline experiments.
    """

    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    text_col: str = "retrieval_text"
    top_k: int = 10
    dense_weight: float = 0.65
    bm25_weight: float = 0.35