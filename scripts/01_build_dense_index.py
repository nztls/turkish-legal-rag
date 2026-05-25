import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ProjectPaths, RetrievalConfig
from data import load_corpus
from retrievers.dense import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dense vector index for Turkish Legal RAG corpus."
    )

    default_paths = ProjectPaths()
    default_config = RetrievalConfig()

    parser.add_argument(
        "--corpus",
        type=Path,
        default=default_paths.corpus_csv,
        help="Path to corpus CSV file."
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=default_paths.indexes_dir / "dense_faiss",
        help="Directory where the dense index will be saved."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=default_config.embedding_model_name,
        help="SentenceTransformer embedding model name."
    )

    parser.add_argument(
        "--text-col",
        type=str,
        default=default_config.text_col,
        help="Corpus column used for retrieval."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size."
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device: cpu, cuda, cuda:0"
    )

    args = parser.parse_args()

    print("Corpus path:", args.corpus)
    print("Index output directory:", args.index_dir)
    print("Embedding model:", args.model)
    print("Text column:", args.text_col)

    corpus = load_corpus(args.corpus)

    retriever = DenseRetriever(
        model_name=args.model,
        text_col=args.text_col,
        device=args.device
    )

    retriever.fit(corpus, batch_size=args.batch_size)
    retriever.save(args.index_dir)

    print(f"\nDense index saved to: {args.index_dir}")


if __name__ == "__main__":
    main()