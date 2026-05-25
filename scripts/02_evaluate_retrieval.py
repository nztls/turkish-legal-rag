import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from config import ProjectPaths, RetrievalConfig
from data import load_benchmark, load_corpus
from evaluation.retrieval_metrics import evaluate_retriever, save_retrieval_results
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever
from retrievers.hybrid import HybridRetriever


def build_retriever(args):
    corpus = load_corpus(args.corpus)
    if args.retriever == "bm25":
        return BM25Retriever(text_col=args.text_col).fit(corpus)

    if args.retriever == "dense":
        if args.index_dir.exists():
            return DenseRetriever.load(args.index_dir, device=args.device)
        retriever = DenseRetriever(args.model, text_col=args.text_col, device=args.device).fit(corpus)
        retriever.save(args.index_dir)
        return retriever

    if args.retriever == "hybrid":
        if args.index_dir.exists():
            dense = DenseRetriever.load(args.index_dir, device=args.device)
        else:
            dense = DenseRetriever(args.model, text_col=args.text_col, device=args.device).fit(corpus)
            dense.save(args.index_dir)
        bm25 = BM25Retriever(text_col=args.text_col).fit(corpus)
        return HybridRetriever(dense, bm25, dense_weight=args.dense_weight, bm25_weight=args.bm25_weight)

    raise ValueError(f"Unknown retriever: {args.retriever}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics on the gold benchmark.")
    parser.add_argument("--corpus", type=Path, default=ProjectPaths().corpus_csv)
    parser.add_argument("--benchmark", type=Path, default=ProjectPaths().benchmark_csv)
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="dense")
    parser.add_argument("--index-dir", type=Path, default=ProjectPaths().indexes_dir / "dense_faiss")
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--model", type=str, default=RetrievalConfig().embedding_model_name)
    parser.add_argument("--text-col", type=str, default=RetrievalConfig().text_col)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dense-weight", type=float, default=RetrievalConfig().dense_weight)
    parser.add_argument("--bm25-weight", type=float, default=RetrievalConfig().bm25_weight)
    args = parser.parse_args()

    output_prefix = args.output_prefix

    if output_prefix is None:
        if args.retriever == "hybrid":
            dense_label = int(round(args.dense_weight * 100))
            bm25_label = int(round(args.bm25_weight * 100))

            output_prefix = (
                    ProjectPaths().metrics_dir
                    / f"retrieval_hybrid_dense{dense_label}_bm25{bm25_label}"
            )
        else:
            output_prefix = ProjectPaths().metrics_dir / f"retrieval_{args.retriever}"

    benchmark = load_benchmark(args.benchmark, only_valid=True)
    retriever = build_retriever(args)
    metrics, details = evaluate_retriever(retriever, benchmark, top_k=args.top_k)
    save_retrieval_results(metrics, details, output_prefix)

    print("\nRetrieval metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nSaved details to: {output_prefix.with_suffix('.csv')}")
    print(f"Saved metrics to: {output_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
