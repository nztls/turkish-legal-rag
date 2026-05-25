from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ProjectPaths
from data import dataset_summary, load_benchmark, load_corpus


def main() -> None:
    paths = ProjectPaths()

    print("Project root:", paths.project_root)
    print("Corpus path:", paths.corpus_csv)
    print("Benchmark path:", paths.benchmark_csv)

    corpus = load_corpus(paths.corpus_csv)
    benchmark = load_benchmark(paths.benchmark_csv, only_valid=True)

    summary = dataset_summary(corpus, benchmark)

    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = paths.metrics_dir / "dataset_readonly_summary.json"

    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()