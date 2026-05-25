from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ProjectPaths
from data import load_benchmark


def stratified_split_by_source(
    df: pd.DataFrame,
    source_col: str = "kaynak",
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split benchmark into train/validation/test subsets.

    The original dataset is not modified.
    This function tries to preserve source distribution by splitting within each source group.
    """

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in df.groupby(source_col, dropna=False):
        group = group.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        n = len(group)

        if n == 1:
            train_parts.append(group)
            continue

        n_test = round(n * test_ratio)
        n_val = round(n * val_ratio)

        if n >= 3:
            n_test = max(1, n_test)
        if n >= 10:
            n_val = max(1, n_val)

        n_train = n - n_val - n_test

        if n_train <= 0:
            n_train = max(1, n - 1)
            n_val = 0
            n_test = n - n_train

        test_part = group.iloc[:n_test]
        val_part = group.iloc[n_test:n_test + n_val]
        train_part = group.iloc[n_test + n_val:]

        train_parts.append(train_part)
        val_parts.append(val_part)
        test_parts.append(test_part)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return train_df, val_df, test_df


def main() -> None:
    paths = ProjectPaths()

    benchmark = load_benchmark(paths.benchmark_csv, only_valid=True)

    train_df, val_df, test_df = stratified_split_by_source(benchmark)

    output_dir = paths.outputs_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "benchmark_train.csv"
    val_path = output_dir / "benchmark_val.csv"
    test_path = output_dir / "benchmark_test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print("Benchmark split completed.")
    print(f"Original benchmark rows: {len(benchmark)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")

    print(f"\nSaved train split to: {train_path}")
    print(f"Saved validation split to: {val_path}")
    print(f"Saved test split to: {test_path}")


if __name__ == "__main__":
    main()