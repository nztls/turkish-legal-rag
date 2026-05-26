from pathlib import Path
import json
import webbrowser

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

OUTPUT_CSV = METRICS_DIR / "accuracy_comparison_table.csv"
OUTPUT_HTML = METRICS_DIR / "accuracy_comparison_table.html"

# Şimdilik accuracy olarak Recall@5 kullanıyoruz.
# Çünkü RAG pipeline'da LLM'e top-5 context verilecekse,
# doğru dokümanın top-5 içinde olması retrieval accuracy gibi yorumlanabilir.
ACCURACY_METRIC = "recall@5"


def infer_split(filename: str) -> str:
    name = filename.lower()

    if name.startswith("test_"):
        return "Test"
    if name.startswith("val_"):
        return "Validation"
    if name.startswith("retrieval_"):
        return "Full Benchmark"

    return "Other"


def infer_method(filename: str) -> str:
    name = filename.replace(".json", "")

    if "reranker" in name and "mmarco" in name:
        return "BM25 + Zero-shot mMARCO Reranker RAG"

    if "reranker" in name and "bge" in name:
        return "BM25 + Zero-shot BGE Reranker RAG"

    if "hybrid" in name:
        return "Hybrid RAG"

    if "bm25" in name and "reranker" not in name:
        return "BM25 RAG"

    if "dense" in name and "hybrid" not in name:
        return "Dense RAG"

    return name


def infer_retrieval_setup(filename: str) -> str:
    name = filename.replace(".json", "")

    if "reranker" in name and "mmarco" in name:
        return "BM25 top-20 + mMARCO reranker"

    if "reranker" in name and "bge" in name:
        return "BM25 top-20 + BGE reranker"

    if "dense10_bm2590" in name:
        return "Hybrid retrieval: Dense 0.10 + BM25 0.90"

    if "dense20_bm2580" in name:
        return "Hybrid retrieval: Dense 0.20 + BM25 0.80"

    if "dense30_bm2570" in name:
        return "Hybrid retrieval: Dense 0.30 + BM25 0.70"

    if "dense40_bm2560" in name:
        return "Hybrid retrieval: Dense 0.40 + BM25 0.60"

    if "hybrid" in name:
        return "Hybrid retrieval"

    if "bm25" in name and "reranker" not in name:
        return "BM25 lexical retrieval"

    if "dense" in name:
        return "Dense vector retrieval"

    return "-"


def infer_prompt(filename: str) -> str:
    # Şu an sadece retrieval ölçüyoruz, LLM prompt yok.
    # LLM answer generation eklenince burayı Base prompt / Strict prompt diye ayıracağız.
    if "reranker" in filename:
        return "No generation prompt"

    return "No generation prompt"


def to_percent(value):
    if value is None:
        return None

    try:
        return round(float(value) * 100, 2)
    except (TypeError, ValueError):
        return None


def build_accuracy_table() -> pd.DataFrame:
    rows = []

    json_files = sorted(METRICS_DIR.glob("*.json"))

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # dataset_readonly_summary.json gibi metric olmayan dosyaları atla
        if ACCURACY_METRIC not in metrics:
            continue

        filename = path.name

        rows.append(
            {
                "split": infer_split(filename),
                "method": infer_method(filename),
                "retrieval_setup": infer_retrieval_setup(filename),
                "prompt": infer_prompt(filename),
                "accuracy": to_percent(metrics.get(ACCURACY_METRIC)),
                "metric_used": ACCURACY_METRIC,
                "file": filename,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No metric JSON files found in: {METRICS_DIR}")

    df = pd.DataFrame(rows)

    split_order = {
        "Validation": 0,
        "Test": 1,
        "Full Benchmark": 2,
        "Other": 3,
    }

    df["split_order"] = df["split"].map(split_order).fillna(99)

    df = df.sort_values(
        by=["split_order", "split", "accuracy"],
        ascending=[True, True, False],
    )

    df = df.drop(columns=["split_order"]).reset_index(drop=True)

    return df


def make_html(df: pd.DataFrame) -> str:
    table_html = df.to_html(index=True, escape=False)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Accuracy Comparison Table</title>
<style>
    body {{
        background-color: #1e1e1e;
        color: #f2f2f2;
        font-family: Arial, sans-serif;
        padding: 24px;
    }}

    h1 {{
        color: #ffffff;
        margin-bottom: 8px;
    }}

    p {{
        color: #cfcfcf;
        margin-bottom: 24px;
    }}

    table {{
        border-collapse: collapse;
        width: 100%;
        font-size: 14px;
        background-color: #2b2b2b;
    }}

    thead tr {{
        background-color: #111111;
    }}

    th {{
        padding: 10px;
        text-align: left;
        color: #ffffff;
        border-bottom: 2px solid #555555;
        white-space: nowrap;
    }}

    td {{
        padding: 9px 10px;
        border-bottom: 1px solid #444444;
        white-space: nowrap;
    }}

    tbody tr:nth-child(even) {{
        background-color: #242424;
    }}

    tbody tr:nth-child(odd) {{
        background-color: #3a3a3a;
    }}

    tbody tr:hover {{
        background-color: #505050;
    }}

    td:nth-child(5) {{
        text-align: right;
        font-weight: bold;
    }}

    .note {{
        margin-top: 18px;
        color: #bbbbbb;
        font-size: 13px;
    }}
</style>
</head>
<body>
    <h1>Accuracy Comparison Table</h1>
    <p>
        Accuracy is calculated as Recall@5 because the current experiments evaluate retrieval performance.
        Answer-level accuracy will be added after LLM answer generation.
    </p>
    {table_html}
    <div class="note">
        Generated from outputs/metrics/*.json files.
    </div>
</body>
</html>
"""
    return html


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_accuracy_table()

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    html = make_html(df)
    OUTPUT_HTML.write_text(html, encoding="utf-8")

    print("\nAccuracy comparison table created.")
    print(f"CSV saved to: {OUTPUT_CSV}")
    print(f"HTML saved to: {OUTPUT_HTML}")

    webbrowser.open(OUTPUT_HTML.resolve().as_uri())


if __name__ == "__main__":
    main()