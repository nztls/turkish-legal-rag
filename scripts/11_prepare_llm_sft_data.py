import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ProjectPaths


def build_instruction(question: str, context: str) -> str:
    return f"""
Aşağıdaki hukuki kaynak metnine dayanarak soruya Türkçe cevap ver.

Kurallar:
- Sadece verilen kaynak metindeki bilgiye dayan.
- Cevabı kısa, açık ve hukuki ifadeye uygun yaz.
- Kaynakta olmayan bilgi ekleme.

Soru:
{question}

Kaynak metin:
{context}

Cevap:
""".strip()


def build_text(question: str, context: str, answer: str) -> str:
    instruction = build_instruction(question, context)

    return f"""<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{answer}
<|im_end|>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SFT JSONL data for LLM fine-tuning."
    )

    paths = ProjectPaths()

    parser.add_argument(
        "--train-csv",
        type=Path,
        default=paths.outputs_dir / "splits" / "benchmark_train.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.outputs_dir / "llm_sft_data",
    )

    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)

    rows = []

    for _, row in df.iterrows():
        question = str(row.get("soru", "")).strip()
        context = str(row.get("context", "")).strip()
        answer = str(row.get("cevap", "")).strip()

        if not question or not context or not answer:
            continue

        rows.append(
            {
                "row_id": row.get("row_id"),
                "question": question,
                "context": context,
                "answer": answer,
                "text": build_text(question, context, answer),
            }
        )

    train_rows, val_rows = train_test_split(
        rows,
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "train_sft.jsonl"
    val_path = args.output_dir / "val_sft.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("SFT data prepared.")
    print("Input train CSV:", args.train_csv)
    print("Total examples:", len(rows))
    print("Train examples:", len(train_rows))
    print("Validation examples:", len(val_rows))
    print("Saved train:", train_path)
    print("Saved val:", val_path)


if __name__ == "__main__":
    main()