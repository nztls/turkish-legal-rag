import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def read_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def message_to_qwen_text(messages: list[dict]) -> str:
    parts = []

    for msg in messages:
        role = msg.get("role", "").strip()
        content = msg.get("content", "").strip()

        if not role or not content:
            continue

        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    return "\n".join(parts)


def collect_eval_chunk_ids(rag_eval_json: Path | None, gold_benchmark_json: Path | None) -> set[str]:
    ids = set()

    if rag_eval_json and rag_eval_json.exists():
        rows = read_json(rag_eval_json)

        for row in rows:
            for chunk_id in row.get("gold_chunk_ids", []) or []:
                ids.add(str(chunk_id))

    if gold_benchmark_json and gold_benchmark_json.exists():
        rows = read_json(gold_benchmark_json)

        for row in rows:
            for source in row.get("gold_sources", []) or []:
                chunk_id = source.get("corpus_row_id") or source.get("source_id")

                if chunk_id:
                    ids.add(str(chunk_id))

    return ids


def get_training_chunk_id(row: dict) -> str:
    metadata = row.get("metadata", {}) or {}

    return str(
        metadata.get("chunk_id")
        or metadata.get("corpus_row_id")
        or metadata.get("source_id")
        or ""
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare friend llm.jsonl messages file for QLoRA SFT training."
    )

    parser.add_argument("--llm-jsonl", type=Path, required=True)
    parser.add_argument("--rag-eval-json", type=Path, default=None)
    parser.add_argument("--gold-benchmark-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    rows = read_jsonl(args.llm_jsonl)
    eval_chunk_ids = collect_eval_chunk_ids(
        rag_eval_json=args.rag_eval_json,
        gold_benchmark_json=args.gold_benchmark_json,
    )

    converted = []
    skipped_overlap = 0
    skipped_bad = 0

    for row in rows:
        chunk_id = get_training_chunk_id(row)

        if chunk_id and chunk_id in eval_chunk_ids:
            skipped_overlap += 1
            continue

        messages = row.get("messages", []) or []
        text = message_to_qwen_text(messages)

        if not text or "<|im_start|>assistant" not in text:
            skipped_bad += 1
            continue

        converted.append(
            {
                "id": row.get("id", ""),
                "chunk_id": chunk_id,
                "text": text,
            }
        )

    random.seed(args.seed)
    random.shuffle(converted)

    val_size = max(1, int(len(converted) * args.val_ratio))
    val_rows = converted[:val_size]
    train_rows = converted[val_size:]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "train_sft.jsonl"
    val_path = args.output_dir / "val_sft.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Friend LLM SFT data prepared.")
    print("Original rows:", len(rows))
    print("Eval chunk ids protected:", len(eval_chunk_ids))
    print("Skipped due to eval/gold overlap:", skipped_overlap)
    print("Skipped bad rows:", skipped_bad)
    print("Usable rows:", len(converted))
    print("Train rows:", len(train_rows))
    print("Val rows:", len(val_rows))
    print("Saved train:", train_path)
    print("Saved val:", val_path)


if __name__ == "__main__":
    main()