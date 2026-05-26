import argparse
import json
import math
from pathlib import Path

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def rows_to_input_examples(rows: list[dict]) -> list[InputExample]:
    examples = []

    for row in rows:
        query = str(row["query"]).strip()
        document = str(row["document"]).strip()
        label = float(row["label"])

        if not query or not document:
            continue

        examples.append(
            InputExample(
                texts=[query, document],
                label=label,
            )
        )

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a cross-encoder reranker on Turkish legal QA pairs."
    )

    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("outputs/reranker_data/train_pairs.jsonl"),
        help="Path to train JSONL pairs.",
    )

    parser.add_argument(
        "--val-path",
        type=Path,
        default=Path("outputs/reranker_data/val_pairs.jsonl"),
        help="Path to validation JSONL pairs.",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        help="Base cross-encoder reranker model.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/models/legal_reranker"),
        help="Directory where the fine-tuned reranker will be saved.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda or cpu. If omitted, sentence-transformers decides automatically.",
    )

    args = parser.parse_args()

    print("Train path:", args.train_path)
    print("Validation path:", args.val_path)
    print("Base model:", args.base_model)
    print("Output directory:", args.output_dir)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.learning_rate)
    print("Device:", args.device)

    train_rows = read_jsonl(args.train_path)
    val_rows = read_jsonl(args.val_path)

    train_examples = rows_to_input_examples(train_rows)
    val_examples = rows_to_input_examples(val_rows)

    print(f"\nTrain examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    if len(train_examples) == 0:
        raise ValueError("No train examples found.")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    warmup_steps = math.ceil(
        len(train_dataloader) * args.epochs * args.warmup_ratio
    )

    print("Warmup steps:", warmup_steps)

    model = CrossEncoder(
        args.base_model,
        num_labels=1,
        device=args.device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(args.output_dir),
        show_progress_bar=True,
    )

    # Save an additional HuggingFace-compatible copy.
    # This folder can be loaded later by CrossEncoder(path).
    hf_output_dir = args.output_dir / "hf_model"
    hf_output_dir.mkdir(parents=True, exist_ok=True)

    model.model.save_pretrained(str(hf_output_dir), safe_serialization=True)
    model.tokenizer.save_pretrained(str(hf_output_dir))

    metadata = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
    }

    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nFine-tuned reranker saved to: {args.output_dir}")
    print(f"HuggingFace-compatible model saved to: {hf_output_dir}")


if __name__ == "__main__":
    main()