import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = clean_text(text)

    if not text:
        return []

    words = text.split()

    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = max(0, end - overlap)

    return chunks


def make_context_key(source_name: str, chunk_index: int, text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    safe_source = re.sub(r"[^a-zA-Z0-9ğüşöçıİĞÜŞÖÇ_-]+", "_", source_name)
    return f"{safe_source}_chunk_{chunk_index}_{digest}"


def load_text_files(input_dir: Path) -> list[dict]:
    supported_suffixes = {".txt", ".md"}

    rows = []

    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_suffixes:
            continue

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        rows.append(
            {
                "source_file": str(file_path),
                "source_name": file_path.stem,
                "text": text,
            }
        )

    return rows


def load_csv_file(input_csv: Path) -> list[dict]:
    df = pd.read_csv(input_csv)

    possible_text_cols = [
        "context",
        "retrieval_text",
        "text",
        "document",
        "content",
        "metin",
    ]

    text_col = None

    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            f"No text column found in {input_csv}. "
            f"Expected one of: {possible_text_cols}"
        )

    rows = []

    for idx, row in df.iterrows():
        source_name = str(
            row.get("kaynak")
            or row.get("source")
            or row.get("title")
            or row.get("document_id")
            or f"document_{idx}"
        )

        rows.append(
            {
                "source_file": str(input_csv),
                "source_name": source_name,
                "text": str(row[text_col]),
                "madde_no": row.get("madde_no", ""),
                "kanun_no": row.get("kanun_no", ""),
                "url": row.get("url", ""),
            }
        )

    return rows


def build_corpus(
    raw_rows: list[dict],
    chunk_size: int,
    overlap: int,
    chunk_strategy: str,
) -> pd.DataFrame:
    corpus_rows = []

    for raw in raw_rows:
        source_name = str(raw["source_name"])
        text = str(raw["text"])

        chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for i, chunk in enumerate(chunks, start=1):
            context_key = make_context_key(source_name, i, chunk)

            corpus_rows.append(
                {
                    "kaynak": source_name,
                    "madde_no": raw.get("madde_no", i),
                    "context_key": context_key,
                    "context": chunk,
                    "retrieval_text": f"{source_name}\n{chunk}",
                    "chunk_strategy": chunk_strategy,
                    "kanun_no": raw.get("kanun_no", ""),
                    "url": raw.get("url", raw.get("source_file", "")),
                }
            )

    return pd.DataFrame(corpus_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert custom documents or CSV into the standard RAG corpus format."
    )

    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)

    parser.add_argument("--chunk-size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--chunk-strategy", type=str, default="custom_word_chunk")

    args = parser.parse_args()

    if args.input_dir is None and args.input_csv is None:
        raise ValueError("You must provide either --input-dir or --input-csv.")

    if args.input_dir is not None and args.input_csv is not None:
        raise ValueError("Provide only one of --input-dir or --input-csv, not both.")

    if args.input_dir is not None:
        raw_rows = load_text_files(args.input_dir)
    else:
        raw_rows = load_csv_file(args.input_csv)

    if not raw_rows:
        raise ValueError("No documents found.")

    corpus = build_corpus(
        raw_rows=raw_rows,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        chunk_strategy=args.chunk_strategy,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(args.output_csv, index=False, encoding="utf-8")

    print("Custom corpus prepared.")
    print("Documents loaded:", len(raw_rows))
    print("Corpus chunks:", len(corpus))
    print("Output:", args.output_csv)
    print("Columns:", list(corpus.columns))


if __name__ == "__main__":
    main()