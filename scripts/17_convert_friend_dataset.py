import argparse
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: dict, key: str, default=""):
    value = d.get(key, default)

    if value is None:
        return default

    return value


def convert_corpus(corpus_jsonl: Path, output_csv: Path) -> pd.DataFrame:
    rows = read_jsonl(corpus_jsonl)
    converted = []

    for item in rows:
        metadata = item.get("metadata", {}) or {}

        chunk_id = safe_get(item, "id")
        text = safe_get(item, "text")
        title = safe_get(item, "title")

        source = safe_get(metadata, "source")
        source_file = safe_get(metadata, "source_file")
        category = safe_get(metadata, "category")
        semantic_topic = safe_get(metadata, "semantic_topic")
        citation_label = safe_get(metadata, "citation_label")
        law_no = safe_get(metadata, "law_no", "")
        article_no = safe_get(metadata, "article_no", "")

        madde_no = article_no or safe_get(metadata, "chunk_index", "") or chunk_id

        retrieval_parts = [
            str(title),
            str(category),
            str(semantic_topic),
            str(citation_label),
            str(text),
        ]

        retrieval_text = "\n".join(
            part for part in retrieval_parts
            if part and part != "None"
        )

        converted.append(
            {
                "kaynak": citation_label or title or source or source_file,
                "madde_no": madde_no,
                "context_key": chunk_id,
                "context": text,
                "retrieval_text": retrieval_text,
                "chunk_strategy": "friend_jsonl_chunk",
                "kanun_no": law_no,
                "url": source_file,
                "title": title,
                "source": source,
                "category": category,
                "semantic_topic": semantic_topic,
                "citation_label": citation_label,
            }
        )

    df = pd.DataFrame(converted)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print("Corpus converted.")
    print("Input:", corpus_jsonl)
    print("Rows:", len(df))
    print("Output:", output_csv)
    print("Columns:", list(df.columns))

    return df


def convert_rag_eval_benchmark(
    rag_eval_json: Path,
    corpus_df: pd.DataFrame,
    output_csv: Path,
) -> pd.DataFrame:
    rows = read_json(rag_eval_json)

    corpus_by_key = {
        str(row["context_key"]): row
        for _, row in corpus_df.iterrows()
    }

    converted = []

    for item in rows:
        query_id = safe_get(item, "query_id")
        query = safe_get(item, "query")
        answer = safe_get(item, "gold_answer_extract")

        gold_chunk_ids = item.get("gold_chunk_ids", []) or []
        first_gold = str(gold_chunk_ids[0]) if gold_chunk_ids else ""

        corpus_row = corpus_by_key.get(first_gold, {})

        converted.append(
            {
                "row_id": query_id,
                "soru": query,
                "cevap": answer,
                "context": corpus_row.get("context", ""),
                "kaynak": corpus_row.get("kaynak", safe_get(item, "source")),
                "Score": 1,
                "source_origin": safe_get(item, "source"),
                "question_template": "friend_rag_eval",
                "selection_score": "",
                "answer_in_chunk": "",
                "chunk_strategy": corpus_row.get("chunk_strategy", "friend_jsonl_chunk"),
                "chunk_len": len(str(corpus_row.get("context", ""))),
                "context_key": first_gold,
                "question_key": query_id,
                "kanun_adi": safe_get(item, "title"),
                "madde_nolari_soru": "",
                "madde_nolari_context": "",
                "madde_no": corpus_row.get("madde_no", first_gold),
                "bolum_fasil": safe_get(item, "category"),
                "heading_count": "",
                "answer_support_overlap": "",
                "question_context_overlap": "",
                "question_answer_overlap": "",
                "answer_in_context_substring": "",
                "open_ended_question": "",
                "answer_mentions_other_source": "",
                "context_starts_mid_sentence": "",
                "score_valid": True,
                "manual_review_reason": "",
                "gold_chunk_ids": json.dumps(gold_chunk_ids, ensure_ascii=False),
                "gold_citation_labels": json.dumps(
                    item.get("gold_citation_labels", []),
                    ensure_ascii=False,
                ),
            }
        )

    df = pd.DataFrame(converted)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    matched = df["context"].astype(str).str.len().gt(0).sum()

    print("\nRAG eval benchmark converted.")
    print("Input:", rag_eval_json)
    print("Rows:", len(df))
    print("Gold context matched:", matched, "/", len(df))
    print("Output:", output_csv)

    return df


def convert_gold_benchmark(
    gold_benchmark_json: Path,
    corpus_df: pd.DataFrame,
    output_csv: Path,
) -> pd.DataFrame:
    rows = read_json(gold_benchmark_json)

    corpus_by_key = {
        str(row["context_key"]): row
        for _, row in corpus_df.iterrows()
    }

    converted = []

    for item in rows:
        question_id = safe_get(item, "question_id")
        question = safe_get(item, "question")
        answer = safe_get(item, "verified_answer")

        gold_sources = item.get("gold_sources", []) or []
        first_source = gold_sources[0] if gold_sources else {}

        context_key = (
            first_source.get("corpus_row_id")
            or first_source.get("source_id")
            or ""
        )

        context_key = str(context_key)
        corpus_row = corpus_by_key.get(context_key, {})

        converted.append(
            {
                "row_id": question_id,
                "soru": question,
                "cevap": answer,
                "context": corpus_row.get("context", ""),
                "kaynak": corpus_row.get("kaynak", first_source.get("source", "")),
                "Score": 1,
                "source_origin": first_source.get("source", ""),
                "question_template": "friend_gold_benchmark",
                "selection_score": "",
                "answer_in_chunk": "",
                "chunk_strategy": corpus_row.get("chunk_strategy", "friend_jsonl_chunk"),
                "chunk_len": len(str(corpus_row.get("context", ""))),
                "context_key": context_key,
                "question_key": question_id,
                "kanun_adi": first_source.get("law_name", ""),
                "madde_nolari_soru": first_source.get("article_no", ""),
                "madde_nolari_context": first_source.get("article_no", ""),
                "madde_no": corpus_row.get("madde_no", first_source.get("article_no", context_key)),
                "bolum_fasil": first_source.get("section", ""),
                "heading_count": "",
                "answer_support_overlap": "",
                "question_context_overlap": "",
                "question_answer_overlap": "",
                "answer_in_context_substring": "",
                "open_ended_question": "",
                "answer_mentions_other_source": "",
                "context_starts_mid_sentence": "",
                "score_valid": True,
                "manual_review_reason": "",
                "gold_sources_json": json.dumps(gold_sources, ensure_ascii=False),
            }
        )

    df = pd.DataFrame(converted)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    matched = df["context"].astype(str).str.len().gt(0).sum()

    print("\nGold benchmark converted.")
    print("Input:", gold_benchmark_json)
    print("Rows:", len(df))
    print("Gold context matched:", matched, "/", len(df))
    print("Output:", output_csv)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert friend's JSON/JSONL legal dataset into this project's standard RAG format."
    )

    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--rag-eval-json", type=Path, default=None)
    parser.add_argument("--gold-benchmark-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus_csv = args.output_dir / "corpus.csv"
    rag_eval_csv = args.output_dir / "benchmark_rag_eval.csv"
    gold_benchmark_csv = args.output_dir / "benchmark_gold.csv"

    corpus_df = convert_corpus(
        corpus_jsonl=args.corpus_jsonl,
        output_csv=corpus_csv,
    )

    if args.rag_eval_json is not None:
        convert_rag_eval_benchmark(
            rag_eval_json=args.rag_eval_json,
            corpus_df=corpus_df,
            output_csv=rag_eval_csv,
        )

    if args.gold_benchmark_json is not None:
        convert_gold_benchmark(
            gold_benchmark_json=args.gold_benchmark_json,
            corpus_df=corpus_df,
            output_csv=gold_benchmark_csv,
        )

    print("\nDone.")
    print("Converted corpus:", corpus_csv)

    if args.rag_eval_json is not None:
        print("Converted RAG eval benchmark:", rag_eval_csv)

    if args.gold_benchmark_json is not None:
        print("Converted gold benchmark:", gold_benchmark_csv)


if __name__ == "__main__":
    main()