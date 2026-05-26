import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def build_judge_prompt(question: str, gold_answer: str, generated_answer: str) -> str:
    return f"""
Sen Türkçe hukuki soru-cevap değerlendiren katı ama adil bir değerlendiricisin.

Görevin:
Verilen model cevabının, gold cevaba anlam olarak uyup uymadığını değerlendir.

Puanlama:
- 1.0: Cevap gold cevapla aynı hukuki anlamı veriyor. Kelimeler birebir aynı olmak zorunda değil.
- 0.5: Cevap kısmen doğru ama eksik, belirsiz veya bazı önemli detayları atlıyor.
- 0.0: Cevap yanlış, alakasız, gold cevabı karşılamıyor veya "yeterli bilgi yok" diyerek cevaplamaktan kaçıyor.

Kurallar:
1. Sadece gold cevap ile model cevabını karşılaştır.
2. Gereksiz uzun açıklama yapma.
3. Çıktıyı sadece geçerli JSON olarak ver.
4. JSON dışında hiçbir metin yazma.

Soru:
{question}

Gold cevap:
{gold_answer}

Model cevabı:
{generated_answer}

JSON formatı:
{{
  "score": 0.0,
  "is_correct": false,
  "reason": "kısa gerekçe"
}}
""".strip()


def call_ollama_judge(
    prompt: str,
    model: str,
    ollama_url: str,
    temperature: float = 0.0,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> str:
    endpoint = f"{ollama_url.rstrip('/')}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 256,
        },
    }

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(endpoint, json=payload, timeout=180)

            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                time.sleep(sleep_seconds * attempt)
                continue

            data = response.json()
            return str(data.get("response", "")).strip()

        except Exception as exc:
            last_error = str(exc)
            time.sleep(sleep_seconds * attempt)

    raise RuntimeError(f"Ollama judge failed. Last error: {last_error}")


def extract_json(text: str) -> dict:
    """
    Ollama bazen JSON öncesi/sonrası ufak metin basarsa onu temizlemeye çalışır.
    """

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        return {
            "score": None,
            "is_correct": None,
            "reason": f"JSON parse failed. Raw output: {text[:300]}",
        }

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "score": None,
            "is_correct": None,
            "reason": f"JSON parse failed. Raw output: {text[:300]}",
        }


def normalize_score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    if score < 0:
        return 0.0
    if score > 1:
        return 1.0

    return score


def evaluate_file(
    input_path: Path,
    output_path: Path,
    model: str,
    ollama_url: str,
    limit: int | None = None,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    if limit is not None:
        df = df.head(limit).copy()

    scored_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Judging {input_path.name}"):
        question = str(row.get("question", ""))
        gold_answer = str(row.get("gold_answer", ""))
        generated_answer = str(row.get("generated_answer", ""))

        prompt = build_judge_prompt(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )

        raw_judge_output = call_ollama_judge(
            prompt=prompt,
            model=model,
            ollama_url=ollama_url,
        )

        parsed = extract_json(raw_judge_output)
        score = normalize_score(parsed.get("score"))

        scored_row = row.to_dict()
        scored_row["judge_model"] = model
        scored_row["judge_score"] = score
        scored_row["judge_is_correct"] = bool(score >= 0.5)
        scored_row["judge_reason"] = parsed.get("reason", "")
        scored_row["raw_judge_output"] = raw_judge_output

        scored_rows.append(scored_row)

        time.sleep(sleep_seconds)

    scored_df = pd.DataFrame(scored_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_path, index=False, encoding="utf-8")

    return scored_df


def summarize(scored_df: pd.DataFrame, method: str, retrieval_setup: str, prompt_name: str, llm: str) -> dict:
    scores = pd.to_numeric(scored_df["judge_score"], errors="coerce").fillna(0.0)

    return {
        "method": method,
        "retrieval_setup": retrieval_setup,
        "prompt": prompt_name,
        "llm": llm,
        "judge_metric": "LLM judge score",
        "num_questions": int(len(scored_df)),
        "average_score": float(scores.mean()),
        "average_score_percent": round(float(scores.mean()) * 100, 2),
        "strict_accuracy_score_1_only": round(float((scores == 1.0).mean()) * 100, 2),
        "relaxed_accuracy_score_at_least_0_5": round(float((scores >= 0.5).mean()) * 100, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated RAG answers using Ollama as an LLM judge."
    )

    parser.add_argument("--base-file", type=Path, required=True)
    parser.add_argument("--optimized-file", type=Path, required=True)

    parser.add_argument("--judge-model", type=str, default="qwen2.5:7b")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")

    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
    )

    args = parser.parse_args()

    base_scored_path = args.output_dir / "base_rag_judged.csv"
    optimized_scored_path = args.output_dir / "optimized_rag_judged.csv"
    summary_path = args.output_dir / "answer_judge_summary.csv"

    base_scored = evaluate_file(
        input_path=args.base_file,
        output_path=base_scored_path,
        model=args.judge_model,
        ollama_url=args.ollama_url,
        limit=args.limit,
    )

    optimized_scored = evaluate_file(
        input_path=args.optimized_file,
        output_path=optimized_scored_path,
        model=args.judge_model,
        ollama_url=args.ollama_url,
        limit=args.limit,
    )

    summary_rows = [
        summarize(
            scored_df=base_scored,
            method="Base RAG",
            retrieval_setup="Dense retrieval top-5",
            prompt_name="Base prompt",
            llm="qwen2.5:7b",
        ),
        summarize(
            scored_df=optimized_scored,
            method="Optimized RAG",
            retrieval_setup="BM25 retrieval top-5",
            prompt_name="Base prompt",
            llm="qwen2.5:7b",
        ),
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print("\nAnswer judge summary")
    print(summary_df)

    print(f"\nBase judged file saved to: {base_scored_path}")
    print(f"Optimized judged file saved to: {optimized_scored_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()