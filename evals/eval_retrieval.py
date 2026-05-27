# evals/eval_retrieval.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.qa_pipeline import run_end_to_end_graphrag_qa
from src.utils.text import make_json_safe


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败：{path}, line={line_no}") from exc

    return rows


def get_ranked_chunk_ids(result: Dict[str, Any], stage: str = "vector") -> List[str]:
    if stage == "vector":
        chunks = result.get("vector_search_results", [])
        return [x.get("chunk_id") for x in chunks if x.get("chunk_id")]

    if stage == "b":
        chunks = result.get("b_score_results", [])
        return [x.get("chunk_id") for x in chunks if x.get("chunk_id")]

    if stage == "c":
        chunks = result.get("c_rerank_results", [])
        return [x.get("chunk_id") for x in chunks if x.get("chunk_id")]

    raise ValueError(f"未知 stage：{stage}")


def hit_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> int:
    if not gold_ids:
        return 0
    return int(any(cid in gold_ids for cid in ranked_ids[:k]))


def recall_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    hits = len(set(ranked_ids[:k]) & gold_ids)
    return hits / len(gold_ids)


def mrr_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if not gold_ids:
        return 0.0

    for idx, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_ids:
            return 1.0 / idx

    return 0.0


def chunk_type_hit_at_k(
    result: Dict[str, Any],
    gold_chunk_types: Set[str],
    k: int,
) -> int:
    if not gold_chunk_types:
        return 0

    chunks = result.get("vector_search_results", [])[:k]
    retrieved_types = {x.get("chunk_type") for x in chunks if x.get("chunk_type")}
    return int(bool(retrieved_types & gold_chunk_types))


def evaluate_one_case(
    case: Dict[str, Any],
    raw_top_k: int,
    final_top_k: int,
) -> Dict[str, Any]:
    question = case["question"]
    gold_ids = set(case.get("gold_chunk_ids", []))
    gold_types = set(case.get("gold_chunk_types", []))

    result = run_end_to_end_graphrag_qa(
        user_question=question,
        raw_top_k=raw_top_k,
        final_top_k=final_top_k,
        save_outputs=False,
        enable_llm=False,
    )

    vector_ids = get_ranked_chunk_ids(result, stage="vector")
    b_ids = get_ranked_chunk_ids(result, stage="b")
    c_ids = get_ranked_chunk_ids(result, stage="c")

    row = {
        "id": case.get("id"),
        "question": question,
        "should_retrieve": case.get("should_retrieve", True),
        "gold_chunk_ids": list(gold_ids),
        "vector_top_ids": vector_ids[:10],
        "b_top_ids": b_ids[:10],
        "c_top_ids": c_ids[:10],
        "vector_hit@1": hit_at_k(vector_ids, gold_ids, 1),
        "vector_hit@3": hit_at_k(vector_ids, gold_ids, 3),
        "vector_hit@5": hit_at_k(vector_ids, gold_ids, 5),
        "vector_hit@10": hit_at_k(vector_ids, gold_ids, 10),
        "vector_recall@3": recall_at_k(vector_ids, gold_ids, 3),
        "vector_recall@5": recall_at_k(vector_ids, gold_ids, 5),
        "vector_recall@10": recall_at_k(vector_ids, gold_ids, 10),
        "vector_mrr@10": mrr_at_k(vector_ids, gold_ids, 10),
        "b_hit@3": hit_at_k(b_ids, gold_ids, 3),
        "b_hit@5": hit_at_k(b_ids, gold_ids, 5),
        "b_mrr@10": mrr_at_k(b_ids, gold_ids, 10),
        "c_hit@3": hit_at_k(c_ids, gold_ids, 3),
        "c_hit@5": hit_at_k(c_ids, gold_ids, 5),
        "c_mrr@10": mrr_at_k(c_ids, gold_ids, 10),
        "chunk_type_hit@5": chunk_type_hit_at_k(result, gold_types, 5),
        "confidence": result.get("confidence"),
        "fallback_triggered": result.get("fallback_triggered"),
    }

    return make_json_safe(row)


def summarize_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    metric_cols = [
        col for col in df.columns
        if any(key in col for key in ["hit@", "recall@", "mrr@", "chunk_type_hit"])
    ]

    summary = {}

    for col in metric_cols:
        summary[col] = round(float(df[col].mean()), 4)

    summary["case_count"] = int(len(df))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "evals" / "datasets" / "retrieval_gold.jsonl"),
    )
    parser.add_argument("--raw-top-k", type=int, default=30)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "eval_reports"),
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_jsonl(dataset_path)

    rows = []
    for case in cases:
        print(f"[eval_retrieval] running {case.get('id')} | {case.get('question')}")
        row = evaluate_one_case(
            case=case,
            raw_top_k=args.raw_top_k,
            final_top_k=args.final_top_k,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = summarize_metrics(df)

    detail_path = output_dir / "retrieval_eval_detail.csv"
    summary_path = output_dir / "retrieval_eval_summary.json"

    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Retrieval Eval Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDetail saved to: {detail_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()