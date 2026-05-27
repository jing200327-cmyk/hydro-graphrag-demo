# evals/eval_answer.py
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.qa_pipeline import run_end_to_end_graphrag_qa
from src.utils.text import make_json_safe


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

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


def contains_all(text: str, items: list[str]) -> bool:
    return all(item in text for item in items if item)


def contains_any(text: str, items: list[str]) -> bool:
    return any(item in text for item in items if item)


def coverage_score(answer: str, expected_points: list[str]) -> float:
    if not expected_points:
        return 0.0

    hit_count = sum(1 for point in expected_points if point and point in answer)
    return round(hit_count / len(expected_points), 4)


def must_contain_score(answer: str, must_contain: list[str]) -> float:
    if not must_contain:
        return 1.0

    hit_count = sum(1 for item in must_contain if item and item in answer)
    return round(hit_count / len(must_contain), 4)


def must_not_contain_pass(answer: str, must_not_contain: list[str]) -> int:
    if not must_not_contain:
        return 1

    return int(not contains_any(answer, must_not_contain))


def abstention_pass(
    answer: str,
    should_abstain: bool,
    fallback_triggered: bool,
) -> int:
    abstain_terms = [
        "无法",
        "未找到",
        "未检索到",
        "证据不足",
        "不能给出",
        "无法给出可靠判断",
    ]

    answer_abstained = contains_any(answer, abstain_terms) or fallback_triggered

    if should_abstain:
        return int(answer_abstained)

    return int(not answer_abstained)


def format_compliance_score(answer: str) -> float:
    sections = ["【结论】", "【依据】", "【水文逻辑解释】", "【补充说明】"]
    hit_count = sum(1 for sec in sections if sec in answer)

    return round(hit_count / len(sections), 4)


def hallucination_risk_flag(answer: str, retrieved_chunks: list[dict[str, Any]]) -> int:
    """
    第一版仅做粗粒度风险标记：
    如果答案出现 k_value / log10(k) / 钻孔编号等强事实表达，
    但召回文本中没有对应表达，则标记为高风险。
    """

    context_text = "\n".join([
        str(x.get("chunk_text", "")) + "\n" + str(x.get("graph_context_text", ""))
        for x in retrieved_chunks
    ])

    risky_patterns = [
        r"k_value\s*[:=]",
        r"k_log10\s*[:=]",
        r"log10\(k\)",
        r"\d+\.?\d*\s*(m/s|cm/s|m/d)",
    ]

    for pattern in risky_patterns:
        answer_hit = re.search(pattern, answer, flags=re.IGNORECASE)
        context_hit = re.search(pattern, context_text, flags=re.IGNORECASE)

        if answer_hit and not context_hit:
            return 1

    return 0


def evaluate_one_case(
    case: dict[str, Any],
    raw_top_k: int,
    final_top_k: int,
    enable_llm: bool,
) -> dict[str, Any]:
    question = case["question"]

    result = run_end_to_end_graphrag_qa(
        user_question=question,
        raw_top_k=raw_top_k,
        final_top_k=final_top_k,
        save_outputs=False,
        enable_llm=enable_llm,
    )

    answer = result.get("final_answer", "") or ""

    expected_points = case.get("expected_answer_points", [])
    must_contain = case.get("must_contain", [])
    must_not_contain = case.get("must_not_contain", [])
    should_abstain = bool(case.get("should_abstain", False))

    row = {
        "id": case.get("id"),
        "question": question,
        "answer": answer,
        "should_abstain": should_abstain,
        "fallback_triggered": result.get("fallback_triggered", False),
        "confidence": result.get("confidence"),
        "coverage_score": coverage_score(answer, expected_points),
        "must_contain_score": must_contain_score(answer, must_contain),
        "must_not_contain_pass": must_not_contain_pass(answer, must_not_contain),
        "abstention_pass": abstention_pass(
            answer=answer,
            should_abstain=should_abstain,
            fallback_triggered=bool(result.get("fallback_triggered", False)),
        ),
        "format_compliance_score": format_compliance_score(answer),
        "hallucination_risk_flag": hallucination_risk_flag(
            answer=answer,
            retrieved_chunks=result.get("vector_search_results", []),
        ),
        "top_chunks": [
            x.get("chunk_id")
            for x in result.get("vector_search_results", [])[:5]
            if x.get("chunk_id")
        ],
    }

    row["overall_rule_score"] = round(
        0.35 * row["coverage_score"]
        + 0.20 * row["must_contain_score"]
        + 0.20 * row["must_not_contain_pass"]
        + 0.15 * row["abstention_pass"]
        + 0.10 * row["format_compliance_score"]
        - 0.30 * row["hallucination_risk_flag"],
        4,
    )

    return make_json_safe(row)


def summarize_metrics(df: pd.DataFrame) -> dict[str, Any]:
    summary = {
        "case_count": int(len(df)),
        "coverage_score_avg": round(float(df["coverage_score"].mean()), 4),
        "must_contain_score_avg": round(float(df["must_contain_score"].mean()), 4),
        "must_not_contain_pass_rate": round(float(df["must_not_contain_pass"].mean()), 4),
        "abstention_pass_rate": round(float(df["abstention_pass"].mean()), 4),
        "format_compliance_score_avg": round(float(df["format_compliance_score"].mean()), 4),
        "hallucination_risk_rate": round(float(df["hallucination_risk_flag"].mean()), 4),
        "overall_rule_score_avg": round(float(df["overall_rule_score"].mean()), 4),
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "evals" / "datasets" / "qa_eval_set.jsonl"),
    )
    parser.add_argument("--raw-top-k", type=int, default=30)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="是否真实调用 LLM。默认不加该参数时，如果 pipeline 未生成答案，评估意义有限。",
    )
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
        print(f"[eval_answer] running {case.get('id')} | {case.get('question')}")
        row = evaluate_one_case(
            case=case,
            raw_top_k=args.raw_top_k,
            final_top_k=args.final_top_k,
            enable_llm=args.enable_llm,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = summarize_metrics(df)

    detail_path = output_dir / "answer_eval_detail.csv"
    summary_path = output_dir / "answer_eval_summary.json"

    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Answer Eval Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDetail saved to: {detail_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()