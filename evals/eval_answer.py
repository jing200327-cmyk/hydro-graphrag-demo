# evals/eval_answer.py
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.qa_pipeline import run_end_to_end_graphrag_qa
from src.utils.text import make_json_safe


DEFAULT_DATASET_PATH = PROJECT_ROOT / "evals" / "datasets" / "rag_mvp_eval_50.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_reports"


# =========================
# JSONL / 通用工具
# =========================

def resolve_default_dataset_path() -> Path:
    return DEFAULT_DATASET_PATH


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"测试集文件不存在：{path}\n"
            f"请确认文件是否位于 evals/datasets/ 下，或通过 --dataset 显式指定。"
        )

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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_safe(row), ensure_ascii=False) + "\n")


def as_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, tuple) or isinstance(value, set):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    return [str(value).strip()] if str(value).strip() else []


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []

    for item in items:
        item = str(item).strip()
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def infer_should_abstain(case: Dict[str, Any]) -> bool:
    if "should_abstain" in case:
        return bool(case.get("should_abstain"))

    if "answerable" in case:
        return not bool(case.get("answerable"))

    category = str(case.get("category", "")).lower()
    if "fallback" in category or "unanswerable" in category or "无法" in category:
        return True

    return False


def infer_should_retrieve(case: Dict[str, Any]) -> bool:
    if "should_retrieve" in case:
        return bool(case.get("should_retrieve"))

    if "answerable" in case:
        return bool(case.get("answerable"))

    if infer_should_abstain(case):
        return False

    return True


def extract_terms_from_expected_answer(expected_answer: str) -> List[str]:
    """
    当旧测试集只有 expected_answer，没有 expected_answer_points / expected_keywords 时，
    自动提取一些常见地质、水文地质关键词，避免 coverage_score 永远为 0。
    """
    if not expected_answer:
        return []

    candidate_terms = [
        "ZK", "钻孔", "层顶", "层底", "岩性", "渗透系数", "透水", "强透水", "较强透水",
        "中等透水", "弱透水", "极弱透水", "粉质黏土", "粉质粘土", "黏土", "粘土",
        "粉土", "粉砂", "细砂", "中砂", "粗砂", "砾砂", "卵石", "圆砾", "碎石",
        "淤泥", "未找到", "无法", "证据不足",
    ]

    terms = [term for term in candidate_terms if term in expected_answer]

    boreholes = re.findall(r"\bZK[-_]?\d+\b", expected_answer, flags=re.IGNORECASE)
    depths = re.findall(r"\d+(?:\.\d+)?\s*(?:m|米)", expected_answer, flags=re.IGNORECASE)

    return dedupe_keep_order(terms + boreholes + depths)


def normalize_eval_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    将旧测试集字段升级为统一字段。

    目标统一结构：
    {
      "id": "qa_001",
      "category": "fact",
      "question": "...",
      "answerable": true,
      "should_retrieve": true,
      "should_abstain": false,

      "gold_chunk_ids": [],
      "gold_chunk_types": [],

      "expected_answer": "...",
      "expected_answer_points": [],
      "must_contain": [],
      "must_not_contain": []
    }
    """
    question = str(case.get("question", "")).strip()
    if not question:
        raise ValueError(f"测试样本缺少 question 字段：{case}")

    category = str(case.get("category", case.get("type", "unknown"))).strip() or "unknown"
    should_abstain = infer_should_abstain(case)
    should_retrieve = infer_should_retrieve(case)
    answerable = bool(case.get("answerable", not should_abstain))

    gold_chunk_ids = dedupe_keep_order(
        as_list(case.get("gold_chunk_ids"))
        or as_list(case.get("expected_chunk_ids"))
        or as_list(case.get("source_chunk_ids"))
        or as_list(case.get("relevant_chunk_ids"))
    )

    gold_chunk_types = dedupe_keep_order(
        as_list(case.get("gold_chunk_types"))
        or as_list(case.get("expected_chunk_types"))
        or as_list(case.get("chunk_types"))
    )

    expected_answer = str(case.get("expected_answer", "") or "").strip()

    expected_keywords = dedupe_keep_order(
        as_list(case.get("expected_keywords"))
        + as_list(case.get("keywords"))
    )

    expected_answer_points = dedupe_keep_order(
        as_list(case.get("expected_answer_points"))
        + expected_keywords
    )

    if not expected_answer_points and expected_answer:
        expected_answer_points = extract_terms_from_expected_answer(expected_answer)

    must_contain = dedupe_keep_order(
        as_list(case.get("must_contain"))
        or expected_keywords
        or expected_answer_points
    )

    must_not_contain = dedupe_keep_order(as_list(case.get("must_not_contain")))

    # 兜底类问题，如果没有配置 must_not_contain，补充一些高风险确定性表达。
    # 注意：这里不放“渗透系数”本身，因为“无法判断渗透系数”是合理表达。
    if should_abstain and not must_not_contain:
        must_not_contain = [
            "渗透系数为",
            "k_value",
            "k_log10",
            "具体数值为",
            "可以确定为",
            "明确为",
        ]

    normalized = dict(case)
    normalized.update({
        "id": str(case.get("id", "")).strip(),
        "category": category,
        "question": question,
        "answerable": answerable,
        "should_retrieve": should_retrieve,
        "should_abstain": should_abstain,
        "gold_chunk_ids": gold_chunk_ids,
        "gold_chunk_types": gold_chunk_types,
        "expected_answer": expected_answer,
        "expected_answer_points": expected_answer_points,
        "must_contain": must_contain,
        "must_not_contain": must_not_contain,
    })

    return normalized


# =========================
# 答案规则指标
# =========================

def contains_any(text: str, items: List[str]) -> bool:
    return any(item in text for item in items if item)


def coverage_score(answer: str, expected_points: List[str]) -> float:
    """
    expected_answer_points 覆盖率。
    """
    if not expected_points:
        return 1.0

    hit_count = sum(1 for point in expected_points if point and point in answer)
    return round(hit_count / len(expected_points), 4)


def must_contain_score(answer: str, must_contain: List[str]) -> float:
    if not must_contain:
        return 1.0

    hit_count = sum(1 for item in must_contain if item and item in answer)
    return round(hit_count / len(must_contain), 4)


def must_not_contain_pass(answer: str, must_not_contain: List[str]) -> int:
    if not must_not_contain:
        return 1

    return int(not contains_any(answer, must_not_contain))


def answer_has_abstention_expression(answer: str) -> bool:
    abstain_terms = [
        "无法",
        "未找到",
        "未检索到",
        "没有找到",
        "资料中未见",
        "资料不足",
        "证据不足",
        "不能给出",
        "无法给出可靠判断",
        "无法确定",
        "不能确定",
        "当前资料无法支持",
        "根据当前资料无法",
    ]

    return contains_any(answer, abstain_terms)


def abstention_pass(
    answer: str,
    should_abstain: bool,
    fallback_triggered: bool,
) -> int:
    answer_abstained = answer_has_abstention_expression(answer) or fallback_triggered

    if should_abstain:
        return int(answer_abstained)

    return int(not answer_abstained)


def format_compliance_score(answer: str, expected_sections: Optional[List[str]] = None) -> float:
    """
    默认检查项目回答格式。
    如果你的 prompt_builder 没有强制这些标题，可以在测试集中为样本配置 expected_sections。
    """
    sections = expected_sections or ["【结论】", "【依据】", "【水文逻辑解释】", "【补充说明】"]

    if not sections:
        return 1.0

    hit_count = sum(1 for sec in sections if sec in answer)
    return round(hit_count / len(sections), 4)


# =========================
# 幻觉风险检测
# =========================

def chunk_to_text(chunk: Dict[str, Any]) -> str:
    text_fields = [
        "chunk_text",
        "text",
        "content",
        "page_content",
        "graph_context_text",
        "lithology",
        "description",
    ]

    parts = []
    for field in text_fields:
        value = chunk.get(field)
        if value:
            parts.append(str(value))

    metadata = chunk.get("metadata")
    if isinstance(metadata, dict):
        for key in ["chunk_text", "text", "content", "lithology", "description"]:
            if metadata.get(key):
                parts.append(str(metadata[key]))

    return "\n".join(parts)


def collect_retrieved_chunks(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    不只看 vector_search_results，还会合并 b_score_results / c_rerank_results / context_chunks 等。
    这样幻觉检测更贴近最终生成所依据的上下文。
    """
    all_chunks: List[Dict[str, Any]] = []

    candidate_keys = [
        "final_context_chunks",
        "context_chunks",
        "c_rerank_results",
        "b_score_results",
        "vector_search_results",
    ]

    for key in candidate_keys:
        value = result.get(key)
        if isinstance(value, list):
            all_chunks.extend([x for x in value if isinstance(x, dict)])

    # 去重
    seen = set()
    deduped = []
    for chunk in all_chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        text = chunk_to_text(chunk)
        fingerprint = chunk_id or text[:100]

        if fingerprint and fingerprint not in seen:
            seen.add(fingerprint)
            deduped.append(chunk)

    return deduped


def build_context_text(result: Dict[str, Any]) -> str:
    chunks = collect_retrieved_chunks(result)
    parts = [chunk_to_text(chunk) for chunk in chunks]

    for key in ["final_context", "context", "context_text", "prompt_context"]:
        if result.get(key):
            parts.append(str(result[key]))

    return "\n".join(parts)


def normalize_fact_token(token: str) -> str:
    token = str(token)
    token = token.strip()
    token = token.replace(" ", "")
    token = token.replace("米", "m")
    token = token.replace("Ｍ", "m")
    token = token.replace("～", "~")
    token = token.replace("—", "-")
    token = token.replace("－", "-")
    return token.lower()


def extract_fact_tokens(text: str) -> Dict[str, List[str]]:
    """
    抽取容易产生幻觉的强事实表达。
    """
    if not text:
        return {
            "borehole_ids": [],
            "depth_ranges": [],
            "depth_values": [],
            "permeability_values": [],
            "k_assignments": [],
            "layer_boundary_values": [],
        }

    patterns = {
        # ZK1, ZK01, ZK-1, ZK_1
        "borehole_ids": r"\bZK[-_]?\d+\b",

        # 12-18m, 12~18m, 12 至 18 m
        "depth_ranges": r"\d+(?:\.\d+)?\s*(?:-|~|～|至|到)\s*\d+(?:\.\d+)?\s*(?:m|米)",

        # 12m, 18.5米
        "depth_values": r"\d+(?:\.\d+)?\s*(?:m|米)",

        # 1.2e-4 m/s, 0.35 cm/s, 2.1 m/d
        "permeability_values": r"\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*(?:m/s|cm/s|m/d|cm/d)",

        # k_value=, k_log10=, log10(k)
        "k_assignments": r"(?:k_value|k_log10|log10\(k\))\s*[:=：]?\s*[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?",

        # 层顶 12m, 层底 18m
        "layer_boundary_values": r"(?:层顶|层底|顶板|底板|层顶埋深|层底埋深)\s*[:：为是]?\s*\d+(?:\.\d+)?\s*(?:m|米)",
    }

    result: Dict[str, List[str]] = {}

    for name, pattern in patterns.items():
        values = re.findall(pattern, text, flags=re.IGNORECASE)
        result[name] = dedupe_keep_order([normalize_fact_token(v) for v in values])

    return result


def hallucination_risk_check(
    question: str,
    answer: str,
    context_text: str,
    should_abstain: bool,
    fallback_triggered: bool,
) -> Tuple[int, List[str]]:
    """
    规则型幻觉风险检测。

    逻辑：
    1. 如果答案中出现强事实表达，但上下文中没有对应表达，标记风险。
    2. 问题中本身出现的实体不算幻觉，例如用户问 ZK999，答案说“未找到 ZK999”不算。
    3. 如果本应兜底，但答案没有兜底且给出强确定性事实，标记风险。
    """
    if not answer:
        return 0, []

    support_text = f"{question}\n{context_text}"
    answer_tokens = extract_fact_tokens(answer)
    support_tokens = extract_fact_tokens(support_text)

    reasons: List[str] = []

    for token_type, tokens in answer_tokens.items():
        support_set = set(support_tokens.get(token_type, []))

        for token in tokens:
            if token and token not in support_set:
                reasons.append(f"答案出现未被问题或上下文支持的 {token_type}: {token}")

    answer_abstained = answer_has_abstention_expression(answer) or fallback_triggered

    strong_assertion_patterns = [
        r"渗透系数为",
        r"岩性为",
        r"层顶.*为",
        r"层底.*为",
        r"可以确定",
        r"明确为",
        r"k_value\s*[:=：]",
        r"k_log10\s*[:=：]",
    ]

    has_strong_assertion = any(
        re.search(pattern, answer, flags=re.IGNORECASE)
        for pattern in strong_assertion_patterns
    )

    if should_abstain and has_strong_assertion and not answer_abstained:
        reasons.append("该样本应兜底，但答案给出了确定性事实判断")

    return int(bool(reasons)), dedupe_keep_order(reasons)


# =========================
# 单样本评估
# =========================

def evaluate_one_case(
    case: Dict[str, Any],
    raw_top_k: int,
    final_top_k: int,
    enable_llm: bool,
) -> Dict[str, Any]:
    normalized_case = normalize_eval_case(case)

    question = normalized_case["question"]
    should_abstain = bool(normalized_case.get("should_abstain", False))

    result = run_end_to_end_graphrag_qa(
        user_question=question,
        raw_top_k=raw_top_k,
        final_top_k=final_top_k,
        save_outputs=False,
        enable_llm=enable_llm,
    )

    answer = result.get("final_answer", "") or ""

    expected_points = as_list(normalized_case.get("expected_answer_points"))
    must_contain = as_list(normalized_case.get("must_contain"))
    must_not_contain = as_list(normalized_case.get("must_not_contain"))
    expected_sections = as_list(normalized_case.get("expected_sections"))

    fallback_triggered = bool(result.get("fallback_triggered", False))
    context_text = build_context_text(result)

    hallucination_flag, hallucination_reasons = hallucination_risk_check(
        question=question,
        answer=answer,
        context_text=context_text,
        should_abstain=should_abstain,
        fallback_triggered=fallback_triggered,
    )

    coverage = coverage_score(answer, expected_points)
    must_contain = must_contain_score(answer, must_contain)
    must_not_pass = must_not_contain_pass(answer, must_not_contain)
    abstain_pass = abstention_pass(
        answer=answer,
        should_abstain=should_abstain,
        fallback_triggered=fallback_triggered,
    )
    format_score = format_compliance_score(answer, expected_sections or None)

    # 规则综合分。
    # 对工程资料类 RAG，幻觉风险惩罚较高。
    overall_rule_score = (
        0.35 * coverage
        + 0.20 * must_contain
        + 0.15 * must_not_pass
        + 0.15 * abstain_pass
        + 0.05 * format_score
        + 0.10 * (1 - hallucination_flag)
    )

    overall_rule_score = max(0.0, min(1.0, round(overall_rule_score, 4)))

    retrieved_chunks = collect_retrieved_chunks(result)
    top_chunk_ids = [
        x.get("chunk_id")
        for x in retrieved_chunks[:5]
        if x.get("chunk_id")
    ]

    row = {
        "id": normalized_case.get("id"),
        "category": normalized_case.get("category"),
        "question": question,
        "answerable": normalized_case.get("answerable"),
        "should_retrieve": normalized_case.get("should_retrieve"),
        "should_abstain": should_abstain,

        "expected_answer": normalized_case.get("expected_answer"),
        "expected_answer_points": expected_points,
        "must_contain": as_list(normalized_case.get("must_contain")),
        "must_not_contain": must_not_contain,

        "answer": answer,
        "enable_llm": enable_llm,

        "fallback_triggered": fallback_triggered,
        "confidence": result.get("confidence"),

        "coverage_score": coverage,
        "must_contain_score": must_contain,
        "must_not_contain_pass": must_not_pass,
        "abstention_pass": abstain_pass,
        "format_compliance_score": format_score,
        "hallucination_risk_flag": hallucination_flag,
        "hallucination_risk_reasons": hallucination_reasons,
        "overall_rule_score": overall_rule_score,

        "top_chunks": top_chunk_ids,
        "context_char_length": len(context_text),
        "error": None,
    }

    return make_json_safe(row)


def evaluate_one_case_safely(
    case: Dict[str, Any],
    raw_top_k: int,
    final_top_k: int,
    enable_llm: bool,
    fail_fast: bool,
) -> Dict[str, Any]:
    try:
        return evaluate_one_case(
            case=case,
            raw_top_k=raw_top_k,
            final_top_k=final_top_k,
            enable_llm=enable_llm,
        )
    except Exception as exc:
        if fail_fast:
            raise

        normalized = normalize_eval_case(case)

        return make_json_safe({
            "id": normalized.get("id"),
            "category": normalized.get("category"),
            "question": normalized.get("question"),
            "answerable": normalized.get("answerable"),
            "should_retrieve": normalized.get("should_retrieve"),
            "should_abstain": normalized.get("should_abstain"),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })


# =========================
# 汇总指标
# =========================

def mean_metric(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(series) == 0:
        return None

    return round(float(series.mean()), 4)


def summarize_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "case_count": int(len(df)),
        "error_count": int(df["error"].notna().sum()) if "error" in df.columns else 0,
        "coverage_score_avg": mean_metric(df, "coverage_score"),
        "must_contain_score_avg": mean_metric(df, "must_contain_score"),
        "must_not_contain_pass_rate": mean_metric(df, "must_not_contain_pass"),
        "abstention_pass_rate": mean_metric(df, "abstention_pass"),
        "format_compliance_score_avg": mean_metric(df, "format_compliance_score"),
        "hallucination_risk_rate": mean_metric(df, "hallucination_risk_flag"),
        "overall_rule_score_avg": mean_metric(df, "overall_rule_score"),
    }

    if "should_abstain" in df.columns:
        fallback_df = df[df["should_abstain"] == True].copy()
        answerable_df = df[df["should_abstain"] == False].copy()

        summary["answerable_case_count"] = int(len(answerable_df))
        summary["fallback_case_count"] = int(len(fallback_df))

        summary["answerable_overall_rule_score_avg"] = mean_metric(answerable_df, "overall_rule_score")
        summary["fallback_abstention_pass_rate"] = mean_metric(fallback_df, "abstention_pass")
        summary["fallback_hallucination_risk_rate"] = mean_metric(fallback_df, "hallucination_risk_flag")

    # 分类型统计
    category_summary: Dict[str, Any] = {}
    if "category" in df.columns:
        for category, group in df.groupby("category"):
            category_summary[str(category)] = {
                "case_count": int(len(group)),
                "coverage_score_avg": mean_metric(group, "coverage_score"),
                "must_contain_score_avg": mean_metric(group, "must_contain_score"),
                "abstention_pass_rate": mean_metric(group, "abstention_pass"),
                "hallucination_risk_rate": mean_metric(group, "hallucination_risk_flag"),
                "overall_rule_score_avg": mean_metric(group, "overall_rule_score"),
            }

    summary["category_summary"] = category_summary

    return make_json_safe(summary)


# =========================
# CLI
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Answer / Generation / Fallback 离线测评脚本")

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(resolve_default_dataset_path()),
        help="测试集 JSONL 路径，默认指向 evals/datasets/rag_mvp_eval_50.jsonl",
    )
    parser.add_argument("--raw-top-k", type=int, default=30)
    parser.add_argument("--final-top-k", type=int, default=10)

    # 关键修改：
    # 默认启用 LLM，避免用户忘记加 --enable-llm 导致 final_answer 为空。
    parser.add_argument(
        "--enable-llm",
        dest="enable_llm",
        action="store_true",
        default=True,
        help="是否真实调用 LLM。现在默认启用。",
    )
    parser.add_argument(
        "--disable-llm",
        dest="enable_llm",
        action="store_false",
        help="禁用 LLM，仅用于调试检索或 pipeline。",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--export-normalized-dataset",
        type=str,
        default="",
        help="可选：导出升级后的统一字段测试集 JSONL 路径。",
    )
    parser.add_argument(
        "--only-normalize",
        action="store_true",
        help="只导出统一字段测试集，不运行测评。",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="遇到单条样本报错时立即终止。默认记录错误并继续。",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_jsonl(dataset_path)
    normalized_cases = [normalize_eval_case(case) for case in cases]

    if args.export_normalized_dataset:
        export_path = Path(args.export_normalized_dataset)
        write_jsonl(export_path, normalized_cases)
        print(f"[eval_answer] normalized dataset saved to: {export_path}")

    if args.only_normalize:
        return

    if not args.enable_llm:
        print(
            "[eval_answer] WARNING: 当前使用 --disable-llm，"
            "如果 pipeline 不生成 final_answer，则答案测评结果可能没有意义。"
        )

    rows = []
    for idx, case in enumerate(normalized_cases, start=1):
        print(
            f"[eval_answer] running {idx}/{len(normalized_cases)} | "
            f"{case.get('id')} | {case.get('category')} | {case.get('question')}"
        )

        row = evaluate_one_case_safely(
            case=case,
            raw_top_k=args.raw_top_k,
            final_top_k=args.final_top_k,
            enable_llm=args.enable_llm,
            fail_fast=args.fail_fast,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = summarize_metrics(df)

    detail_csv_path = output_dir / "answer_eval_detail.csv"
    detail_jsonl_path = output_dir / "answer_eval_detail.jsonl"
    summary_path = output_dir / "answer_eval_summary.json"

    df.to_csv(detail_csv_path, index=False, encoding="utf-8-sig")
    write_jsonl(detail_jsonl_path, rows)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Answer Eval Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDetail CSV saved to: {detail_csv_path}")
    print(f"Detail JSONL saved to: {detail_jsonl_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()