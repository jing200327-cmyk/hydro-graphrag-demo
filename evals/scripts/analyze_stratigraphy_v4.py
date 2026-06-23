from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[2]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.pipeline.qa_pipeline import run_end_to_end_graphrag_qa  # noqa: E402
from src.retrieval.stratigraphy_retriever import STRATIGRAPHY_ROUTE  # noqa: E402
from src.utils.text import make_json_safe, safe_text, unique_keep_order  # noqa: E402


DEFAULT_DATASET = PROJECT_DIR / "evals" / "datasets" / "rag_mvp_eval_50_stratigraphy_v4.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "evals" / "results" / "v4_stratigraphy"
DEFAULT_EVAL_DETAILS = DEFAULT_OUTPUT_DIR / "retrieval_v2_rankgpt_off_details.jsonl"
DEFAULT_V3_DETAILS = PROJECT_DIR / "evals" / "reports_v3" / "retrieval_v3_rankgpt_off_details.jsonl"

STRAT_SUBTYPE = "stratigraphy_fact_query"
LAYER_RE = re.compile(r"\bCHGC\d+_\d+\b", re.IGNORECASE)
BOREHOLE_RE = re.compile(r"\bCHGC\d+\b", re.IGNORECASE)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
            rows.append(row)
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_safe(row), ensure_ascii=False) + "\n")


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(value), f, ensure_ascii=False, indent=2)


def metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def is_strat_case(row: Dict[str, Any]) -> bool:
    return safe_text(metadata(row).get("subtype")) == STRAT_SUBTYPE


def percent(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100.0, 2)


def id_set(values: Iterable[Any]) -> Set[str]:
    return {safe_text(value).upper() for value in values if safe_text(value)}


def chunk_layer_ids(chunks: List[Dict[str, Any]]) -> List[str]:
    values: List[str] = []
    for chunk in chunks:
        props = chunk.get("source_props") or {}
        layer_id = safe_text(props.get("layer_id"))
        if not layer_id:
            text = safe_text(chunk.get("chunk_id"))
            match = LAYER_RE.search(text)
            layer_id = match.group(0) if match else ""
        if layer_id:
            values.append(layer_id)
    return unique_keep_order(values)


def chunk_borehole_ids(chunks: List[Dict[str, Any]]) -> List[str]:
    values: List[str] = []
    for chunk in chunks:
        props = chunk.get("source_props") or {}
        borehole_id = safe_text(props.get("borehole_id"))
        if borehole_id:
            values.append(borehole_id)
    return unique_keep_order(values)


def normalize_number_text(value: Any) -> List[str]:
    text = safe_text(value)
    if not text:
        return []
    values = [text]
    try:
        number = float(text)
    except Exception:
        return values
    values.append(f"{number:g}")
    values.append(f"{number:.1f}")
    return unique_keep_order(values)


def contains_any(text: str, values: Iterable[Any]) -> bool:
    candidates = [safe_text(value) for value in values if safe_text(value)]
    return any(candidate in text for candidate in candidates)


def answer_has_required_table_fields(answer: str) -> bool:
    required = ["钻孔", "X", "Y", "地面高程", "对应分层"]
    return all(item in answer for item in required)


def answer_contains_ground_truth_fields(answer: str, ground_truth: List[Dict[str, Any]]) -> bool:
    for row in ground_truth:
        checks = [
            [row.get("borehole_id")],
            [row.get("layer_id")],
            normalize_number_text(row.get("top_depth")),
            normalize_number_text(row.get("bottom_depth")),
            normalize_number_text(row.get("thickness")),
            [row.get("geologic_period")],
            [row.get("geologic_epoch")],
            [row.get("strat_group")],
            [row.get("strat_member")],
            [row.get("lithology_major_v2")],
            [row.get("lithology_minor_v3")],
        ]
        for candidates in checks:
            if candidates and not contains_any(answer, candidates):
                return False
    return True


def parse_answer_ids(answer: str) -> Dict[str, Set[str]]:
    layer_ids = {match.group(0).upper() for match in LAYER_RE.finditer(answer)}
    borehole_ids = {
        match.group(0).upper()
        for match in BOREHOLE_RE.finditer(answer)
        if f"{match.group(0).upper()}_" not in layer_ids
    }
    for layer_id in layer_ids:
        borehole_ids.add(layer_id.split("_", 1)[0])
    return {"layer_ids": layer_ids, "borehole_ids": borehole_ids}


def evaluate_strat_case(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = metadata(row)
    question = safe_text(row.get("question"))
    ground_truth = meta.get("stratigraphy_ground_truth") or []
    expected_layer_ids = id_set(meta.get("ground_truth_layer_ids") or [])
    expected_borehole_ids = id_set(meta.get("ground_truth_borehole_ids") or [])

    result = run_end_to_end_graphrag_qa(
        user_question=question,
        raw_top_k=30,
        final_top_k=10,
        save_outputs=False,
        enable_llm=False,
        enable_rankgpt=False,
    )

    chunks = result.get("vector_search_results") or []
    actual_layer_ids = id_set(chunk_layer_ids(chunks))
    actual_borehole_ids = id_set(chunk_borehole_ids(chunks))

    correct_layer_ids = expected_layer_ids & actual_layer_ids
    retrieved_count = len(actual_layer_ids)
    ground_truth_count = len(expected_layer_ids)

    recall = len(correct_layer_ids) / ground_truth_count if ground_truth_count else 0.0
    precision = len(correct_layer_ids) / retrieved_count if retrieved_count else 0.0

    answer = safe_text(result.get("final_answer") or result.get("direct_answer"))
    answer_ids = parse_answer_ids(answer)
    extra_layers = sorted(answer_ids["layer_ids"] - expected_layer_ids)
    extra_boreholes = sorted(answer_ids["borehole_ids"] - expected_borehole_ids)

    answer_field_complete = answer_has_required_table_fields(answer) and answer_contains_ground_truth_fields(
        answer,
        ground_truth,
    )
    no_hallucination = not extra_layers and not extra_boreholes
    route_hit = (
        safe_text(result.get("retrieval_route")) == STRATIGRAPHY_ROUTE
        and safe_text(result.get("rerank_route")) == "stratigraphy_rule"
        and result.get("rankgpt_enabled") is False
        and not result.get("deepseek_usage")
    )

    badcase_types: List[str] = []
    if not route_hit:
        badcase_types.append("route_miss")
    if route_hit and retrieved_count == 0:
        badcase_types.append("retrieval_empty")
    if retrieved_count > 0 and recall < 1 and precision == 1:
        badcase_types.append("retrieval_partial")
    if retrieved_count > 0 and precision < 1:
        badcase_types.append("retrieval_noise")
    if not answer_field_complete:
        badcase_types.append("answer_format_error")
    if not no_hallucination:
        badcase_types.append("hallucination")

    return {
        "case_id": row.get("id"),
        "question": question,
        "is_stratigraphy_case": True,
        "stratigraphy_dimension": meta.get("stratigraphy_dimension"),
        "stratigraphy_term": meta.get("stratigraphy_term"),
        "expected_route": STRATIGRAPHY_ROUTE,
        "actual_route": result.get("retrieval_route"),
        "actual_rerank_route": result.get("rerank_route"),
        "route_hit": route_hit,
        "ground_truth_layer_count": ground_truth_count,
        "actual_layer_count": retrieved_count,
        "correct_layer_count": len(correct_layer_ids),
        "retrieval_recall": round(recall, 4),
        "retrieval_precision": round(precision, 4),
        "answer_field_complete": answer_field_complete,
        "no_hallucination": no_hallucination,
        "extra_layer_ids": extra_layers,
        "extra_borehole_ids": extra_boreholes,
        "expected_layer_ids": sorted(expected_layer_ids),
        "actual_layer_ids": sorted(actual_layer_ids),
        "expected_borehole_ids": sorted(expected_borehole_ids),
        "actual_borehole_ids": sorted(actual_borehole_ids),
        "final_answer": answer,
        "badcase_types": badcase_types,
    }


def regression_reason(
    row: Dict[str, Any],
    v3_row: Optional[Dict[str, Any]],
) -> List[str]:
    reasons: List[str] = []

    if row.get("error"):
        reasons.append("runtime_error")

    if safe_text(row.get("actual_route")) == STRATIGRAPHY_ROUTE:
        reasons.append("route_false_positive")

    if v3_row is None:
        return reasons

    question_type = safe_text(row.get("question_type"))
    if v3_row.get("route_correct") is True and row.get("route_correct") is False:
        reasons.append("route_regression")

    if question_type == "fact_query":
        if v3_row.get("effective_hit_layer_top10") is True and row.get("effective_hit_layer_top10") is False:
            reasons.append("fact_retrieval_regression")

    if question_type == "fallback_unanswerable":
        if v3_row.get("fallback_triggered") is True and row.get("fallback_triggered") is not True:
            reasons.append("fallback_regression")

    if question_type not in {"fact_query", "fallback_unanswerable", "causal_explanation"}:
        if v3_row.get("keyword_hit_top10") is True and row.get("keyword_hit_top10") is False:
            reasons.append("keyword_regression")

    if question_type == "causal_explanation":
        for key in ["causal_feature_value_hit", "causal_permeability_effect_hit", "causal_mechanism_hit"]:
            if v3_row.get(key) is True and row.get(key) is False:
                reasons.append(f"{key}_regression")

    return unique_keep_order(reasons)


def build_badcase_rows(
    strat_results: List[Dict[str, Any]],
    regression_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    badcases: List[Dict[str, Any]] = []

    for row in strat_results:
        for badcase_type in row.get("badcase_types") or []:
            badcases.append(
                {
                    "case_id": row.get("case_id"),
                    "question": row.get("question"),
                    "badcase_type": badcase_type,
                    "expected": (
                        f"route={STRATIGRAPHY_ROUTE}, "
                        f"layers={row.get('ground_truth_layer_count')}"
                    ),
                    "actual": (
                        f"route={row.get('actual_route')}, "
                        f"rerank={row.get('actual_rerank_route')}, "
                        f"layers={row.get('actual_layer_count')}, "
                        f"recall={row.get('retrieval_recall')}, "
                        f"precision={row.get('retrieval_precision')}"
                    ),
                    "suspected_reason": suspected_reason(badcase_type),
                    "suggested_fix": suggested_fix(badcase_type),
                    "related_file": related_file(badcase_type),
                    "suggested_code_location": suggested_location(badcase_type),
                }
            )

    for row in regression_rows:
        for reason in row.get("regression_reasons") or []:
            badcases.append(
                {
                    "case_id": row.get("id"),
                    "question": row.get("question"),
                    "badcase_type": "regression" if reason != "route_false_positive" else "route_false_positive",
                    "expected": f"v3_route={row.get('v3_actual_route')}",
                    "actual": f"v4_route={row.get('actual_route')}, reason={reason}",
                    "suspected_reason": "新增 Stratigraphy 路由误伤或原有链路指标下降。",
                    "suggested_fix": "检查 is_stratigraphy_query 的排除词、事实返回词和词表匹配边界。",
                    "related_file": "src/retrieval/stratigraphy_retriever.py",
                    "suggested_code_location": "is_stratigraphy_query()",
                }
            )

    return badcases


def suspected_reason(badcase_type: str) -> str:
    mapping = {
        "route_miss": "地层词表未命中、事实返回关键词不足，或问题被推理关键词排除。",
        "retrieval_empty": "路由命中但 Cypher 条件没有返回 LithologyLayer。",
        "retrieval_partial": "Cypher 条件、LIMIT 或词表匹配导致部分 layer_id 漏召回。",
        "retrieval_noise": "字段条件过宽或多维条件组合不准确，混入非目标层位。",
        "answer_format_error": "检索正确但 Markdown 表格没有完整呈现纪、世、组、段、厚度等字段。",
        "hallucination": "答案中出现 ground truth 之外的钻孔或层位编号。",
    }
    return mapping.get(badcase_type, "需要结合单条结果进一步定位。")


def suggested_fix(badcase_type: str) -> str:
    mapping = {
        "route_miss": "补充词表加载字段或调整事实返回关键词，但不要放宽推理问题排除规则。",
        "retrieval_empty": "核对 LithologyLayer 的纪、世、组、段属性值和 Strat* 节点词表是否一致。",
        "retrieval_partial": "检查 retrieve_stratigraphy_exact() 的 LIMIT、排序和多字段过滤条件。",
        "retrieval_noise": "收紧 Cypher WHERE 条件，确保命中的每个字段都参与过滤。",
        "answer_format_error": "扩展 build_stratigraphy_direct_answer() 表格列，显式展示层位 ID、深度、厚度、纪、世、组、段、岩性。",
        "hallucination": "保持 direct_answer 只遍历检索 chunks，不拼接非检索来源的实体。",
    }
    return mapping.get(badcase_type, "补充单测后再修复。")


def related_file(badcase_type: str) -> str:
    if badcase_type in {"route_miss", "route_false_positive"}:
        return "src/retrieval/stratigraphy_retriever.py"
    if badcase_type in {"retrieval_empty", "retrieval_partial", "retrieval_noise"}:
        return "src/retrieval/stratigraphy_retriever.py"
    if badcase_type in {"answer_format_error", "hallucination"}:
        return "src/retrieval/stratigraphy_retriever.py"
    return "src/pipeline/qa_pipeline.py"


def suggested_location(badcase_type: str) -> str:
    if badcase_type in {"route_miss", "route_false_positive"}:
        return "is_stratigraphy_query() / _load_stratigraphy_vocab()"
    if badcase_type in {"retrieval_empty", "retrieval_partial", "retrieval_noise"}:
        return "retrieve_stratigraphy_exact()"
    if badcase_type in {"answer_format_error", "hallucination"}:
        return "build_stratigraphy_direct_answer()"
    return "run_end_to_end_graphrag_qa()"


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(safe_text(value).replace("\n", "<br>") for value in row) + " |")
    return "\n".join(lines)


def write_reports(
    output_dir: Path,
    dataset_rows: List[Dict[str, Any]],
    strat_results: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    badcases: List[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(
        [
            {
                **row,
                "analysis_scope": "stratigraphy"
                if any(item.get("case_id") == row.get("id") for item in strat_results)
                else "regression",
            }
            for row in eval_rows
        ],
        output_dir / "v4_stratigraphy_eval_results.jsonl",
    )
    save_json(metrics, output_dir / "v4_stratigraphy_metrics.json")

    badcase_rows = [
        [
            item.get("case_id"),
            item.get("question"),
            item.get("badcase_type"),
            item.get("expected"),
            item.get("actual"),
            item.get("suspected_reason"),
            item.get("suggested_fix"),
        ]
        for item in badcases
    ]
    badcases_md = "# v4 Stratigraphy Badcases\n\n"
    badcases_md += markdown_table(
        ["case_id", "question", "badcase_type", "expected", "actual", "suspected_reason", "suggested_fix"],
        badcase_rows,
    ) if badcase_rows else "本轮未发现 Badcase。\n"
    (output_dir / "v4_stratigraphy_badcases.md").write_text(badcases_md, encoding="utf-8")

    case_lookup = {safe_text(row.get("id")): row for row in dataset_rows}
    strat_case_rows = []
    for item in strat_results:
        dataset_row = case_lookup.get(safe_text(item.get("case_id")), {})
        meta = metadata(dataset_row)
        strat_case_rows.append(
            [
                item.get("case_id"),
                meta.get("replaced_from_question"),
                item.get("question"),
                item.get("stratigraphy_dimension"),
                item.get("stratigraphy_term"),
                STRATIGRAPHY_ROUTE,
                item.get("ground_truth_layer_count"),
            ]
        )

    metric_rows = [
        ["Stratigraphy 路由命中率", f"{metrics['stratigraphy']['route_hit_count']}/10", "是否成功进入新路由"],
        ["检索召回率", f"{metrics['stratigraphy']['avg_retrieval_recall_percent']}%", "layer_id 级别"],
        ["检索准确率", f"{metrics['stratigraphy']['avg_retrieval_precision_percent']}%", "layer_id 级别"],
        ["答案准确率", f"{metrics['stratigraphy']['answer_accuracy_percent']}%", "Markdown 表格字段完整性"],
        ["零幻觉率", f"{metrics['stratigraphy']['no_hallucination_percent']}%", "是否存在额外编造"],
        ["原有 40 条回归数", metrics["regression"]["regression_count"], "对比 v3"],
    ]

    summary_md = "# v4 Stratigraphy Evaluation Summary\n\n"
    summary_md += "## 新增 10 条 Stratigraphy 测试问题\n\n"
    summary_md += markdown_table(
        ["case_id", "原问题", "新问题", "地层维度", "地层名称", "expected_route", "ground truth 数量"],
        strat_case_rows,
    )
    summary_md += "\n\n## v4 测评指标\n\n"
    summary_md += markdown_table(["指标", "数值", "说明"], metric_rows)
    summary_md += "\n\n## Badcase 明细\n\n"
    summary_md += markdown_table(
        ["case_id", "question", "badcase_type", "expected", "actual", "suspected_reason", "suggested_fix"],
        badcase_rows,
    ) if badcase_rows else "本轮未发现 Badcase。\n"
    summary_md += "\n\n## 修复建议\n\n"
    summary_md += "1. P0：优先修复 route_miss、retrieval_empty、retrieval_noise 和 route_false_positive。\n"
    summary_md += "2. P1：修复 answer_format_error，保证直接答案表格完整展示纪、世、组、段、厚度和岩性字段。\n"
    summary_md += "3. P2：补充 Stratigraphy 专项单测，覆盖词表加载、Cypher 查询和 Markdown 输出格式。\n"
    summary_md += "\n## 可复现命令\n\n"
    summary_md += "```bash\n"
    summary_md += "python evals/scripts/build_stratigraphy_eval_v4.py\n"
    summary_md += "python evals/eval_retrieval_v2.py --dataset evals/datasets/rag_mvp_eval_50_stratigraphy_v4.jsonl --output-dir evals/results/v4_stratigraphy --rankgpt off\n"
    summary_md += "python evals/scripts/analyze_stratigraphy_v4.py --dataset evals/datasets/rag_mvp_eval_50_stratigraphy_v4.jsonl --eval-details evals/results/v4_stratigraphy/retrieval_v2_rankgpt_off_details.jsonl --output-dir evals/results/v4_stratigraphy\n"
    summary_md += "```\n"
    (output_dir / "v4_stratigraphy_summary.md").write_text(summary_md, encoding="utf-8")


def analyze(
    dataset_path: Path,
    eval_details_path: Path,
    v3_details_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    dataset_rows = load_jsonl(dataset_path)
    eval_rows = load_jsonl(eval_details_path)
    v3_rows = load_jsonl(v3_details_path)

    if len(dataset_rows) != 50:
        raise ValueError(f"v4 数据集应为 50 条，实际为 {len(dataset_rows)}。")

    strat_dataset_rows = [row for row in dataset_rows if is_strat_case(row)]
    if len(strat_dataset_rows) != 10:
        raise ValueError(f"Stratigraphy 专项样本应为 10 条，实际为 {len(strat_dataset_rows)}。")

    strat_results = [evaluate_strat_case(row) for row in strat_dataset_rows]

    v3_by_id = {safe_text(row.get("id")): row for row in v3_rows}
    strat_ids = {safe_text(row.get("id")) for row in strat_dataset_rows}
    regression_rows: List[Dict[str, Any]] = []

    for row in eval_rows:
        row_id = safe_text(row.get("id"))
        if row_id in strat_ids:
            continue
        reasons = regression_reason(row, v3_by_id.get(row_id))
        if reasons:
            regression_rows.append(
                {
                    **row,
                    "v3_actual_route": (v3_by_id.get(row_id) or {}).get("actual_route", ""),
                    "regression_reasons": reasons,
                }
            )

    route_hit_count = sum(1 for row in strat_results if row.get("route_hit"))
    answer_ok_count = sum(1 for row in strat_results if row.get("answer_field_complete"))
    no_hallucination_count = sum(1 for row in strat_results if row.get("no_hallucination"))

    metrics = {
        "dataset": {
            "path": str(dataset_path),
            "total_count": len(dataset_rows),
            "stratigraphy_case_count": len(strat_dataset_rows),
            "regression_case_count": len(dataset_rows) - len(strat_dataset_rows),
        },
        "stratigraphy": {
            "route_hit_count": route_hit_count,
            "route_hit_rate_percent": percent(route_hit_count, len(strat_results)),
            "avg_retrieval_recall": round(
                sum(row["retrieval_recall"] for row in strat_results) / len(strat_results),
                4,
            ),
            "avg_retrieval_recall_percent": percent(
                sum(row["retrieval_recall"] for row in strat_results),
                len(strat_results),
            ),
            "avg_retrieval_precision": round(
                sum(row["retrieval_precision"] for row in strat_results) / len(strat_results),
                4,
            ),
            "avg_retrieval_precision_percent": percent(
                sum(row["retrieval_precision"] for row in strat_results),
                len(strat_results),
            ),
            "answer_accuracy_count": answer_ok_count,
            "answer_accuracy_percent": percent(answer_ok_count, len(strat_results)),
            "no_hallucination_count": no_hallucination_count,
            "no_hallucination_percent": percent(no_hallucination_count, len(strat_results)),
        },
        "regression": {
            "regression_count": len(regression_rows),
            "regression_case_ids": [row.get("id") for row in regression_rows],
        },
    }

    badcases = build_badcase_rows(strat_results, regression_rows)
    save_jsonl(strat_results, output_dir / "v4_stratigraphy_case_results.jsonl")
    save_jsonl(badcases, output_dir / "v4_stratigraphy_badcases.jsonl")
    write_reports(
        output_dir=output_dir,
        dataset_rows=dataset_rows,
        strat_results=strat_results,
        eval_rows=eval_rows,
        metrics=metrics,
        badcases=badcases,
    )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze v4 Stratigraphy route metrics and badcases.",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="v4 JSONL 测评集。")
    parser.add_argument(
        "--eval-details",
        default=str(DEFAULT_EVAL_DETAILS),
        help="eval_retrieval_v2.py 生成的 details JSONL。",
    )
    parser.add_argument(
        "--v3-details",
        default=str(DEFAULT_V3_DETAILS),
        help="v3 基线 details JSONL。",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = analyze(
        dataset_path=Path(args.dataset).resolve(),
        eval_details_path=Path(args.eval_details).resolve(),
        v3_details_path=Path(args.v3_details).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )
    print(json.dumps(make_json_safe(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
