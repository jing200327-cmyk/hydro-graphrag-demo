from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ============================================================
# 1. 项目路径与 .env
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_DIR / ".env")
except Exception:
    pass


# ============================================================
# 2. 导入项目模块
# ============================================================

from src.pipeline.qa_pipeline import run_end_to_end_graphrag_qa
from src.retrieval.exact_retriever import extract_exact_ids
from src.retrieval.hydro_feature_retriever import (
    extract_hydro_feature_query,
    score_mechanism_coverage,
)
from src.rerank.rankgpt_reranker import rankgpt_rerank_chunks
from src.utils.text import safe_text, make_json_safe


# ============================================================
# 3. 默认路径
# ============================================================

DEFAULT_DATASET_PATH = PROJECT_DIR / "evals" / "datasets" / "rag_mvp_eval_50.jsonl"
DEFAULT_REPORT_DIR = PROJECT_DIR / "evals" / "reports_v2"


# ============================================================
# 4. 数据读取
# ============================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"数据集不存在：{path}")

    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()

            if not text:
                continue

            try:
                obj = json.loads(text)
            except Exception as exc:
                raise ValueError(f"JSONL 第 {line_no} 行解析失败：{exc}") from exc

            if not isinstance(obj, dict):
                raise ValueError(f"JSONL 第 {line_no} 行不是对象。")

            rows.append(obj)

    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_safe(row), ensure_ascii=False) + "\n")


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(obj), f, ensure_ascii=False, indent=2)


# ============================================================
# 5. 通用工具
# ============================================================

def as_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        # 兼容逗号分隔
        if "," in text:
            return [x.strip() for x in text.split(",") if x.strip()]

        # 兼容中文顿号
        if "、" in text:
            return [x.strip() for x in text.split("、") if x.strip()]

        return [text]

    return [value]


def normalize_id(value: Any) -> str:
    return safe_text(value).replace(" ", "").upper()


def normalize_text(value: Any) -> str:
    return safe_text(value).replace(" ", "").upper()


def contains_id(text: str, target_id: str) -> bool:
    text_norm = normalize_text(text)
    target_norm = normalize_id(target_id)

    if not text_norm or not target_norm:
        return False

    return target_norm in text_norm


def get_question_type(item: Dict[str, Any]) -> str:
    for key in ["question_type", "type", "category", "类别", "问题类型"]:
        value = safe_text(item.get(key, ""))
        if value:
            return value
    return ""


def get_question_id(item: Dict[str, Any], idx: int) -> str:
    for key in ["id", "qid", "question_id"]:
        value = safe_text(item.get(key, ""))
        if value:
            return value
    return f"case_{idx:03d}"


def get_question(item: Dict[str, Any]) -> str:
    for key in ["question", "user_question", "query", "问题"]:
        value = safe_text(item.get(key, ""))
        if value:
            return value
    return ""


def is_fact_type(question_type: str) -> bool:
    qt = safe_text(question_type).lower()

    return qt in {
        "fact",
        "fact_query",
        "事实查询",
        "factual",
    }


def is_fallback_type(question_type: str, item: Dict[str, Any]) -> bool:
    qt = safe_text(question_type).lower()

    if qt in {
        "fallback",
        "fallback_unanswerable",
        "unanswerable",
        "兜底",
        "无法回答",
        "兜底/无法回答",
    }:
        return True

    return bool(item.get("should_abstain", False))


def is_causal_type(question_type: str) -> bool:
    return safe_text(question_type).lower() == "causal_explanation"


def infer_expected_route(question_type: str, item: Dict[str, Any]) -> str:
    """
    评测期望路由。

    fact_query 应该走：
    - fact_exact_retrieval

    permeability_level 应该走：
    - lithology_type_exact_retrieval

    其他类型默认走：
    - vector_graph_retrieval

    注意：
    fallback_unanswerable 可能因为问题中带精确 ID 被识别成 fact_query。
    这类问题更关注是否能拒答，不强制 route。
    这里返回空字符串表示不参与 route_accuracy。
    """

    if is_fallback_type(question_type, item):
        return ""

    if is_fact_type(question_type):
        return "fact_exact_retrieval"

    if safe_text(question_type) == "permeability_level":
        return "lithology_type_exact_retrieval"

    if is_causal_type(question_type):
        return "hydro_feature_exact_retrieval"

    return "vector_graph_retrieval"


def get_result_chunks(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = result.get("vector_search_results", [])

    if isinstance(value, list):
        return value

    return []


def get_c_rerank_results(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = result.get("c_rerank_results", [])

    if isinstance(value, list):
        return value

    return []


def get_effective_rerank_results(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = result.get("effective_rerank_results", [])

    if isinstance(value, list) and value:
        return value

    return get_c_rerank_results(result)


def build_chunk_map(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    for chunk in chunks:
        cid = safe_text(chunk.get("chunk_id", ""))

        if cid:
            result[cid] = chunk

    return result


def merge_rerank_with_chunks(
    rerank_results: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    chunk_map = build_chunk_map(chunks)
    merged: List[Dict[str, Any]] = []

    for item in rerank_results:
        cid = safe_text(item.get("chunk_id", ""))
        base = dict(chunk_map.get(cid, {}))
        base.update(item)

        if "chunk_text" not in base and cid in chunk_map:
            base["chunk_text"] = chunk_map[cid].get("chunk_text", "")

        merged.append(base)

    return merged


def candidate_text(item: Dict[str, Any]) -> str:
    fields = [
        "chunk_id",
        "source_id",
        "source_label",
        "chunk_type",
        "chunk_text",
        "graph_context_text",
        "exact_match_value",
        "exact_match_type",
        "rerank_reason",
    ]

    values: List[str] = []

    for field in fields:
        value = item.get(field)

        if isinstance(value, (dict, list)):
            values.append(json.dumps(make_json_safe(value), ensure_ascii=False))
        else:
            values.append(safe_text(value))

    source_props = item.get("source_props")

    if source_props:
        values.append(json.dumps(make_json_safe(source_props), ensure_ascii=False))

    return " ".join([v for v in values if v])


# ============================================================
# 6. 期望字段抽取
# ============================================================

def infer_expected_layer_ids(item: Dict[str, Any], question: str) -> List[str]:
    values: List[str] = []

    for key in ["expected_layer_id", "expected_layer_ids", "target_layer_id", "target_layer_ids"]:
        values.extend(as_list(item.get(key)))

    expected_chunk_ids = as_list(item.get("expected_chunk_ids"))
    for cid in expected_chunk_ids:
        text = safe_text(cid)
        if "_" in text:
            values.append(text)

    try:
        exact_ids = extract_exact_ids(question)
        values.extend(exact_ids.get("layer_ids", []))
    except Exception:
        pass

    return unique_normalized_ids(values)


def infer_expected_borehole_ids(item: Dict[str, Any], question: str) -> List[str]:
    values: List[str] = []

    for key in ["expected_borehole_id", "expected_borehole_ids", "target_borehole_id", "target_borehole_ids"]:
        values.extend(as_list(item.get(key)))

    expected_chunk_ids = as_list(item.get("expected_chunk_ids"))
    for cid in expected_chunk_ids:
        text = safe_text(cid)
        if "_" in text:
            values.append(text.split("_", 1)[0])

    try:
        exact_ids = extract_exact_ids(question)
        values.extend(exact_ids.get("borehole_ids", []))
    except Exception:
        pass

    return unique_normalized_ids(values)


def infer_expected_chunk_ids(item: Dict[str, Any]) -> List[str]:
    values: List[str] = []

    for key in ["expected_chunk_id", "expected_chunk_ids", "gold_chunk_id", "gold_chunk_ids", "target_chunk_ids"]:
        values.extend(as_list(item.get(key)))

    return [safe_text(v) for v in values if safe_text(v)]


def infer_expected_keywords(item: Dict[str, Any]) -> List[str]:
    values: List[str] = []

    for key in ["expected_keywords", "answer_keywords", "keywords", "must_include"]:
        values.extend(as_list(item.get(key)))

    return [safe_text(v) for v in values if safe_text(v)]


def infer_expected_causal_targets(
    item: Dict[str, Any],
    question: str,
) -> Dict[str, str]:
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    extracted = extract_hydro_feature_query(question)

    feature_value = (
        safe_text(metadata.get("feature_value", ""))
        or safe_text(item.get("feature_value", ""))
        or safe_text(extracted.get("feature_value", ""))
    )
    permeability_effect = (
        safe_text(metadata.get("permeability_effect", ""))
        or safe_text(metadata.get("impact", ""))
        or safe_text(item.get("permeability_effect", ""))
        or safe_text(item.get("impact", ""))
        or safe_text(extracted.get("permeability_effect", ""))
    )

    return {
        "feature_value": feature_value,
        "permeability_effect": permeability_effect,
    }


def unique_normalized_ids(values: Iterable[Any]) -> List[str]:
    seen = set()
    results: List[str] = []

    for value in values:
        text = normalize_id(value)

        if not text:
            continue

        if text in seen:
            continue

        seen.add(text)
        results.append(text)

    return results


# ============================================================
# 7. 命中判断
# ============================================================

def hit_expected_chunk_ids(
    ranked_items: List[Dict[str, Any]],
    expected_chunk_ids: List[str],
    k: int,
) -> bool:
    if not expected_chunk_ids:
        return False

    expected_set = set([safe_text(x) for x in expected_chunk_ids if safe_text(x)])

    for item in ranked_items[:k]:
        cid = safe_text(item.get("chunk_id", ""))

        if cid in expected_set:
            return True

    return False


def hit_expected_ids_in_items(
    ranked_items: List[Dict[str, Any]],
    expected_ids: List[str],
    k: int,
) -> bool:
    if not expected_ids:
        return False

    for item in ranked_items[:k]:
        text = candidate_text(item)

        for target_id in expected_ids:
            if contains_id(text, target_id):
                return True

    return False


def top1_contains_id(
    ranked_items: List[Dict[str, Any]],
    expected_ids: List[str],
) -> bool:
    if not ranked_items or not expected_ids:
        return False

    return hit_expected_ids_in_items(ranked_items, expected_ids, k=1)


def hit_keywords_in_items(
    ranked_items: List[Dict[str, Any]],
    expected_keywords: List[str],
    k: int,
) -> bool:
    if not expected_keywords:
        return False

    text = " ".join(candidate_text(item) for item in ranked_items[:k])

    return all(keyword in text for keyword in expected_keywords if keyword)


def _structured_field_equals(item: Dict[str, Any], field_names: List[str], target: str) -> bool:
    if not target:
        return False

    for field in field_names:
        if safe_text(item.get(field, "")) == target:
            return True

    source_props = item.get("source_props")
    if isinstance(source_props, dict):
        for field in field_names:
            if safe_text(source_props.get(field, "")) == target:
                return True

    graph_context = item.get("graph_context")
    if isinstance(graph_context, dict):
        for key in ["source_props", "props", "properties"]:
            props = graph_context.get(key)
            if isinstance(props, dict):
                for field in field_names:
                    if safe_text(props.get(field, "")) == target:
                        return True

    return False


def _field_phrase_hit(text: str, field_names: List[str], target: str) -> bool:
    if not target:
        return False

    phrases = []
    if "feature_value" in field_names:
        phrases.extend(
            [
                f"特征取值为“{target}”",
                f"特征取值为\"{target}\"",
                f'"feature_value": "{target}"',
                f'"feature_value":"{target}"',
            ]
        )
    if "permeability_effect" in field_names or "impact" in field_names:
        phrases.extend(
            [
                f"对渗透率的影响为“{target}”",
                f"对渗透率的影响为\"{target}\"",
                f'"permeability_effect": "{target}"',
                f'"permeability_effect":"{target}"',
                f'"impact": "{target}"',
                f'"impact":"{target}"',
            ]
        )

    return any(phrase in text for phrase in phrases)


def exact_field_hit_in_items(
    ranked_items: List[Dict[str, Any]],
    field_names: List[str],
    target: str,
    k: int,
) -> bool:
    if not target:
        return False

    for item in ranked_items[:k]:
        if _structured_field_equals(item, field_names, target):
            return True

    text = " ".join(candidate_text(item) for item in ranked_items[:k])
    return _field_phrase_hit(text, field_names, target)


def eval_causal_constraints(
    ranked_items: List[Dict[str, Any]],
    item: Dict[str, Any],
    question: str,
    k: int,
) -> Dict[str, Any]:
    targets = infer_expected_causal_targets(item, question)
    feature_value = targets["feature_value"]
    permeability_effect = targets["permeability_effect"]

    feature_value_hit = exact_field_hit_in_items(
        ranked_items,
        ["feature_value"],
        feature_value,
        k=k,
    )
    permeability_effect_hit = exact_field_hit_in_items(
        ranked_items,
        ["permeability_effect", "impact"],
        permeability_effect,
        k=k,
    )

    text = " ".join(candidate_text(x) for x in ranked_items[:k])
    mechanism_eval = score_mechanism_coverage(text)

    return {
        "causal_feature_value": feature_value,
        "causal_permeability_effect": permeability_effect,
        "causal_feature_value_hit": feature_value_hit,
        "causal_permeability_effect_hit": permeability_effect_hit,
        "causal_mechanism_hit": bool(mechanism_eval.get("mechanism_hit", False)),
        "causal_mechanism_score": mechanism_eval.get("mechanism_score", 0.0),
        "causal_mechanism_terms": mechanism_eval.get("mechanism_terms", []),
    }


def first_rank_of_expected_id(
    ranked_items: List[Dict[str, Any]],
    expected_ids: List[str],
) -> Optional[int]:
    if not expected_ids:
        return None

    for idx, item in enumerate(ranked_items, start=1):
        text = candidate_text(item)

        for target_id in expected_ids:
            if contains_id(text, target_id):
                return idx

    return None


def first_rank_of_expected_chunk(
    ranked_items: List[Dict[str, Any]],
    expected_chunk_ids: List[str],
) -> Optional[int]:
    if not expected_chunk_ids:
        return None

    expected_set = set([safe_text(x) for x in expected_chunk_ids if safe_text(x)])

    for idx, item in enumerate(ranked_items, start=1):
        cid = safe_text(item.get("chunk_id", ""))

        if cid in expected_set:
            return idx

    return None


def reciprocal_rank(rank: Optional[int]) -> float:
    if not rank or rank <= 0:
        return 0.0
    return 1.0 / rank


# ============================================================
# 8. 单条样本评测
# ============================================================

def run_pipeline_for_eval(
    question: str,
    raw_top_k: int,
    final_top_k: int,
    enable_pipeline_rankgpt: bool = False,
) -> Dict[str, Any]:
    """
    调用 qa_pipeline。

    注意：
    - 这里默认 enable_llm=False，避免生成最终答案。
    - 如果 enable_pipeline_rankgpt=True，因为当前 qa_pipeline 里 RankGPT 依赖 enable_llm，
      会触发最终答案生成，不推荐 retrieval 评测使用。
    - 本脚本默认使用独立 RankGPT 调用，因此这里保持 False。
    """

    return run_end_to_end_graphrag_qa(
        user_question=question,
        raw_top_k=raw_top_k,
        final_top_k=final_top_k,
        save_outputs=False,
        enable_llm=False,
        enable_rankgpt=enable_pipeline_rankgpt,
    )


def maybe_apply_rankgpt_for_eval(
    question: str,
    pipeline_result: Dict[str, Any],
    rankgpt_mode: str,
    final_top_k: int,
    rankgpt_candidate_top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """
    Retrieval 评测中的 RankGPT 调用。

    为了避免触发最终答案生成，这里不通过 qa_pipeline 调 RankGPT，
    而是直接调用 rankgpt_rerank_chunks。

    返回：
    - effective_results
    - rankgpt_results
    - rankgpt_used_any
    """

    c_results = get_c_rerank_results(pipeline_result)
    retrieved_chunks = get_result_chunks(pipeline_result)

    if rankgpt_mode != "on":
        effective = get_effective_rerank_results(pipeline_result)
        return effective, [], False

    if not c_results:
        return [], [], False

    rankgpt_results = rankgpt_rerank_chunks(
        user_question=question,
        reranked_results=c_results,
        retrieved_chunks=retrieved_chunks,
        question_analysis=pipeline_result.get("question_analysis", {}),
        candidate_top_k=rankgpt_candidate_top_k,
        final_top_k=final_top_k,
        enabled=True,
        fail_open=True,
    )

    rankgpt_used_any = any(bool(item.get("rankgpt_used", False)) for item in rankgpt_results)

    effective_results = rankgpt_results if rankgpt_results else c_results

    return effective_results, rankgpt_results, rankgpt_used_any


def eval_one_case(
    item: Dict[str, Any],
    idx: int,
    raw_top_k: int,
    final_top_k: int,
    rankgpt_mode: str,
    rankgpt_candidate_top_k: int,
    stop_on_error: bool = False,
) -> Dict[str, Any]:
    qid = get_question_id(item, idx)
    question = get_question(item)
    question_type = get_question_type(item)

    if not question:
        raise ValueError(f"{qid} 缺少 question 字段。")

    expected_route = infer_expected_route(question_type, item)
    expected_layer_ids = infer_expected_layer_ids(item, question)
    expected_borehole_ids = infer_expected_borehole_ids(item, question)
    expected_chunk_ids = infer_expected_chunk_ids(item)
    expected_keywords = infer_expected_keywords(item)

    started = time.perf_counter()

    try:
        pipeline_result = run_pipeline_for_eval(
            question=question,
            raw_top_k=raw_top_k,
            final_top_k=final_top_k,
            enable_pipeline_rankgpt=False,
        )

        effective_results, rankgpt_results, rankgpt_used_any = maybe_apply_rankgpt_for_eval(
            question=question,
            pipeline_result=pipeline_result,
            rankgpt_mode=rankgpt_mode,
            final_top_k=final_top_k,
            rankgpt_candidate_top_k=rankgpt_candidate_top_k,
        )

        elapsed_sec = round(time.perf_counter() - started, 3)

        retrieved_chunks = get_result_chunks(pipeline_result)
        c_results = get_c_rerank_results(pipeline_result)

        retrieved_merged = retrieved_chunks
        c_merged = merge_rerank_with_chunks(c_results, retrieved_chunks)
        effective_merged = merge_rerank_with_chunks(effective_results, retrieved_chunks)

        actual_route = safe_text(pipeline_result.get("retrieval_route", ""))
        actual_rerank_route = safe_text(pipeline_result.get("rerank_route", ""))
        actual_intent = safe_text(pipeline_result.get("question_intent", ""))

        route_eval_applicable = bool(expected_route)
        route_correct = None

        if route_eval_applicable:
            route_correct = actual_route == expected_route

        retrieval_hit_layer_top1 = top1_contains_id(retrieved_merged, expected_layer_ids)
        retrieval_hit_layer_top3 = hit_expected_ids_in_items(retrieved_merged, expected_layer_ids, k=3)
        retrieval_hit_layer_top10 = hit_expected_ids_in_items(retrieved_merged, expected_layer_ids, k=10)

        effective_hit_layer_top1 = top1_contains_id(effective_merged, expected_layer_ids)
        effective_hit_layer_top3 = hit_expected_ids_in_items(effective_merged, expected_layer_ids, k=3)
        effective_hit_layer_top10 = hit_expected_ids_in_items(effective_merged, expected_layer_ids, k=10)

        retrieval_hit_borehole_top10 = hit_expected_ids_in_items(retrieved_merged, expected_borehole_ids, k=10)
        effective_hit_borehole_top10 = hit_expected_ids_in_items(effective_merged, expected_borehole_ids, k=10)

        retrieval_hit_chunk_top10 = hit_expected_chunk_ids(retrieved_merged, expected_chunk_ids, k=10)
        effective_hit_chunk_top10 = hit_expected_chunk_ids(effective_merged, expected_chunk_ids, k=10)

        keyword_hit_top10 = hit_keywords_in_items(effective_merged, expected_keywords, k=10)
        causal_eval = eval_causal_constraints(effective_merged, item, question, k=10)
        if not is_causal_type(question_type):
            causal_eval = {
                "causal_feature_value": "",
                "causal_permeability_effect": "",
                "causal_feature_value_hit": None,
                "causal_permeability_effect_hit": None,
                "causal_mechanism_hit": None,
                "causal_mechanism_score": 0.0,
                "causal_mechanism_terms": [],
            }

        retrieval_rank_layer = first_rank_of_expected_id(retrieved_merged, expected_layer_ids)
        effective_rank_layer = first_rank_of_expected_id(effective_merged, expected_layer_ids)
        effective_rank_chunk = first_rank_of_expected_chunk(effective_merged, expected_chunk_ids)

        rankgpt_top1_before = None
        rankgpt_top1_after = None
        rankgpt_top1_improved = None

        if rankgpt_mode == "on":
            before_merged = c_merged
            after_merged = effective_merged

            before_rank = first_rank_of_expected_id(before_merged, expected_layer_ids)
            after_rank = first_rank_of_expected_id(after_merged, expected_layer_ids)

            if expected_layer_ids:
                rankgpt_top1_before = bool(before_rank == 1)
                rankgpt_top1_after = bool(after_rank == 1)

                if before_rank is not None and after_rank is not None:
                    rankgpt_top1_improved = after_rank < before_rank
                elif before_rank is None and after_rank is not None:
                    rankgpt_top1_improved = True
                else:
                    rankgpt_top1_improved = False

        top_retrieved_chunk_id = safe_text(retrieved_merged[0].get("chunk_id", "")) if retrieved_merged else ""
        top_c_rerank_chunk_id = safe_text(c_merged[0].get("chunk_id", "")) if c_merged else ""
        top_effective_chunk_id = safe_text(effective_merged[0].get("chunk_id", "")) if effective_merged else ""

        rankgpt_order_changed = False
        if rankgpt_mode == "on" and c_merged and effective_merged:
            before_ids = [safe_text(x.get("chunk_id", "")) for x in c_merged]
            after_ids = [safe_text(x.get("chunk_id", "")) for x in effective_merged]
            rankgpt_order_changed = before_ids != after_ids

        rankgpt_reason = ""
        rankgpt_invalid_ids: List[str] = []

        if rankgpt_mode == "on" and rankgpt_results:
            rankgpt_reason = safe_text(rankgpt_results[0].get("rankgpt_reason", ""))
            invalid_ids = rankgpt_results[0].get("rankgpt_invalid_ids", [])
            if isinstance(invalid_ids, list):
                rankgpt_invalid_ids = [safe_text(x) for x in invalid_ids if safe_text(x)]

        row = {
            "id": qid,
            "question": question,
            "question_type": question_type,
            "actual_intent": actual_intent,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "actual_rerank_route": actual_rerank_route,
            "route_eval_applicable": route_eval_applicable,
            "route_correct": route_correct,
            "rankgpt_mode": rankgpt_mode,
            "rankgpt_used_any": rankgpt_used_any,

            "expected_layer_ids": expected_layer_ids,
            "expected_borehole_ids": expected_borehole_ids,
            "expected_chunk_ids": expected_chunk_ids,
            "expected_keywords": expected_keywords,

            "retrieved_count": len(retrieved_chunks),
            "c_rerank_count": len(c_results),
            "effective_rerank_count": len(effective_results),

            "top_retrieved_chunk_id": top_retrieved_chunk_id,
            "top_effective_chunk_id": top_effective_chunk_id,

            "top_c_rerank_chunk_id": top_c_rerank_chunk_id,
            "rankgpt_order_changed": rankgpt_order_changed,
            "rankgpt_reason": rankgpt_reason,
            "rankgpt_invalid_ids": rankgpt_invalid_ids,

            "retrieval_hit_layer_top1": retrieval_hit_layer_top1,
            "retrieval_hit_layer_top3": retrieval_hit_layer_top3,
            "retrieval_hit_layer_top10": retrieval_hit_layer_top10,

            "effective_hit_layer_top1": effective_hit_layer_top1,
            "effective_hit_layer_top3": effective_hit_layer_top3,
            "effective_hit_layer_top10": effective_hit_layer_top10,

            "retrieval_hit_borehole_top10": retrieval_hit_borehole_top10,
            "effective_hit_borehole_top10": effective_hit_borehole_top10,

            "retrieval_hit_chunk_top10": retrieval_hit_chunk_top10,
            "effective_hit_chunk_top10": effective_hit_chunk_top10,

            "keyword_hit_top10": keyword_hit_top10,
            **causal_eval,

            "retrieval_rank_layer": retrieval_rank_layer,
            "effective_rank_layer": effective_rank_layer,
            "effective_rank_chunk": effective_rank_chunk,
            "retrieval_mrr_layer": reciprocal_rank(retrieval_rank_layer),
            "effective_mrr_layer": reciprocal_rank(effective_rank_layer),
            "effective_mrr_chunk": reciprocal_rank(effective_rank_chunk),

            "rankgpt_top1_before": rankgpt_top1_before,
            "rankgpt_top1_after": rankgpt_top1_after,
            "rankgpt_top1_improved": rankgpt_top1_improved,

            "confidence": pipeline_result.get("confidence", 0.0),
            "fallback_triggered": pipeline_result.get("fallback_triggered", False),
            "elapsed_sec": elapsed_sec,
            "error": "",
        }

        # 保存简略调试字段，方便 bad case 分析
        row["top5_effective"] = [
            {
                "rank": i + 1,
                "chunk_id": safe_text(x.get("chunk_id", "")),
                "source_id": safe_text(x.get("source_id", "")),
                "final_score": x.get("final_score", None),
                "rankgpt_used": x.get("rankgpt_used", None),
                "feature_value_hit": x.get("feature_value_hit", None),
                "permeability_effect_hit": x.get("permeability_effect_hit", None),
                "mechanism_hit": x.get("mechanism_hit", None),
                "mechanism_score": x.get("mechanism_score", None),
                "rerank_reason": safe_text(x.get("rerank_reason", ""))[:300],
            }
            for i, x in enumerate(effective_merged[:5])
        ]

        return row

    except Exception as exc:
        elapsed_sec = round(time.perf_counter() - started, 3)

        if stop_on_error:
            raise

        return {
            "id": qid,
            "question": question,
            "question_type": question_type,
            "actual_intent": "",
            "expected_route": expected_route,
            "actual_route": "",
            "actual_rerank_route": "",
            "route_eval_applicable": bool(expected_route),
            "route_correct": None,
            "rankgpt_mode": rankgpt_mode,
            "rankgpt_used_any": False,
            "expected_layer_ids": expected_layer_ids,
            "expected_borehole_ids": expected_borehole_ids,
            "expected_chunk_ids": expected_chunk_ids,
            "expected_keywords": expected_keywords,
            "retrieved_count": 0,
            "c_rerank_count": 0,
            "effective_rerank_count": 0,
            "top_retrieved_chunk_id": "",
            "top_effective_chunk_id": "",
            "retrieval_hit_layer_top1": False,
            "retrieval_hit_layer_top3": False,
            "retrieval_hit_layer_top10": False,
            "effective_hit_layer_top1": False,
            "effective_hit_layer_top3": False,
            "effective_hit_layer_top10": False,
            "retrieval_hit_borehole_top10": False,
            "effective_hit_borehole_top10": False,
            "retrieval_hit_chunk_top10": False,
            "effective_hit_chunk_top10": False,
            "keyword_hit_top10": False,
            "causal_feature_value": "",
            "causal_permeability_effect": "",
            "causal_feature_value_hit": None,
            "causal_permeability_effect_hit": None,
            "causal_mechanism_hit": None,
            "causal_mechanism_score": 0.0,
            "causal_mechanism_terms": [],
            "retrieval_rank_layer": None,
            "effective_rank_layer": None,
            "effective_rank_chunk": None,
            "retrieval_mrr_layer": 0.0,
            "effective_mrr_layer": 0.0,
            "effective_mrr_chunk": 0.0,
            "rankgpt_top1_before": None,
            "rankgpt_top1_after": None,
            "rankgpt_top1_improved": None,
            "confidence": 0.0,
            "fallback_triggered": False,
            "elapsed_sec": elapsed_sec,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "top5_effective": [],
        }


# ============================================================
# 9. 汇总统计
# ============================================================

def bool_rate(rows: List[Dict[str, Any]], key: str, only_applicable_key: Optional[str] = None) -> Optional[float]:
    values: List[bool] = []

    for row in rows:
        if only_applicable_key and not row.get(only_applicable_key):
            continue

        value = row.get(key)

        if value is None:
            continue

        values.append(bool(value))

    if not values:
        return None

    return round(sum(values) / len(values), 4)


def mean_value(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values: List[float] = []

    for row in rows:
        value = row.get(key)

        if value is None:
            continue

        try:
            values.append(float(value))
        except Exception:
            continue

    if not values:
        return None

    return round(sum(values) / len(values), 4)


def count_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counter = Counter()

    for row in rows:
        counter[safe_text(row.get(key, ""))] += 1

    return dict(counter)


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    error_count = sum(1 for row in rows if row.get("error"))

    fact_rows = [r for r in rows if is_fact_type(safe_text(r.get("question_type", "")))]
    non_fact_rows = [
        r
        for r in rows
        if not is_fact_type(safe_text(r.get("question_type", "")))
        and not is_fallback_type(safe_text(r.get("question_type", "")), r)
    ]
    fallback_rows = [r for r in rows if is_fallback_type(safe_text(r.get("question_type", "")), r)]
    causal_rows = [r for r in rows if is_causal_type(safe_text(r.get("question_type", "")))]

    summary = {
        "total": total,
        "error_count": error_count,
        "success_count": total - error_count,

        "question_type_counts": count_by(rows, "question_type"),
        "actual_intent_counts": count_by(rows, "actual_intent"),
        "actual_route_counts": count_by(rows, "actual_route"),
        "actual_rerank_route_counts": count_by(rows, "actual_rerank_route"),

        "overall": {
            "route_accuracy": bool_rate(rows, "route_correct", "route_eval_applicable"),
            "effective_non_empty_rate": bool_rate(
                [
                    {**r, "effective_non_empty": bool(r.get("effective_rerank_count", 0) > 0)}
                    for r in rows
                ],
                "effective_non_empty",
            ),
            "retrieved_non_empty_rate": bool_rate(
                [
                    {**r, "retrieved_non_empty": bool(r.get("retrieved_count", 0) > 0)}
                    for r in rows
                ],
                "retrieved_non_empty",
            ),
            "avg_elapsed_sec": mean_value(rows, "elapsed_sec"),
            "avg_confidence": mean_value(rows, "confidence"),
        },

        "fact_query": {
            "count": len(fact_rows),
            "route_accuracy": bool_rate(fact_rows, "route_correct", "route_eval_applicable"),
            "retrieval_hit_layer_top1": bool_rate(fact_rows, "retrieval_hit_layer_top1"),
            "retrieval_hit_layer_top3": bool_rate(fact_rows, "retrieval_hit_layer_top3"),
            "retrieval_hit_layer_top10": bool_rate(fact_rows, "retrieval_hit_layer_top10"),
            "effective_hit_layer_top1": bool_rate(fact_rows, "effective_hit_layer_top1"),
            "effective_hit_layer_top3": bool_rate(fact_rows, "effective_hit_layer_top3"),
            "effective_hit_layer_top10": bool_rate(fact_rows, "effective_hit_layer_top10"),
            "effective_hit_borehole_top10": bool_rate(fact_rows, "effective_hit_borehole_top10"),
            "effective_mrr_layer": mean_value(fact_rows, "effective_mrr_layer"),
            "retrieval_mrr_layer": mean_value(fact_rows, "retrieval_mrr_layer"),
            "rankgpt_top1_improved_rate": bool_rate(fact_rows, "rankgpt_top1_improved"),
        },

        "non_fact_query": {
            "count": len(non_fact_rows),
            "route_accuracy": bool_rate(non_fact_rows, "route_correct", "route_eval_applicable"),
            "b_score_non_empty_rate": bool_rate(
                [
                    {**r, "b_non_empty": bool(r.get("c_rerank_count", 0) > 0)}
                    for r in non_fact_rows
                ],
                "b_non_empty",
            ),
            "effective_non_empty_rate": bool_rate(
                [
                    {**r, "effective_non_empty": bool(r.get("effective_rerank_count", 0) > 0)}
                    for r in non_fact_rows
                ],
                "effective_non_empty",
            ),
            "keyword_hit_top10": bool_rate(non_fact_rows, "keyword_hit_top10"),
        },

        "causal_explanation": {
            "count": len(causal_rows),
            "route_accuracy": bool_rate(causal_rows, "route_correct", "route_eval_applicable"),
            "feature_value_hit_rate": bool_rate(causal_rows, "causal_feature_value_hit"),
            "permeability_effect_hit_rate": bool_rate(causal_rows, "causal_permeability_effect_hit"),
            "mechanism_hit_rate": bool_rate(causal_rows, "causal_mechanism_hit"),
            "avg_mechanism_score": mean_value(causal_rows, "causal_mechanism_score"),
        },

        "fallback": {
            "count": len(fallback_rows),
            "fallback_triggered_rate": bool_rate(fallback_rows, "fallback_triggered"),
        },
    }

    return summary


def build_bad_cases(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    构造 bad cases。

    增强版：
    1. 运行错误
    2. 路由错误
    3. fact_query 目标 layer_id 没有进入 Top1 / Top10
    4. effective_rerank 为空
    5. 非事实查询 expected_keywords 未命中
    6. fallback_unanswerable 没有触发 fallback
    7. RankGPT 开启但未实际使用
    8. RankGPT 返回非法 chunk_id
    9. RankGPT 开启后仍然没有命中关键词
    10. RankGPT 对排序没有产生变化，作为观察型 bad case
    """

    bad_cases: List[Dict[str, Any]] = []

    for row in rows:
        reasons: List[str] = []

        question_type = safe_text(row.get("question_type", ""))

        is_fact = is_fact_type(question_type)
        is_fallback = is_fallback_type(question_type, row)
        is_causal = is_causal_type(question_type)
        is_non_fact = not is_fact and not is_fallback

        expected_layer_ids = row.get("expected_layer_ids") or []
        expected_keywords = row.get("expected_keywords") or []

        rankgpt_mode = safe_text(row.get("rankgpt_mode", ""))
        rankgpt_is_on = rankgpt_mode == "on"

        # 1. 运行错误
        if row.get("error"):
            reasons.append("runtime_error")

        # 2. 路由错误
        if row.get("route_eval_applicable") and row.get("route_correct") is False:
            reasons.append("route_wrong")

        # 3. fact_query 专项
        if is_fact:
            if expected_layer_ids and not row.get("effective_hit_layer_top1"):
                reasons.append("fact_top1_layer_miss")

            if expected_layer_ids and not row.get("effective_hit_layer_top10"):
                reasons.append("fact_top10_layer_miss")

        # 4. effective rerank 为空
        # fallback_unanswerable 正确拒答时可以没有有效证据，不能因此算 badcase。
        if not is_fallback and not row.get("effective_rerank_count", 0):
            reasons.append("empty_effective_rerank")

        # 5. 非事实查询关键词未覆盖
        # causal_explanation 已拆成 feature/effect/mechanism 三层评测，不再只看 Top10 关键词。
        if is_causal:
            if row.get("causal_feature_value_hit") is False:
                reasons.append("causal_feature_value_miss")

            if row.get("causal_permeability_effect_hit") is False:
                reasons.append("causal_permeability_effect_miss")

            if row.get("causal_mechanism_hit") is False:
                reasons.append("causal_mechanism_weak")

        if is_non_fact and not is_causal and expected_keywords:
            if row.get("keyword_hit_top10") is False:
                reasons.append("non_fact_keyword_miss_top10")

        # 6. 兜底问题没有触发 fallback
        if is_fallback:
            if row.get("fallback_triggered") is not True:
                reasons.append("fallback_not_triggered")

        # 7. RankGPT 开启但未实际使用
        if rankgpt_is_on:
            if row.get("rankgpt_used_any") is not True:
                reasons.append("rankgpt_not_used")

        # 8. RankGPT 返回了非法 chunk_id
        invalid_ids = row.get("rankgpt_invalid_ids") or []
        if isinstance(invalid_ids, list) and invalid_ids:
            reasons.append("rankgpt_invalid_chunk_ids")

        # 9. RankGPT 开启后仍然没有命中关键词
        if rankgpt_is_on and is_non_fact and not is_causal and expected_keywords:
            if row.get("keyword_hit_top10") is False:
                reasons.append("rankgpt_keyword_still_miss_top10")

        # 10. RankGPT 对排序没有变化
        # 这是观察型 bad case，不一定是错误。
        # 对非 causal 的关键词型问题，如果关键词仍未命中且排序没变化，说明 RankGPT 没起到作用。
        if rankgpt_is_on and is_non_fact:
            if row.get("rankgpt_order_changed") is False:
                reasons.append("rankgpt_no_order_change")

        # 11. RankGPT 可能造成 fact_query 退化
        if row.get("rankgpt_top1_before") is True and row.get("rankgpt_top1_after") is False:
            reasons.append("rankgpt_top1_degraded")

        if reasons:
            bad_case = dict(row)
            bad_case["bad_case_reasons"] = reasons
            bad_cases.append(bad_case)

    return bad_cases


# ============================================================
# 10. 主评测流程
# ============================================================

def filter_dataset(
    rows: List[Dict[str, Any]],
    case_type: str,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    filtered = rows

    if case_type != "all":
        filtered = [
            row
            for row in filtered
            if safe_text(get_question_type(row)).lower() == case_type.lower()
        ]

    if limit is not None and limit > 0:
        filtered = filtered[:limit]

    return filtered


def run_eval(
    dataset_path: Path,
    output_dir: Path,
    case_type: str,
    limit: Optional[int],
    raw_top_k: int,
    final_top_k: int,
    rankgpt_mode: str,
    rankgpt_candidate_top_k: int,
    stop_on_error: bool,
) -> Dict[str, Any]:
    rows = load_jsonl(dataset_path)
    rows = filter_dataset(rows, case_type=case_type, limit=limit)

    if not rows:
        raise ValueError("没有可评测的数据。")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("Retrieval Eval V2")
    print("=" * 120)
    print(f"dataset_path: {dataset_path}")
    print(f"output_dir: {output_dir}")
    print(f"case_count: {len(rows)}")
    print(f"case_type: {case_type}")
    print(f"raw_top_k: {raw_top_k}")
    print(f"final_top_k: {final_top_k}")
    print(f"rankgpt_mode: {rankgpt_mode}")
    print("=" * 120)

    eval_rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(rows, start=1):
        qid = get_question_id(item, idx)
        question = get_question(item)
        question_type = get_question_type(item)

        print(f"\n[{idx}/{len(rows)}] {qid} | {question_type}")
        print(f"Q: {question}")

        row = eval_one_case(
            item=item,
            idx=idx,
            raw_top_k=raw_top_k,
            final_top_k=final_top_k,
            rankgpt_mode=rankgpt_mode,
            rankgpt_candidate_top_k=rankgpt_candidate_top_k,
            stop_on_error=stop_on_error,
        )

        eval_rows.append(row)

        print(
            "route=",
            row.get("actual_route"),
            "| intent=",
            row.get("actual_intent"),
            "| retrieved=",
            row.get("retrieved_count"),
            "| effective=",
            row.get("effective_rerank_count"),
            "| top_effective=",
            row.get("top_effective_chunk_id"),
            "| err=",
            bool(row.get("error")),
        )

    summary = summarize_results(eval_rows)
    bad_cases = build_bad_cases(eval_rows)

    tag = f"retrieval_v2_rankgpt_{rankgpt_mode}"

    detail_jsonl_path = output_dir / f"{tag}_details.jsonl"
    detail_xlsx_path = output_dir / f"{tag}_details.xlsx"
    summary_json_path = output_dir / f"{tag}_summary.json"
    bad_cases_jsonl_path = output_dir / f"{tag}_bad_cases.jsonl"
    bad_cases_xlsx_path = output_dir / f"{tag}_bad_cases.xlsx"

    save_jsonl(eval_rows, detail_jsonl_path)
    save_json(summary, summary_json_path)
    save_jsonl(bad_cases, bad_cases_jsonl_path)

    # Excel 中复杂字段转字符串，避免写入失败
    excel_rows = []
    for row in eval_rows:
        excel_row = {}
        for k, v in row.items():
            if isinstance(v, (list, dict)):
                excel_row[k] = json.dumps(make_json_safe(v), ensure_ascii=False)
            else:
                excel_row[k] = v
        excel_rows.append(excel_row)

    pd.DataFrame(excel_rows).to_excel(detail_xlsx_path, index=False)

    bad_excel_rows = []
    for row in bad_cases:
        excel_row = {}
        for k, v in row.items():
            if isinstance(v, (list, dict)):
                excel_row[k] = json.dumps(make_json_safe(v), ensure_ascii=False)
            else:
                excel_row[k] = v
        bad_excel_rows.append(excel_row)

    pd.DataFrame(bad_excel_rows).to_excel(bad_cases_xlsx_path, index=False)

    print("\n" + "=" * 120)
    print("评测完成")
    print("=" * 120)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n输出文件：")
    print(f"- details jsonl: {detail_jsonl_path}")
    print(f"- details xlsx : {detail_xlsx_path}")
    print(f"- summary json : {summary_json_path}")
    print(f"- bad jsonl    : {bad_cases_jsonl_path}")
    print(f"- bad xlsx     : {bad_cases_xlsx_path}")

    return {
        "summary": summary,
        "details": eval_rows,
        "bad_cases": bad_cases,
        "paths": {
            "detail_jsonl": str(detail_jsonl_path),
            "detail_xlsx": str(detail_xlsx_path),
            "summary_json": str(summary_json_path),
            "bad_cases_jsonl": str(bad_cases_jsonl_path),
            "bad_cases_xlsx": str(bad_cases_xlsx_path),
        },
    }


# ============================================================
# 11. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval / Rerank Eval V2 for routed Hydro GraphRAG.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="JSONL 测评集路径。",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_REPORT_DIR),
        help="评测报告输出目录。",
    )

    parser.add_argument(
        "--case-type",
        type=str,
        default="all",
        help="只评测某类问题，例如 fact / permeability_level / causal_explanation / multi_condition / fallback_unanswerable。",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="限制评测条数。0 表示不限制。",
    )

    parser.add_argument(
        "--raw-top-k",
        type=int,
        default=30,
        help="初始召回数量。",
    )

    parser.add_argument(
        "--final-top-k",
        type=int,
        default=10,
        help="最终重排数量。",
    )

    parser.add_argument(
        "--rankgpt",
        choices=["on", "off"],
        default="off",
        help="是否在 retrieval eval 中单独调用 RankGPT。默认 off，避免消耗 LLM。",
    )

    parser.add_argument(
        "--rankgpt-candidate-top-k",
        type=int,
        default=10,
        help="输入 RankGPT 的候选数量。",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="遇到单条样本错误时立即停止。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()

    limit = args.limit if args.limit and args.limit > 0 else None

    run_eval(
        dataset_path=dataset_path,
        output_dir=output_dir,
        case_type=args.case_type,
        limit=limit,
        raw_top_k=args.raw_top_k,
        final_top_k=args.final_top_k,
        rankgpt_mode=args.rankgpt,
        rankgpt_candidate_top_k=args.rankgpt_candidate_top_k,
        stop_on_error=args.stop_on_error,
    )


if __name__ == "__main__":
    main()
