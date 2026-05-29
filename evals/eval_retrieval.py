# evals/eval_retrieval.py
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
    """
    兼容 JSONL 中字段可能是 str / list / None 的情况。
    """
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
    当旧测试集没有 expected_answer_points / expected_keywords 时，
    从 expected_answer 中粗略提取一些可用于检索命中的关键词。

    注意：
    这只是兜底逻辑，不如人工维护 expected_answer_points 稳定。
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

    # 提取类似 ZK1、ZK01、ZK-1 之类的钻孔编号
    import re
    boreholes = re.findall(r"\bZK[-_]?\d+\b", expected_answer, flags=re.IGNORECASE)

    # 提取类似 12m、18.5m、12 米
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
# 检索结果解析
# =========================

def get_stage_chunks(result: Dict[str, Any], stage: str = "vector") -> List[Dict[str, Any]]:
    """
    兼容 qa_pipeline.py 可能返回的不同字段名。
    避免因为字段名不一致导致 B/C 阶段评估全为 0。
    """

    stage_key_candidates = {
        "vector": [
            "vector_search_results",
            "vector_results",
            "raw_vector_results",
            "retrieval_results",
            "initial_retrieval_results",
            "initial_results",
        ],
        "b": [
            "b_score_results",
            "b_scored_results",
            "b_results",
            "graph_expanded_results",
            "graph_expansion_results",
            "graph_results",
            "expanded_results",
        ],
        "c": [
            "c_rerank_results",
            "c_reranked_results",
            "c_results",
            "rerank_results",
            "reranked_results",
            "final_rerank_results",
            "final_results",
            "final_context_chunks",
            "context_chunks",
        ],
    }

    if stage not in stage_key_candidates:
        raise ValueError(f"未知 stage：{stage}")

    for key in stage_key_candidates[stage]:
        value = result.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]

    return []


def get_ranked_chunk_ids(result: Dict[str, Any], stage: str = "vector") -> List[str]:
    chunks = get_stage_chunks(result, stage=stage)

    ranked_ids: List[str] = []
    for x in chunks:
        chunk_id = x.get("chunk_id")
        if chunk_id:
            ranked_ids.append(str(chunk_id))

    return ranked_ids


def chunk_to_text(chunk: Dict[str, Any]) -> str:
    """
    从 chunk 中尽可能提取文本。
    兼容：
    - chunk_text
    - text
    - content
    - page_content
    - graph_context_text
    - metadata
    - node
    - document
    - nested dict/list
    """

    text_fields = [
        "chunk_text",
        "text",
        "content",
        "page_content",
        "graph_context_text",
        "context_text",
        "final_context",
        "lithology",
        "description",
        "name",
        "title",
    ]

    parts: List[str] = []

    for field in text_fields:
        value = chunk.get(field)
        if value:
            parts.append(str(value))

    def collect_nested_text(obj: Any, depth: int = 0) -> None:
        if depth > 3:
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k).lower()

                if key in {
                    "chunk_text",
                    "text",
                    "content",
                    "page_content",
                    "graph_context_text",
                    "context_text",
                    "lithology",
                    "description",
                    "name",
                    "title",
                    "major_lithology",
                    "minor_lithology",
                    "borehole_id",
                    "depth_range",
                }:
                    if v:
                        parts.append(str(v))

                elif isinstance(v, (dict, list)):
                    collect_nested_text(v, depth + 1)

        elif isinstance(obj, list):
            for item in obj:
                collect_nested_text(item, depth + 1)

    for nested_key in ["metadata", "node", "document", "data", "properties"]:
        if nested_key in chunk:
            collect_nested_text(chunk[nested_key])

    return "\n".join(str(x) for x in parts if str(x).strip())
def enrich_chunks_with_text_from_reference(
    chunks: List[Dict[str, Any]],
    reference_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    B/C 阶段结果有时只有 chunk_id + score，没有 chunk_text。
    这里用 vector 阶段的 chunk_id -> chunk_text 映射，把文本回填到 B/C 结果中。
    """

    id_to_reference: Dict[str, Dict[str, Any]] = {}

    for ref in reference_chunks:
        chunk_id = ref.get("chunk_id")
        if chunk_id:
            id_to_reference[str(chunk_id)] = ref

    enriched: List[Dict[str, Any]] = []

    for chunk in chunks:
        new_chunk = dict(chunk)
        chunk_id = new_chunk.get("chunk_id")

        current_text = chunk_to_text(new_chunk)

        if chunk_id and not current_text:
            ref = id_to_reference.get(str(chunk_id))
            if ref:
                ref_text = chunk_to_text(ref)
                if ref_text:
                    new_chunk["chunk_text"] = ref_text

                # 顺便补充 chunk_type
                if not new_chunk.get("chunk_type") and ref.get("chunk_type"):
                    new_chunk["chunk_type"] = ref.get("chunk_type")

        enriched.append(new_chunk)

    return enriched

def chunks_to_text(chunks: List[Dict[str, Any]], k: int) -> str:
    return "\n".join(chunk_to_text(x) for x in chunks[:k])


# =========================
# 指标函数：gold chunk id
# =========================

def hit_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> Optional[int]:
    if not gold_ids:
        return None
    return int(any(cid in gold_ids for cid in ranked_ids[:k]))


def recall_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> Optional[float]:
    if not gold_ids:
        return None

    hits = len(set(ranked_ids[:k]) & gold_ids)
    return round(hits / len(gold_ids), 4)


def mrr_at_k(ranked_ids: List[str], gold_ids: Set[str], k: int) -> Optional[float]:
    if not gold_ids:
        return None

    for idx, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_ids:
            return round(1.0 / idx, 4)

    return 0.0


def chunk_type_hit_at_k(
    chunks: List[Dict[str, Any]],
    gold_chunk_types: Set[str],
    k: int,
) -> Optional[int]:
    if not gold_chunk_types:
        return None

    retrieved_types = {
        str(x.get("chunk_type"))
        for x in chunks[:k]
        if x.get("chunk_type")
    }

    return int(bool(retrieved_types & gold_chunk_types))


# =========================
# 指标函数：没有 gold_chunk_ids 时的关键词证据命中
# =========================

def keyword_hit_at_k(
    chunks: List[Dict[str, Any]],
    expected_terms: List[str],
    k: int,
) -> Optional[int]:
    if not expected_terms:
        return None

    context = chunks_to_text(chunks, k=k)
    return int(any(term in context for term in expected_terms if term))


def keyword_recall_at_k(
    chunks: List[Dict[str, Any]],
    expected_terms: List[str],
    k: int,
) -> Optional[float]:
    if not expected_terms:
        return None

    context = chunks_to_text(chunks, k=k)
    hit_count = sum(1 for term in expected_terms if term and term in context)

    return round(hit_count / len(expected_terms), 4)


def safe_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 4)


# =========================
# 单样本评估
# =========================

def evaluate_one_case(
    case: Dict[str, Any],
    raw_top_k: int,
    final_top_k: int,
) -> Dict[str, Any]:
    normalized_case = normalize_eval_case(case)

    question = normalized_case["question"]
    gold_ids = set(normalized_case.get("gold_chunk_ids", []))
    gold_types = set(normalized_case.get("gold_chunk_types", []))
    expected_terms = dedupe_keep_order(
        as_list(normalized_case.get("expected_answer_points"))
        + as_list(normalized_case.get("must_contain"))
    )

    should_retrieve = bool(normalized_case.get("should_retrieve", True))

    result = run_end_to_end_graphrag_qa(
        user_question=question,
        raw_top_k=raw_top_k,
        final_top_k=final_top_k,
        save_outputs=False,
        enable_llm=False,
    )

    vector_chunks = get_stage_chunks(result, stage="vector")
    b_chunks_raw = get_stage_chunks(result, stage="b")
    c_chunks_raw = get_stage_chunks(result, stage="c")

    # B/C 阶段可能只有 chunk_id，没有 chunk_text；
    # 因此用 vector 阶段结果按 chunk_id 回填文本。
    b_chunks = enrich_chunks_with_text_from_reference(
        chunks=b_chunks_raw,
        reference_chunks=vector_chunks,
    )
    c_chunks = enrich_chunks_with_text_from_reference(
        chunks=c_chunks_raw,
        reference_chunks=vector_chunks + b_chunks,
)

    vector_ids = get_ranked_chunk_ids(result, stage="vector")
    b_ids = get_ranked_chunk_ids(result, stage="b")
    c_ids = get_ranked_chunk_ids(result, stage="c")

    # should_retrieve=False 的样本不参与标准 retrieval 平均分。
    # 但仍保存检索结果，方便观察是否检索到了无关内容。
    compute_standard_retrieval_metrics = should_retrieve

    if compute_standard_retrieval_metrics:
        vector_hit_1 = hit_at_k(vector_ids, gold_ids, 1)
        vector_hit_3 = hit_at_k(vector_ids, gold_ids, 3)
        vector_hit_5 = hit_at_k(vector_ids, gold_ids, 5)
        vector_hit_10 = hit_at_k(vector_ids, gold_ids, 10)

        vector_recall_3 = recall_at_k(vector_ids, gold_ids, 3)
        vector_recall_5 = recall_at_k(vector_ids, gold_ids, 5)
        vector_recall_10 = recall_at_k(vector_ids, gold_ids, 10)
        vector_mrr_10 = mrr_at_k(vector_ids, gold_ids, 10)

        b_hit_3 = hit_at_k(b_ids, gold_ids, 3)
        b_hit_5 = hit_at_k(b_ids, gold_ids, 5)
        b_hit_10 = hit_at_k(b_ids, gold_ids, 10)
        b_recall_10 = recall_at_k(b_ids, gold_ids, 10)
        b_mrr_10 = mrr_at_k(b_ids, gold_ids, 10)

        c_hit_3 = hit_at_k(c_ids, gold_ids, 3)
        c_hit_5 = hit_at_k(c_ids, gold_ids, 5)
        c_hit_10 = hit_at_k(c_ids, gold_ids, 10)
        c_recall_10 = recall_at_k(c_ids, gold_ids, 10)
        c_mrr_10 = mrr_at_k(c_ids, gold_ids, 10)
    else:
        vector_hit_1 = vector_hit_3 = vector_hit_5 = vector_hit_10 = None
        vector_recall_3 = vector_recall_5 = vector_recall_10 = None
        vector_mrr_10 = None

        b_hit_3 = b_hit_5 = b_hit_10 = None
        b_recall_10 = b_mrr_10 = None

        c_hit_3 = c_hit_5 = c_hit_10 = None
        c_recall_10 = c_mrr_10 = None

    row = {
        "id": normalized_case.get("id"),
        "category": normalized_case.get("category"),
        "question": question,
        "answerable": normalized_case.get("answerable"),
        "should_retrieve": should_retrieve,
        "should_abstain": normalized_case.get("should_abstain"),

        "gold_chunk_ids": list(gold_ids),
        "gold_chunk_types": list(gold_types),
        "expected_terms": expected_terms,

        "retrieval_evaluable_by_gold_ids": bool(gold_ids),
        "retrieval_evaluable_by_keywords": bool(expected_terms),

        "vector_top_ids": vector_ids[:10],
        "b_top_ids": b_ids[:10],
        "c_top_ids": c_ids[:10],
                "vector_count": len(vector_chunks),
        "b_count": len(b_chunks),
        "c_count": len(c_chunks),

        "vector_text_chars@5": len(chunks_to_text(vector_chunks, 5)),
        "b_text_chars@5": len(chunks_to_text(b_chunks, 5)),
        "c_text_chars@5": len(chunks_to_text(c_chunks, 5)),

        "vector_text_preview@1": chunks_to_text(vector_chunks, 1)[:300],
        "b_text_preview@1": chunks_to_text(b_chunks, 1)[:300],
        "c_text_preview@1": chunks_to_text(c_chunks, 1)[:300],

        # 标准 gold_chunk_ids 指标
        "vector_hit@1": vector_hit_1,
        "vector_hit@3": vector_hit_3,
        "vector_hit@5": vector_hit_5,
        "vector_hit@10": vector_hit_10,
        "vector_recall@3": vector_recall_3,
        "vector_recall@5": vector_recall_5,
        "vector_recall@10": vector_recall_10,
        "vector_mrr@10": vector_mrr_10,

        "b_hit@3": b_hit_3,
        "b_hit@5": b_hit_5,
        "b_hit@10": b_hit_10,
        "b_recall@10": b_recall_10,
        "b_mrr@10": b_mrr_10,

        "c_hit@3": c_hit_3,
        "c_hit@5": c_hit_5,
        "c_hit@10": c_hit_10,
        "c_recall@10": c_recall_10,
        "c_mrr@10": c_mrr_10,

        # 没有 gold_chunk_ids 时的关键词证据命中指标
        "vector_keyword_hit@5": keyword_hit_at_k(vector_chunks, expected_terms, 5) if should_retrieve else None,
        "vector_keyword_recall@10": keyword_recall_at_k(vector_chunks, expected_terms, 10) if should_retrieve else None,
        "b_keyword_hit@5": keyword_hit_at_k(b_chunks, expected_terms, 5) if should_retrieve else None,
        "b_keyword_recall@10": keyword_recall_at_k(b_chunks, expected_terms, 10) if should_retrieve else None,
        "c_keyword_hit@5": keyword_hit_at_k(c_chunks, expected_terms, 5) if should_retrieve else None,
        "c_keyword_recall@10": keyword_recall_at_k(c_chunks, expected_terms, 10) if should_retrieve else None,

        # chunk_type 命中
        "vector_chunk_type_hit@5": chunk_type_hit_at_k(vector_chunks, gold_types, 5) if should_retrieve else None,
        "b_chunk_type_hit@5": chunk_type_hit_at_k(b_chunks, gold_types, 5) if should_retrieve else None,
        "c_chunk_type_hit@5": chunk_type_hit_at_k(c_chunks, gold_types, 5) if should_retrieve else None,

        # GraphRAG / Rerank 增益
        "graph_gain_hit@5": safe_delta(b_hit_5, vector_hit_5),
        "graph_gain_recall@10": safe_delta(b_recall_10, vector_recall_10),
        "graph_gain_mrr@10": safe_delta(b_mrr_10, vector_mrr_10),

        "rerank_gain_hit@5": safe_delta(c_hit_5, b_hit_5),
        "rerank_gain_recall@10": safe_delta(c_recall_10, b_recall_10),
        "rerank_gain_mrr@10": safe_delta(c_mrr_10, b_mrr_10),

        "end_to_end_gain_hit@5": safe_delta(c_hit_5, vector_hit_5),
        "end_to_end_gain_recall@10": safe_delta(c_recall_10, vector_recall_10),
        "end_to_end_gain_mrr@10": safe_delta(c_mrr_10, vector_mrr_10),

        "confidence": result.get("confidence"),
        "fallback_triggered": result.get("fallback_triggered"),
        "error": None,

    }

    return make_json_safe(row)


def evaluate_one_case_safely(
    case: Dict[str, Any],
    raw_top_k: int,
    final_top_k: int,
    fail_fast: bool,
) -> Dict[str, Any]:
    try:
        return evaluate_one_case(
            case=case,
            raw_top_k=raw_top_k,
            final_top_k=final_top_k,
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
    summary: Dict[str, Any] = {}

    summary["case_count"] = int(len(df))
    summary["error_count"] = int(df["error"].notna().sum()) if "error" in df.columns else 0

    if "should_retrieve" in df.columns:
        retrieve_df = df[df["should_retrieve"] == True].copy()
        non_retrieve_df = df[df["should_retrieve"] == False].copy()
    else:
        retrieve_df = df.copy()
        non_retrieve_df = df.iloc[0:0].copy()

    summary["should_retrieve_case_count"] = int(len(retrieve_df))
    summary["should_not_retrieve_case_count"] = int(len(non_retrieve_df))

    metric_cols = [
        "vector_hit@1",
        "vector_hit@3",
        "vector_hit@5",
        "vector_hit@10",
        "vector_recall@3",
        "vector_recall@5",
        "vector_recall@10",
        "vector_mrr@10",

        "b_hit@3",
        "b_hit@5",
        "b_hit@10",
        "b_recall@10",
        "b_mrr@10",

        "c_hit@3",
        "c_hit@5",
        "c_hit@10",
        "c_recall@10",
        "c_mrr@10",

        "vector_keyword_hit@5",
        "vector_keyword_recall@10",
        "b_keyword_hit@5",
        "b_keyword_recall@10",
        "c_keyword_hit@5",
        "c_keyword_recall@10",

        "vector_chunk_type_hit@5",
        "b_chunk_type_hit@5",
        "c_chunk_type_hit@5",

        "graph_gain_hit@5",
        "graph_gain_recall@10",
        "graph_gain_mrr@10",

        "rerank_gain_hit@5",
        "rerank_gain_recall@10",
        "rerank_gain_mrr@10",

        "end_to_end_gain_hit@5",
        "end_to_end_gain_recall@10",
        "end_to_end_gain_mrr@10",
    ]

    for col in metric_cols:
        summary[col] = mean_metric(retrieve_df, col)

    # 分类型统计
    category_summary: Dict[str, Any] = {}
    if "category" in retrieve_df.columns:
        for category, group in retrieve_df.groupby("category"):
            category_summary[str(category)] = {
                "case_count": int(len(group)),
                "vector_hit@5": mean_metric(group, "vector_hit@5"),
                "vector_keyword_hit@5": mean_metric(group, "vector_keyword_hit@5"),
                "c_hit@5": mean_metric(group, "c_hit@5"),
                "c_keyword_hit@5": mean_metric(group, "c_keyword_hit@5"),
                "c_recall@10": mean_metric(group, "c_recall@10"),
                "c_mrr@10": mean_metric(group, "c_mrr@10"),
                "end_to_end_gain_hit@5": mean_metric(group, "end_to_end_gain_hit@5"),
            }

    summary["category_summary"] = category_summary

    return make_json_safe(summary)


# =========================
# CLI
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Retrieval / GraphRAG / Rerank 离线测评脚本")

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(resolve_default_dataset_path()),
        help="测试集 JSONL 路径，默认指向 evals/datasets/rag_mvp_eval_50.jsonl",
    )
    parser.add_argument("--raw-top-k", type=int, default=30)
    parser.add_argument("--final-top-k", type=int, default=10)
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
        print(f"[eval_retrieval] normalized dataset saved to: {export_path}")

    if args.only_normalize:
        return

    rows = []
    for idx, case in enumerate(normalized_cases, start=1):
        print(
            f"[eval_retrieval] running {idx}/{len(normalized_cases)} | "
            f"{case.get('id')} | {case.get('category')} | {case.get('question')}"
        )

        row = evaluate_one_case_safely(
            case=case,
            raw_top_k=args.raw_top_k,
            final_top_k=args.final_top_k,
            fail_fast=args.fail_fast,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = summarize_metrics(df)

    detail_csv_path = output_dir / "retrieval_eval_detail.csv"
    detail_jsonl_path = output_dir / "retrieval_eval_detail.jsonl"
    summary_path = output_dir / "retrieval_eval_summary.json"

    df.to_csv(detail_csv_path, index=False, encoding="utf-8-sig")
    write_jsonl(detail_jsonl_path, rows)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Retrieval Eval Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDetail CSV saved to: {detail_csv_path}")
    print(f"Detail JSONL saved to: {detail_jsonl_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()