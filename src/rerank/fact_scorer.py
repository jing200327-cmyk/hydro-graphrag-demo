from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.query.entity_extractor import extract_depth_expressions
from src.retrieval.exact_retriever import extract_exact_ids, normalize_exact_id
from src.utils.text import safe_text, unique_keep_order, make_json_safe


# ============================================================
# 1. 事实查询评分权重
# ============================================================

FACT_SCORE_WEIGHTS: Dict[str, float] = {
    "layer_id": 50.0,
    "borehole_id": 15.0,
    "depth": 15.0,
    "lithology": 15.0,
    "semantic": 5.0,
}


LITHOLOGY_FIELD_KEYWORDS = [
    "岩性",
    "岩性大类",
    "岩性小类",
    "土层",
    "地层",
    "岩土",
]


LITHOLOGY_VALUE_KEYWORDS = [
    "粉质黏土",
    "粉质粘土",
    "黏土",
    "粘土",
    "淤泥",
    "粉土",
    "粉砂",
    "细砂",
    "中砂",
    "粗砂",
    "砂土",
    "砂",
    "砾砂",
    "含砾砂",
    "圆砾",
    "卵石",
    "砾石",
    "碎石",
    "强风化",
    "中风化",
    "弱风化",
    "全风化",
    "泥岩",
    "砂岩",
    "灰岩",
]


LITHOLOGY_PROP_KEYS = [
    "lithology_major_v2",
    "lithology_minor_v3",
    "lithology_major",
    "lithology_minor",
    "lithology",
    "rock_type",
    "soil_type",
]


DEPTH_PROP_KEY_PAIRS = [
    ("top_depth", "bottom_depth"),
    ("depth_top", "depth_bottom"),
    ("layer_top", "layer_bottom"),
    ("start_depth", "end_depth"),
    ("top", "bottom"),
    ("层顶", "层底"),
]


DEPTH_SINGLE_KEYS = [
    "depth",
    "middle_depth",
    "center_depth",
    "埋深",
    "中间深度",
]


# ============================================================
# 2. 通用工具函数
# ============================================================

def _clamp(value: float, min_value: float = 0.0, max_value: float = 100.0) -> float:
    return max(min_value, min(max_value, value))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    text = safe_text(value)

    if not text:
        return None

    match = re.search(r"-?\d+\.?\d*", text)

    if not match:
        return None

    try:
        return float(match.group(0))
    except Exception:
        return None


def _flatten_values(value: Any) -> List[str]:
    """
    将 dict / list / tuple / set / 标量统一展开成字符串列表，
    用于构造匹配文本。
    """

    results: List[str] = []

    if value is None:
        return results

    if isinstance(value, dict):
        for k, v in value.items():
            results.append(safe_text(k))
            results.extend(_flatten_values(v))
        return results

    if isinstance(value, (list, tuple, set)):
        for item in value:
            results.extend(_flatten_values(item))
        return results

    text = safe_text(value)

    if text:
        results.append(text)

    return results


def _normalize_match_text(text: str) -> str:
    """
    用于 ID 匹配的标准化文本：
    - 去空白
    - 转大写
    """
    return re.sub(r"\s+", "", safe_text(text)).upper()


def _contains_exact_id(text: str, exact_id: str) -> bool:
    """
    判断文本中是否包含完整 ID。

    避免 CHGC001_1 错误命中 CHGC001_10。
    """

    normalized_text = _normalize_match_text(text)
    normalized_id = normalize_exact_id(exact_id)

    if not normalized_text or not normalized_id:
        return False

    pattern = rf"(?<![A-Z0-9_]){re.escape(normalized_id)}(?![A-Z0-9_])"

    return bool(re.search(pattern, normalized_text))


def _chunk_match_text(chunk: Dict[str, Any]) -> str:
    """
    将 chunk 中所有可能含有匹配信息的字段拼成一段文本。
    """

    values: List[str] = []

    candidate_keys = [
        "chunk_id",
        "chunk_type",
        "chunk_text",
        "source_label",
        "source_id",
        "retrieval_method",
        "exact_match_type",
        "exact_match_value",
    ]

    for key in candidate_keys:
        values.extend(_flatten_values(chunk.get(key)))

    values.extend(_flatten_values(chunk.get("source_props")))
    values.extend(_flatten_values(chunk.get("metadata")))
    values.extend(_flatten_values(chunk.get("graph_context")))

    return " ".join([safe_text(v) for v in values if safe_text(v)])


def _get_source_props(chunk: Dict[str, Any]) -> Dict[str, Any]:
    props = chunk.get("source_props")

    if isinstance(props, dict):
        return props

    return {}


def _get_vector_score(chunk: Dict[str, Any]) -> float:
    """
    读取语义相似度。

    兼容：
    - vector_score: 0~1
    - vector_score: 0~100
    - score: 0~1
    """

    raw = chunk.get("vector_score", chunk.get("score", 0.0))

    try:
        value = float(raw)
    except Exception:
        return 0.0

    if value > 1.0:
        value = value / 100.0

    return _clamp(value, 0.0, 1.0)


# ============================================================
# 3. 问题侧约束抽取
# ============================================================

def _extract_lithology_values_from_entities(
    entities: Optional[List[Dict[str, Any]]],
) -> List[str]:
    if not entities:
        return []

    values: List[str] = []

    for entity in entities:
        entity_type = safe_text(entity.get("实体类型", ""))
        entity_value = safe_text(entity.get("实体值", ""))

        if not entity_value:
            continue

        if entity_type in {"岩性大类", "岩性小类"}:
            values.append(entity_value)

    return unique_keep_order(values)


def _extract_lithology_values_from_question(user_question: str) -> List[str]:
    question = safe_text(user_question)

    values: List[str] = []

    for keyword in LITHOLOGY_VALUE_KEYWORDS:
        if keyword and keyword in question:
            values.append(keyword)

    return unique_keep_order(values)


def _is_lithology_field_requested(user_question: str) -> bool:
    question = safe_text(user_question)
    return any(keyword in question for keyword in LITHOLOGY_FIELD_KEYWORDS)


def _extract_depth_constraints(user_question: str) -> Dict[str, Any]:
    """
    抽取深度约束。

    返回结构固定为：
    {
        "ranges": [(12.0, 18.0, "12m 到 18m")],
        "points": [(12.0, "12m")]
    }

    注意：
    - ranges 永远是三元组：(left, right, expr)
    - points 永远是二元组：(point, expr)
    """

    question = safe_text(user_question)

    ranges: List[Tuple[float, float, str]] = []
    points: List[Tuple[float, str]] = []

    consumed_spans: List[Tuple[int, int]] = []

    # 1. 优先抽取范围表达：12m 到 18m / 12-18m / 12~18米
    range_pattern = re.compile(
        r"(\d+\.?\d*)\s*(?:m|米)?\s*(?:-|~|—|–|至|到)\s*(\d+\.?\d*)\s*(?:m|米)?",
        re.IGNORECASE,
    )

    for match in range_pattern.finditer(question):
        start = _safe_float(match.group(1))
        end = _safe_float(match.group(2))

        if start is None or end is None:
            continue

        left = min(start, end)
        right = max(start, end)

        ranges.append((left, right, match.group(0)))
        consumed_spans.append(match.span())

    def _inside_consumed_span(pos: int) -> bool:
        for left, right in consumed_spans:
            if left <= pos < right:
                return True
        return False

    # 2. 再抽取单点表达：12m / 18米
    # 已经属于范围表达的数字不再加入 points。
    point_pattern = re.compile(
        r"(\d+\.?\d*)\s*(?:m|米)",
        re.IGNORECASE,
    )

    for match in point_pattern.finditer(question):
        if _inside_consumed_span(match.start()):
            continue

        value = _safe_float(match.group(1))

        if value is None:
            continue

        points.append((value, match.group(0)))

    # 3. 兼容 extract_depth_expressions 的输出。
    # 它可能会把 “12m 到 18m” 和 “12m”“18m” 都抽出来。
    # 所以这里要严格区分 range / point，避免三元组混入 points。
    for expr in extract_depth_expressions(question):
        expr_text = safe_text(expr)

        if not expr_text:
            continue

        # 如果该表达本身是范围表达，加入 ranges，而不是 points
        range_match = range_pattern.search(expr_text)

        if range_match:
            start = _safe_float(range_match.group(1))
            end = _safe_float(range_match.group(2))

            if start is None or end is None:
                continue

            left = min(start, end)
            right = max(start, end)
            range_tuple = (left, right, expr_text)

            if range_tuple not in ranges:
                ranges.append(range_tuple)

            continue

        # 否则作为单点表达
        value = _safe_float(expr_text)

        if value is None:
            continue

        point_tuple = (value, expr_text)

        if point_tuple not in points:
            points.append(point_tuple)

    return {
        "ranges": ranges,
        "points": points,
    }

def collect_fact_query_constraints(
    user_question: str,
    entities: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    汇总事实查询评分所需的约束。

    包括：
    - layer_ids
    - borehole_ids
    - depth_constraints
    - lithology_values
    - lithology_field_requested
    """

    exact_ids = extract_exact_ids(
        user_question=user_question,
        entities=entities,
    )

    lithology_values = unique_keep_order(
        _extract_lithology_values_from_entities(entities)
        + _extract_lithology_values_from_question(user_question)
    )

    depth_constraints = _extract_depth_constraints(user_question)

    return {
        "layer_ids": exact_ids.get("layer_ids", []),
        "borehole_ids": exact_ids.get("borehole_ids", []),
        "depth_constraints": depth_constraints,
        "lithology_values": lithology_values,
        "lithology_field_requested": _is_lithology_field_requested(user_question),
    }


# ============================================================
# 4. 各维度评分
# ============================================================

def _score_layer_id_match(
    constraints: Dict[str, Any],
    chunk: Dict[str, Any],
) -> Tuple[float, List[str]]:
    layer_ids = constraints.get("layer_ids", [])

    if not layer_ids:
        return 0.0, []

    text = _chunk_match_text(chunk)
    props = _get_source_props(chunk)

    reasons: List[str] = []

    exact_match_type = safe_text(chunk.get("exact_match_type", ""))
    exact_match_value = normalize_exact_id(chunk.get("exact_match_value", ""))

    for layer_id in layer_ids:
        normalized_layer_id = normalize_exact_id(layer_id)

        if exact_match_type == "layer_id" and exact_match_value == normalized_layer_id:
            reasons.append(f"exact_match_value 命中 layer_id={normalized_layer_id}")
            return FACT_SCORE_WEIGHTS["layer_id"], reasons

        prop_layer_id = normalize_exact_id(props.get("layer_id", ""))

        if prop_layer_id == normalized_layer_id:
            reasons.append(f"source_props.layer_id 命中 {normalized_layer_id}")
            return FACT_SCORE_WEIGHTS["layer_id"], reasons

        if _contains_exact_id(text, normalized_layer_id):
            reasons.append(f"chunk 文本或元数据命中 layer_id={normalized_layer_id}")
            return FACT_SCORE_WEIGHTS["layer_id"], reasons

    return 0.0, reasons


def _score_borehole_id_match(
    constraints: Dict[str, Any],
    chunk: Dict[str, Any],
) -> Tuple[float, List[str]]:
    borehole_ids = constraints.get("borehole_ids", [])

    if not borehole_ids:
        return 0.0, []

    text = _chunk_match_text(chunk)
    props = _get_source_props(chunk)

    reasons: List[str] = []

    exact_match_type = safe_text(chunk.get("exact_match_type", ""))
    exact_match_value = normalize_exact_id(chunk.get("exact_match_value", ""))

    prop_borehole_id = normalize_exact_id(props.get("borehole_id", ""))
    prop_layer_id = normalize_exact_id(props.get("layer_id", ""))

    for borehole_id in borehole_ids:
        normalized_borehole_id = normalize_exact_id(borehole_id)

        if exact_match_type == "borehole_id" and exact_match_value == normalized_borehole_id:
            reasons.append(f"exact_match_value 命中 borehole_id={normalized_borehole_id}")
            return FACT_SCORE_WEIGHTS["borehole_id"], reasons

        if prop_borehole_id == normalized_borehole_id:
            reasons.append(f"source_props.borehole_id 命中 {normalized_borehole_id}")
            return FACT_SCORE_WEIGHTS["borehole_id"], reasons

        # layer_id 可以反推 borehole_id，例如 CHGC001_2 -> CHGC001
        if prop_layer_id.startswith(normalized_borehole_id + "_"):
            reasons.append(f"source_props.layer_id 反推命中 borehole_id={normalized_borehole_id}")
            return FACT_SCORE_WEIGHTS["borehole_id"], reasons

        if exact_match_type == "layer_id" and exact_match_value.startswith(normalized_borehole_id + "_"):
            reasons.append(f"exact layer_id 反推命中 borehole_id={normalized_borehole_id}")
            return FACT_SCORE_WEIGHTS["borehole_id"], reasons

        # 对 borehole_id 不强制完整边界，因为 chunk_text 里可能只有 CHGC001_2，
        # 但 CHGC001 仍然是其钻孔前缀。
        normalized_text = _normalize_match_text(text)

        if normalized_borehole_id in normalized_text:
            reasons.append(f"chunk 文本或元数据命中 borehole_id={normalized_borehole_id}")
            return FACT_SCORE_WEIGHTS["borehole_id"], reasons

    return 0.0, reasons


def _get_chunk_depth_intervals(
    chunk: Dict[str, Any],
) -> List[Tuple[float, float, str]]:
    props = _get_source_props(chunk)

    intervals: List[Tuple[float, float, str]] = []

    for top_key, bottom_key in DEPTH_PROP_KEY_PAIRS:
        top_value = _safe_float(props.get(top_key))
        bottom_value = _safe_float(props.get(bottom_key))

        if top_value is None or bottom_value is None:
            continue

        left = min(top_value, bottom_value)
        right = max(top_value, bottom_value)

        intervals.append((left, right, f"{top_key}-{bottom_key}"))

    return intervals


def _get_chunk_depth_points(
    chunk: Dict[str, Any],
) -> List[Tuple[float, str]]:
    props = _get_source_props(chunk)

    points: List[Tuple[float, str]] = []

    for key in DEPTH_SINGLE_KEYS:
        value = _safe_float(props.get(key))

        if value is not None:
            points.append((value, key))

    return points


def _interval_overlap(
    a_left: float,
    a_right: float,
    b_left: float,
    b_right: float,
) -> bool:
    return max(a_left, b_left) <= min(a_right, b_right)


def _interval_contains(
    outer_left: float,
    outer_right: float,
    inner_left: float,
    inner_right: float,
) -> bool:
    return outer_left <= inner_left and inner_right <= outer_right


def _score_depth_match(
    constraints: Dict[str, Any],
    chunk: Dict[str, Any],
) -> Tuple[float, List[str]]:
    """
    深度字段匹配评分。

    支持：
    - ranges: [(left, right, expr)]
    - points: [(point, expr)]

    该版本会防御 points 中误混入 range 三元组的情况。
    """

    depth_constraints = constraints.get("depth_constraints", {}) or {}

    raw_ranges = depth_constraints.get("ranges", []) or []
    raw_points = depth_constraints.get("points", []) or []

    ranges: List[Tuple[float, float, str]] = []
    points: List[Tuple[float, str]] = []

    # 1. 清洗 ranges，确保都是三元组
    for item in raw_ranges:
        if not isinstance(item, (list, tuple)):
            continue

        if len(item) < 3:
            continue

        left = _safe_float(item[0])
        right = _safe_float(item[1])
        expr = safe_text(item[2])

        if left is None or right is None:
            continue

        ranges.append((min(left, right), max(left, right), expr))

    # 2. 清洗 points
    # 正常 point 是二元组：(point, expr)
    # 如果误传入三元组：(left, right, expr)，则转移到 ranges
    for item in raw_points:
        if not isinstance(item, (list, tuple)):
            value = _safe_float(item)

            if value is not None:
                points.append((value, safe_text(item)))

            continue

        if len(item) >= 3:
            left = _safe_float(item[0])
            right = _safe_float(item[1])
            expr = safe_text(item[2])

            if left is not None and right is not None:
                ranges.append((min(left, right), max(left, right), expr))

            continue

        if len(item) == 2:
            value = _safe_float(item[0])
            expr = safe_text(item[1])

            if value is not None:
                points.append((value, expr))

    if not ranges and not points:
        return 0.0, []

    reasons: List[str] = []

    intervals = _get_chunk_depth_intervals(chunk)
    depth_points = _get_chunk_depth_points(chunk)
    text = _chunk_match_text(chunk)

    best_score = 0.0

    # 3. 匹配深度范围，例如：12m 到 18m
    for q_left, q_right, expr in ranges:
        for c_left, c_right, source in intervals:
            if _interval_contains(c_left, c_right, q_left, q_right):
                best_score = max(best_score, FACT_SCORE_WEIGHTS["depth"])
                reasons.append(
                    f"深度范围 {expr} 被 chunk 深度区间 {c_left}-{c_right} 覆盖，来源={source}"
                )

            elif _interval_overlap(q_left, q_right, c_left, c_right):
                partial_score = FACT_SCORE_WEIGHTS["depth"] * 0.8
                best_score = max(best_score, partial_score)
                reasons.append(
                    f"深度范围 {expr} 与 chunk 深度区间 {c_left}-{c_right} 有交集，来源={source}"
                )

        if expr and expr in text:
            best_score = max(best_score, FACT_SCORE_WEIGHTS["depth"])
            reasons.append(f"chunk 文本直接命中深度表达：{expr}")

    # 4. 匹配深度点，例如：12m
    for q_point, expr in points:
        for c_left, c_right, source in intervals:
            if c_left <= q_point <= c_right:
                best_score = max(best_score, FACT_SCORE_WEIGHTS["depth"])
                reasons.append(
                    f"深度点 {expr} 落在 chunk 深度区间 {c_left}-{c_right}，来源={source}"
                )

        for c_point, source in depth_points:
            if abs(c_point - q_point) <= 1e-6:
                best_score = max(best_score, FACT_SCORE_WEIGHTS["depth"])
                reasons.append(
                    f"深度点 {expr} 与 chunk 深度字段 {source}={c_point} 精确一致"
                )

        if expr and expr in text:
            best_score = max(best_score, FACT_SCORE_WEIGHTS["depth"])
            reasons.append(f"chunk 文本直接命中深度表达：{expr}")

    return _clamp(
        best_score,
        0.0,
        FACT_SCORE_WEIGHTS["depth"],
    ), unique_keep_order(reasons)


def _score_lithology_match(
    constraints: Dict[str, Any],
    chunk: Dict[str, Any],
) -> Tuple[float, List[str]]:
    lithology_values = constraints.get("lithology_values", [])
    lithology_field_requested = bool(constraints.get("lithology_field_requested", False))

    if not lithology_values and not lithology_field_requested:
        return 0.0, []

    text = _chunk_match_text(chunk)
    props = _get_source_props(chunk)

    reasons: List[str] = []

    # 1. 用户问题中出现了明确岩性值，例如“粉质黏土”
    for lithology_value in lithology_values:
        value = safe_text(lithology_value)

        if not value:
            continue

        if value in text:
            reasons.append(f"chunk 文本或属性命中岩性值：{value}")
            return FACT_SCORE_WEIGHTS["lithology"], reasons

    # 2. 用户只是问“岩性是什么”，则只要求 chunk 中存在岩性字段
    if lithology_field_requested:
        for key in LITHOLOGY_PROP_KEYS:
            if props.get(key) is not None and safe_text(props.get(key)):
                reasons.append(f"source_props 存在岩性字段：{key}={props.get(key)}")
                return FACT_SCORE_WEIGHTS["lithology"], reasons

        if any(keyword in text for keyword in LITHOLOGY_FIELD_KEYWORDS):
            reasons.append("chunk 文本包含岩性字段关键词")
            return FACT_SCORE_WEIGHTS["lithology"], reasons

        if any(keyword in text for keyword in LITHOLOGY_VALUE_KEYWORDS):
            reasons.append("chunk 文本包含常见岩性值")
            return FACT_SCORE_WEIGHTS["lithology"] * 0.8, reasons

    return 0.0, reasons


def _score_semantic_similarity(
    chunk: Dict[str, Any],
) -> Tuple[float, List[str]]:
    vector_score = _get_vector_score(chunk)
    semantic_score = vector_score * FACT_SCORE_WEIGHTS["semantic"]

    return semantic_score, [f"vector_score={vector_score:.4f}"]


# ============================================================
# 5. 单条 chunk 事实评分
# ============================================================

def _build_active_dimensions(
    constraints: Dict[str, Any],
) -> List[str]:
    active: List[str] = []

    if constraints.get("layer_ids"):
        active.append("layer_id")

    if constraints.get("borehole_ids"):
        active.append("borehole_id")

    depth_constraints = constraints.get("depth_constraints", {}) or {}

    if depth_constraints.get("ranges") or depth_constraints.get("points"):
        active.append("depth")

    if constraints.get("lithology_values") or constraints.get("lithology_field_requested"):
        active.append("lithology")

    # 语义相似度始终作为弱辅助信号
    active.append("semantic")

    return unique_keep_order(active)


def _normalize_raw_score(
    raw_score: float,
    active_dimensions: List[str],
) -> Tuple[float, float]:
    """
    将 raw_score 按本题实际激活的维度归一化到 0~100。

    例如：
    问题是 “CHGC001_2 的岩性是什么？”
    激活维度为：
    - layer_id: 50
    - borehole_id: 15
    - lithology: 15
    - semantic: 5

    active_weight = 85
    如果全部命中，raw_score=85，final_score=100。
    """

    active_weight = sum(FACT_SCORE_WEIGHTS.get(dim, 0.0) for dim in active_dimensions)

    if active_weight <= 0:
        return 0.0, 0.0

    final_score = raw_score / active_weight * 100.0

    return _clamp(final_score, 0.0, 100.0), active_weight


def score_single_fact_chunk(
    user_question: str,
    chunk: Dict[str, Any],
    entities: Optional[List[Dict[str, Any]]] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    对单条 retrieved chunk 进行 fact_query 专用评分。

    返回结果会保留原始 chunk 字段，并新增：
    - fact_raw_score
    - final_score
    - layer_id_score
    - borehole_id_score
    - depth_score
    - lithology_score
    - semantic_score
    - active_dimensions
    - matched_fields
    - rerank_reason
    """

    if constraints is None:
        constraints = collect_fact_query_constraints(
            user_question=user_question,
            entities=entities,
        )

    active_dimensions = _build_active_dimensions(constraints)

    layer_score, layer_reasons = _score_layer_id_match(constraints, chunk)
    borehole_score, borehole_reasons = _score_borehole_id_match(constraints, chunk)
    depth_score, depth_reasons = _score_depth_match(constraints, chunk)
    lithology_score, lithology_reasons = _score_lithology_match(constraints, chunk)
    semantic_score, semantic_reasons = _score_semantic_similarity(chunk)

    raw_score = (
        layer_score
        + borehole_score
        + depth_score
        + lithology_score
        + semantic_score
    )

    final_score, active_weight = _normalize_raw_score(
        raw_score=raw_score,
        active_dimensions=active_dimensions,
    )

    matched_fields: List[str] = []

    if layer_score > 0:
        matched_fields.append("layer_id")

    if borehole_score > 0:
        matched_fields.append("borehole_id")

    if depth_score > 0:
        matched_fields.append("depth")

    if lithology_score > 0:
        matched_fields.append("lithology")

    if semantic_score > 0:
        matched_fields.append("semantic")

    all_reasons = (
        layer_reasons
        + borehole_reasons
        + depth_reasons
        + lithology_reasons
        + semantic_reasons
    )

    scored = dict(chunk)

    scored.update(
        {
            "fact_raw_score": round(raw_score, 4),
            "final_score": round(final_score, 4),
            "active_weight": round(active_weight, 4),
            "active_dimensions": active_dimensions,
            "layer_id_score": round(layer_score, 4),
            "borehole_id_score": round(borehole_score, 4),
            "depth_score": round(depth_score, 4),
            "lithology_score": round(lithology_score, 4),
            "semantic_score": round(semantic_score, 4),
            "matched_fields": matched_fields,
            "fact_match_details": unique_keep_order(all_reasons),
            "rerank_reason": "；".join(unique_keep_order(all_reasons)),
            "use_for_context": final_score > 0,
            "scoring_method": "fact_query_rule_score",
            "score_weights": make_json_safe(FACT_SCORE_WEIGHTS),
        }
    )

    return scored


# ============================================================
# 6. 批量评分与重排序入口
# ============================================================

def fact_score_chunks(
    user_question: str,
    chunks: List[Dict[str, Any]],
    entities: Optional[List[Dict[str, Any]]] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    fact_query 专用批量评分入口。

    推荐后续在 qa_pipeline.py 中使用：

    if question_analysis.get("问题意图") == "fact_query":
        retrieved_chunks = exact_retrieve_chunks(...)
        reranked_chunks = fact_score_chunks(
            user_question=user_question,
            chunks=retrieved_chunks,
            entities=question_analysis.get("核心实体", []),
            top_k=10,
        )
    else:
        b_results = b_score_chunks(...)
        c_results = c_rerank_chunks(...)
    """

    if not chunks:
        return []

    constraints = collect_fact_query_constraints(
        user_question=user_question,
        entities=entities,
    )

    scored_chunks: List[Dict[str, Any]] = []

    for chunk in chunks:
        scored = score_single_fact_chunk(
            user_question=user_question,
            chunk=chunk,
            entities=entities,
            constraints=constraints,
        )
        scored_chunks.append(scored)

    scored_chunks.sort(
        key=lambda x: (
            float(x.get("final_score", 0.0) or 0.0),
            float(x.get("fact_raw_score", 0.0) or 0.0),
            float(x.get("layer_id_score", 0.0) or 0.0),
            float(x.get("borehole_id_score", 0.0) or 0.0),
            float(x.get("semantic_score", 0.0) or 0.0),
        ),
        reverse=True,
    )

    for idx, chunk in enumerate(scored_chunks, start=1):
        chunk["fact_rank"] = idx
        chunk["rank"] = idx

    if top_k is not None:
        return scored_chunks[: int(top_k)]

    return scored_chunks


def fact_rerank_chunks(
    user_question: str,
    chunks: List[Dict[str, Any]],
    entities: Optional[List[Dict[str, Any]]] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    fact_score_chunks 的别名。

    命名上更贴近 pipeline 中的 rerank 阶段。
    """
    return fact_score_chunks(
        user_question=user_question,
        chunks=chunks,
        entities=entities,
        top_k=top_k,
    )