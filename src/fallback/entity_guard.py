from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.query.entity_extractor import (
    BOREHOLE_ID_RE,
    LAYER_ID_RE,
    extract_layer_ids,
    normalize_structured_id,
)
from src.retrieval.exact_retriever import extract_exact_ids
from src.utils.text import make_json_safe, safe_text, unique_keep_order


DEPTH_VALUE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|米)", re.IGNORECASE)

PERMEABILITY_QUERY_TERMS = [
    "渗透率",
    "渗透系数",
    "k值",
    "K值",
    "k=",
    "K=",
    "k_log10",
    "log10",
    "log10(k)",
    "水力传导系数",
]


def _fallback_answer(reason: str) -> str:
    return (
        "根据当前知识库检索结果，找不到与问题中指定实体完全匹配的可靠记录，"
        "因此不能编造具体岩性、渗透率、k 值或透水等级。\n\n"
        f"原因：{reason}"
    )


def _run_cypher(
    driver: Any,
    database: str,
    cypher: str,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    with driver.session(database=database) as session:
        return session.run(cypher, params).data()


def _entity_values(
    question_analysis: Dict[str, Any],
    entity_type_names: Iterable[str],
) -> List[str]:
    names = set(entity_type_names)
    values: List[str] = []

    for key in ["核心实体", "entities", "core_entities"]:
        entities = question_analysis.get(key)
        if not isinstance(entities, list):
            continue

        for entity in entities:
            if not isinstance(entity, dict):
                continue

            entity_type = safe_text(entity.get("实体类型", ""))
            entity_value = safe_text(entity.get("实体值", ""))

            if entity_type in names and entity_value:
                values.append(entity_value)

    return unique_keep_order(values)


def _extract_explicit_borehole_ids(question: str) -> List[str]:
    text = safe_text(question)
    text_without_layers = LAYER_ID_RE.sub(" ", text)
    values = [
        normalize_structured_id(match.group(1))
        for match in BOREHOLE_ID_RE.finditer(text_without_layers)
    ]
    return unique_keep_order([v for v in values if v])


def extract_lithology_values(
    question: str,
    question_analysis: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Extract explicitly requested lithology major/minor values.

    This is intentionally strict enough for closed-set guards and
    LithologyType exact retrieval; vague lithology mentions should keep using
    the normal GraphRAG route.
    """

    text = safe_text(question)

    entity_major_values = _entity_values(question_analysis, ["岩性大类"])
    entity_minor_values = _entity_values(question_analysis, ["岩性小类"])
    regex_major_values: List[str] = []
    regex_minor_values: List[str] = []

    major_patterns = [
        r"岩性大类\s*(?:为|是|=|：|:)\s*([^，。、,；;？?\s的]+)",
        r"大类\s*(?:为|是|=|：|:)\s*([^，。、,；;？?\s的]+)",
    ]
    minor_patterns = [
        r"岩性小类\s*(?:为|是|=|：|:)\s*([^，。、,；;？?\s的]+)",
        r"小类\s*(?:为|是|=|：|:)\s*([^，。、,；;？?\s的]+)",
    ]

    for pattern in major_patterns:
        regex_major_values.extend(re.findall(pattern, text))

    for pattern in minor_patterns:
        regex_minor_values.extend(re.findall(pattern, text))

    major_values = regex_major_values or entity_major_values
    minor_values = regex_minor_values or entity_minor_values

    return _clean_lithology_values(major_values), _clean_lithology_values(minor_values)


def _clean_lithology_values(values: List[str]) -> List[str]:
    cleaned: List[str] = []

    for value in values:
        text = safe_text(value).strip(" ，。,；;：:")

        for suffix in ["时", "中", "里", "下"]:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]

        if text:
            cleaned.append(text)

    return unique_keep_order(cleaned)


def _extract_depth_values(question: str) -> List[float]:
    values: List[float] = []

    for match in DEPTH_VALUE_RE.finditer(safe_text(question)):
        try:
            values.append(float(match.group(1)))
        except Exception:
            continue

    return values


def _has_permeability_record_request(question: str) -> bool:
    text = safe_text(question)
    return any(term in text for term in PERMEABILITY_QUERY_TERMS)


def _fetch_existing_boreholes(
    driver: Any,
    database: str,
    borehole_ids: List[str],
) -> List[str]:
    if not borehole_ids:
        return []

    rows = _run_cypher(
        driver,
        database,
        """
        MATCH (b:Borehole)
        WHERE toUpper(toString(b.borehole_id)) IN $borehole_ids
        RETURN collect(DISTINCT toUpper(toString(b.borehole_id))) AS values
        """,
        {"borehole_ids": borehole_ids},
    )

    return rows[0].get("values", []) if rows else []


def _fetch_existing_layers(
    driver: Any,
    database: str,
    layer_ids: List[str],
) -> List[str]:
    if not layer_ids:
        return []

    rows = _run_cypher(
        driver,
        database,
        """
        MATCH (l:LithologyLayer)
        WHERE toUpper(toString(l.layer_id)) IN $layer_ids
        RETURN collect(DISTINCT toUpper(toString(l.layer_id))) AS values
        """,
        {"layer_ids": layer_ids},
    )

    return rows[0].get("values", []) if rows else []


def _fetch_lithology_type_values(
    driver: Any,
    database: str,
    major_values: List[str],
    minor_values: List[str],
) -> Dict[str, List[str]]:
    rows = _run_cypher(
        driver,
        database,
        """
        MATCH (lt:LithologyType)
        WHERE
            ($major_values = [] OR toString(lt.lithology_major_v2) IN $major_values)
            OR
            ($minor_values = [] OR toString(lt.lithology_minor_v3) IN $minor_values)
        RETURN
            collect(DISTINCT toString(lt.lithology_major_v2)) AS major_values,
            collect(DISTINCT toString(lt.lithology_minor_v3)) AS minor_values
        """,
        {
            "major_values": major_values,
            "minor_values": minor_values,
        },
    )

    if not rows:
        return {"major_values": [], "minor_values": []}

    return {
        "major_values": rows[0].get("major_values", []) or [],
        "minor_values": rows[0].get("minor_values", []) or [],
    }


def _fetch_layers_for_depth_check(
    driver: Any,
    database: str,
    borehole_ids: List[str],
) -> List[Dict[str, Any]]:
    if not borehole_ids:
        return []

    return _run_cypher(
        driver,
        database,
        """
        MATCH (l:LithologyLayer)
        WHERE toUpper(toString(l.borehole_id)) IN $borehole_ids
        RETURN properties(l) AS props
        """,
        {"borehole_ids": borehole_ids},
    )


def _fetch_permeability_observation_count(
    driver: Any,
    database: str,
    layer_ids: List[str],
    borehole_ids: List[str],
) -> int:
    rows = _run_cypher(
        driver,
        database,
        """
        MATCH (p:PermeabilityObservation)
        WITH
            p,
            toUpper(toString(coalesce(p.layer_id, ''))) AS layer_id,
            toUpper(toString(coalesce(p.borehole_id, ''))) AS borehole_id,
            toUpper(toString(coalesce(p.id, ''))) AS obs_id,
            toUpper(toString(coalesce(p.source_id, ''))) AS source_id
        WHERE
            ($layer_ids <> [] AND (
                layer_id IN $layer_ids
                OR any(x IN $layer_ids WHERE obs_id CONTAINS x OR source_id CONTAINS x)
            ))
            OR
            ($borehole_ids <> [] AND (
                borehole_id IN $borehole_ids
                OR any(x IN $borehole_ids WHERE obs_id CONTAINS x OR source_id CONTAINS x)
            ))
        RETURN count(p) AS count
        """,
        {
            "layer_ids": layer_ids,
            "borehole_ids": borehole_ids,
        },
    )

    if not rows:
        return 0

    try:
        return int(rows[0].get("count", 0) or 0)
    except Exception:
        return 0


def _float_prop(props: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        value = props.get(key)
        if value is None:
            continue

        try:
            return float(value)
        except Exception:
            continue

    return None


def _depth_is_covered(depth: float, layer_rows: List[Dict[str, Any]]) -> bool:
    for row in layer_rows:
        props = row.get("props") or {}
        if not isinstance(props, dict):
            continue

        top = _float_prop(
            props,
            ["top_depth", "layer_top", "depth_top", "start_depth", "top_m"],
        )
        bottom = _float_prop(
            props,
            ["bottom_depth", "layer_bottom", "depth_bottom", "end_depth", "bottom_m"],
        )

        if top is None or bottom is None:
            continue

        low = min(top, bottom)
        high = max(top, bottom)

        if low <= depth <= high:
            return True

    return False


def _candidate_text(item: Dict[str, Any]) -> str:
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
            values.append(str(make_json_safe(value)))
        else:
            values.append(safe_text(value))

    for field in ["source_props", "chunk_props"]:
        value = item.get(field)
        if value:
            values.append(str(make_json_safe(value)))

    return " ".join([v for v in values if v])


def _merge_top_items(
    retrieved_chunks: List[Dict[str, Any]],
    effective_rerank_results: List[Dict[str, Any]],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    chunk_map = {
        safe_text(item.get("chunk_id", "")): item
        for item in retrieved_chunks
        if safe_text(item.get("chunk_id", ""))
    }

    source_map = {
        safe_text(item.get("source_id", "")): item
        for item in retrieved_chunks
        if safe_text(item.get("source_id", ""))
    }

    merged: List[Dict[str, Any]] = []

    for item in effective_rerank_results[:limit]:
        base = dict(chunk_map.get(safe_text(item.get("chunk_id", "")), {}))

        if not base:
            base = dict(source_map.get(safe_text(item.get("source_id", "")), {}))

        base.update(item)
        merged.append(base)

    if not merged:
        merged = retrieved_chunks[:limit]

    return merged


def _text_has_layer_id(text: str, layer_id: str) -> bool:
    escaped = re.escape(layer_id.upper())
    return bool(re.search(rf"(?<![A-Z0-9_]){escaped}(?![A-Z0-9_])", text.upper()))


def _text_has_borehole_id(text: str, borehole_id: str) -> bool:
    escaped = re.escape(borehole_id.upper())
    return bool(
        re.search(
            rf"(?<![A-Z0-9_]){escaped}(?:_\d+)?(?![A-Z0-9_])",
            text.upper(),
        )
    )


def _item_has_lithology_value(item: Dict[str, Any], field: str, value: str) -> bool:
    value = safe_text(value)

    if not value:
        return False

    source_props = item.get("source_props") or item.get("chunk_props") or {}
    if isinstance(source_props, dict) and safe_text(source_props.get(field, "")) == value:
        return True

    source_id = safe_text(item.get("source_id", ""))
    if source_id.startswith("LithologyType:"):
        parts = source_id.replace("LithologyType:", "", 1).split("|")

        if field == "lithology_major_v2" and len(parts) >= 1 and parts[0] == value:
            return True

        if field == "lithology_minor_v3" and len(parts) >= 2 and parts[1] == value:
            return True

    text = _candidate_text(item)

    return (
        f"{field}={value}" in text
        or f"'{field}': '{value}'" in text
        or f'"{field}": "{value}"' in text
    )


def _top10_contains_closed_entities(
    retrieved_chunks: List[Dict[str, Any]],
    effective_rerank_results: List[Dict[str, Any]],
    layer_ids: List[str],
    explicit_borehole_ids: List[str],
    lithology_major_values: List[str],
    lithology_minor_values: List[str],
) -> Tuple[bool, str]:
    top_items = _merge_top_items(retrieved_chunks, effective_rerank_results, limit=10)

    for layer_id in layer_ids:
        if not any(_text_has_layer_id(_candidate_text(item), layer_id) for item in top_items):
            return False, f"Top10 证据不包含指定分层号 {layer_id}。"

    for borehole_id in explicit_borehole_ids:
        if not any(_text_has_borehole_id(_candidate_text(item), borehole_id) for item in top_items):
            return False, f"Top10 证据不包含指定钻孔号 {borehole_id}。"

    for value in lithology_major_values:
        if not any(_item_has_lithology_value(item, "lithology_major_v2", value) for item in top_items):
            return False, f"Top10 证据不包含指定岩性大类 {value}。"

    for value in lithology_minor_values:
        if not any(_item_has_lithology_value(item, "lithology_minor_v3", value) for item in top_items):
            return False, f"Top10 证据不包含指定岩性小类 {value}。"

    return True, ""


def _blocked_result(reason: str, checks: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fallback_triggered": True,
        "fallback_reason": reason,
        "fallback_answer": _fallback_answer(reason),
        "entity_guard_checks": make_json_safe(checks),
    }


def apply_entity_existence_guard(
    driver: Any,
    database: str,
    user_question: str,
    question_analysis: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    effective_rerank_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    question = safe_text(user_question)
    entities = question_analysis.get("核心实体", [])
    exact_ids = extract_exact_ids(user_question=question, entities=entities)

    layer_ids = exact_ids.get("layer_ids", [])
    explicit_borehole_ids = _extract_explicit_borehole_ids(question)
    all_borehole_ids = unique_keep_order(exact_ids.get("borehole_ids", []) + explicit_borehole_ids)
    depth_values = _extract_depth_values(question)
    lithology_major_values, lithology_minor_values = extract_lithology_values(
        question,
        question_analysis,
    )

    checks: Dict[str, Any] = {
        "layer_ids": layer_ids,
        "explicit_borehole_ids": explicit_borehole_ids,
        "all_borehole_ids": all_borehole_ids,
        "depth_values": depth_values,
        "lithology_major_values": lithology_major_values,
        "lithology_minor_values": lithology_minor_values,
        "permeability_record_requested": _has_permeability_record_request(question),
    }

    if explicit_borehole_ids:
        existing = _fetch_existing_boreholes(driver, database, explicit_borehole_ids)
        missing = [x for x in explicit_borehole_ids if x not in existing]
        checks["existing_borehole_ids"] = existing

        if missing:
            return _blocked_result(f"资料中无记录：钻孔号 {', '.join(missing)} 不存在。", checks)

    if layer_ids:
        existing = _fetch_existing_layers(driver, database, layer_ids)
        missing = [x for x in layer_ids if x not in existing]
        checks["existing_layer_ids"] = existing

        if missing:
            return _blocked_result(f"资料中无记录：分层号 {', '.join(missing)} 不存在。", checks)

    if depth_values and all_borehole_ids:
        layer_rows = _fetch_layers_for_depth_check(driver, database, all_borehole_ids)
        uncovered = [
            depth
            for depth in depth_values
            if not _depth_is_covered(depth, layer_rows)
        ]
        checks["depth_layer_count"] = len(layer_rows)
        checks["uncovered_depth_values"] = uncovered

        if uncovered:
            depth_text = ", ".join(f"{value:g}m" for value in uncovered)
            borehole_text = ", ".join(all_borehole_ids)
            return _blocked_result(
                f"深度 {depth_text} 超出钻孔 {borehole_text} 的已记录层位范围。",
                checks,
            )

    if lithology_major_values or lithology_minor_values:
        existing = _fetch_lithology_type_values(
            driver,
            database,
            lithology_major_values,
            lithology_minor_values,
        )
        existing_major = existing.get("major_values", [])
        existing_minor = existing.get("minor_values", [])
        missing_major = [x for x in lithology_major_values if x not in existing_major]
        missing_minor = [x for x in lithology_minor_values if x not in existing_minor]
        checks["existing_lithology_major_values"] = existing_major
        checks["existing_lithology_minor_values"] = existing_minor

        if missing_major:
            return _blocked_result(
                f"资料中无记录：岩性大类 {', '.join(missing_major)} 不存在于 LithologyType。",
                checks,
            )

        if missing_minor:
            return _blocked_result(
                f"资料中无记录：岩性小类 {', '.join(missing_minor)} 不存在于 LithologyType。",
                checks,
            )

    if _has_permeability_record_request(question) and (layer_ids or all_borehole_ids):
        count = _fetch_permeability_observation_count(
            driver,
            database,
            layer_ids,
            all_borehole_ids,
        )
        checks["permeability_observation_count"] = count

        if count <= 0:
            target = ", ".join(layer_ids or all_borehole_ids)
            return _blocked_result(
                f"资料中无记录：{target} 没有对应的 PermeabilityObservation 渗透率记录。",
                checks,
            )

    if retrieved_chunks or effective_rerank_results:
        contains, miss_reason = _top10_contains_closed_entities(
            retrieved_chunks=retrieved_chunks,
            effective_rerank_results=effective_rerank_results,
            layer_ids=layer_ids,
            explicit_borehole_ids=explicit_borehole_ids,
            lithology_major_values=lithology_major_values,
            lithology_minor_values=lithology_minor_values,
        )
        checks["top10_contains_closed_entities"] = contains

        if not contains:
            return _blocked_result(
                f"{miss_reason} 禁止基于相似但不匹配的证据生成事实答案。",
                checks,
            )

    return {
        "fallback_triggered": False,
        "fallback_reason": "",
        "fallback_answer": "",
        "entity_guard_checks": make_json_safe(checks),
    }
