from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from configs.settings import NEO4J_DATABASE
from src.query.entity_extractor import extract_layer_ids, extract_borehole_ids
from src.utils.env import get_driver
from src.utils.text import safe_text, unique_keep_order, make_json_safe


# ============================================================
# 1. ID 标准化与基础工具
# ============================================================

def normalize_exact_id(value: str) -> str:
    """
    标准化 layer_id / borehole_id。

    示例：
    - chgc001_2 -> CHGC001_2
    - CHGC001 _ 2 -> CHGC001_2
    - chgc001 -> CHGC001
    """
    text = safe_text(value)
    text = re.sub(r"\s+", "", text)
    return text.upper()


def _normalize_id_list(values: List[str]) -> List[str]:
    normalized = [normalize_exact_id(v) for v in values if safe_text(v)]
    normalized = [v for v in normalized if v]
    return unique_keep_order(normalized)


def _safe_limit(limit: int, default: int = 10, max_limit: int = 50) -> int:
    """
    将 limit 转成安全整数。

    说明：
    某些 Neo4j 版本对 LIMIT $limit 支持不稳定，
    所以后续 Cypher 中直接使用安全整数插值。
    """
    try:
        value = int(limit)
    except Exception:
        value = default

    if value <= 0:
        value = default

    return min(value, max_limit)


def _run_cypher(cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    统一执行 Neo4j Cypher。

    作用：
    1. 所有 exact retrieval 查询都从这里执行。
    2. 如果报 CypherSyntaxError，会打印完整 Cypher、参数和 Neo4j 错误信息。
    3. 便于后续快速定位到底是哪条查询不兼容。
    """
    from neo4j.exceptions import Neo4jError

    driver = get_driver()

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            records = session.run(cypher, params).data()

        return records

    except Neo4jError as exc:
        print("\n" + "=" * 120)
        print("Neo4j Cypher 执行失败")
        print("=" * 120)
        print("error code:", getattr(exc, "code", None))
        print("error message:", getattr(exc, "message", None))
        print("\nCypher:")
        print(cypher)
        print("\nParams:")
        print(params)
        print("=" * 120 + "\n")
        raise


# ============================================================
# 2. 从 Query Translation 结果中抽取精确 ID
# ============================================================

def _extract_ids_from_entities(
    entities: Optional[List[Dict[str, Any]]],
) -> Tuple[List[str], List[str]]:
    """
    从 query translation 的 entities 中抽取 layer_id / borehole_id。

    兼容实体格式：

    {
        "实体类型": "层位",
        "实体值": "CHGC001_2",
        "来源": "Regex.layer_id",
        "字段名": "layer_id"
    }

    {
        "实体类型": "钻孔",
        "实体值": "CHGC001",
        "来源": "Regex.borehole_id",
        "字段名": "borehole_id"
    }
    """

    if not entities:
        return [], []

    layer_ids: List[str] = []
    borehole_ids: List[str] = []

    for entity in entities:
        entity_type = safe_text(entity.get("实体类型", ""))
        entity_value = safe_text(entity.get("实体值", ""))
        source = safe_text(entity.get("来源", ""))
        field_name = safe_text(entity.get("字段名", ""))

        if not entity_value:
            continue

        if field_name == "layer_id" or source == "Regex.layer_id":
            layer_ids.append(entity_value)
            continue

        if field_name == "borehole_id" or source == "Regex.borehole_id":
            borehole_ids.append(entity_value)
            continue

        # 兼容 Neo4j 词表命中的层位 ID
        if entity_type == "层位" and "_" in entity_value:
            layer_ids.append(entity_value)
            continue

        # 兼容 Neo4j 词表命中的钻孔 ID
        if entity_type == "钻孔":
            borehole_ids.append(entity_value)
            continue

    layer_ids = _normalize_id_list(layer_ids)
    borehole_ids = _normalize_id_list(borehole_ids)

    # 从 CHGC001_2 反推 CHGC001
    for layer_id in layer_ids:
        if "_" in layer_id:
            borehole_ids.append(layer_id.split("_", 1)[0])

    borehole_ids = _normalize_id_list(borehole_ids)

    return layer_ids, borehole_ids


def extract_exact_ids(
    user_question: str,
    entities: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[str]]:
    """
    综合从用户问题和 query translation entities 中提取精确 ID。

    返回示例：
    {
        "layer_ids": ["CHGC001_2"],
        "borehole_ids": ["CHGC001"]
    }
    """

    question = safe_text(user_question)

    regex_layer_ids = extract_layer_ids(question)
    regex_borehole_ids = extract_borehole_ids(question)

    entity_layer_ids, entity_borehole_ids = _extract_ids_from_entities(entities)

    layer_ids = _normalize_id_list(regex_layer_ids + entity_layer_ids)
    borehole_ids = _normalize_id_list(regex_borehole_ids + entity_borehole_ids)

    # 从 layer_id 反推出 borehole_id
    for layer_id in layer_ids:
        if "_" in layer_id:
            borehole_ids.append(layer_id.split("_", 1)[0])

    borehole_ids = _normalize_id_list(borehole_ids)

    return {
        "layer_ids": layer_ids,
        "borehole_ids": borehole_ids,
    }


# ============================================================
# 3. Neo4j 源节点查询
# ============================================================

def _fetch_layers_by_id(
    layer_ids: List[str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    精确查询 LithologyLayer 节点。

    只查结构化源节点，不做复杂图路径展开。
    """

    layer_ids = _normalize_id_list(layer_ids)

    if not layer_ids:
        return []

    safe_limit = _safe_limit(limit)

    cypher = f"""
    MATCH (l:LithologyLayer)
    WHERE toUpper(toString(l.layer_id)) IN $layer_ids
    RETURN
        'LithologyLayer' AS matched_label,
        toString(l.layer_id) AS matched_value,
        properties(l) AS source_props
    LIMIT {safe_limit}
    """

    return _run_cypher(
        cypher,
        {
            "layer_ids": layer_ids,
        },
    )


def _fetch_boreholes_by_id(
    borehole_ids: List[str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    精确查询 Borehole 节点。
    """

    borehole_ids = _normalize_id_list(borehole_ids)

    if not borehole_ids:
        return []

    safe_limit = _safe_limit(limit)

    cypher = f"""
    MATCH (b:Borehole)
    WHERE toUpper(toString(b.borehole_id)) IN $borehole_ids
    RETURN
        'Borehole' AS matched_label,
        toString(b.borehole_id) AS matched_value,
        properties(b) AS source_props
    LIMIT {safe_limit}
    """

    return _run_cypher(
        cypher,
        {
            "borehole_ids": borehole_ids,
        },
    )


def _fetch_chunks_by_exact_values(
    values: List[str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    按 source_id / chunk_id / chunk_text 精确查 EvidenceChunk。

    使用 UNWIND 写法，兼容性比复杂路径查询更好。
    """

    values = _normalize_id_list(values)

    if not values:
        return []

    safe_limit = _safe_limit(limit)

    cypher = f"""
    MATCH (c:EvidenceChunk)
    WITH
        c,
        toUpper(toString(coalesce(c.source_id, ''))) AS source_id_text,
        toUpper(toString(coalesce(c.chunk_id, ''))) AS chunk_id_text,
        toUpper(toString(coalesce(c.chunk_text, ''))) AS chunk_text_text
    UNWIND $values AS v
    WITH c, v, source_id_text, chunk_id_text, chunk_text_text
    WHERE source_id_text = v
       OR chunk_id_text = v
       OR chunk_text_text CONTAINS v
    RETURN DISTINCT
        toString(coalesce(c.chunk_id, '')) AS chunk_id,
        toString(coalesce(c.chunk_type, '')) AS chunk_type,
        toString(coalesce(c.chunk_text, '')) AS chunk_text,
        toString(coalesce(c.source_label, 'EvidenceChunk')) AS source_label,
        toString(coalesce(c.source_id, '')) AS source_id,
        properties(c) AS chunk_props
    LIMIT {safe_limit}
    """

    return _run_cypher(
        cypher,
        {
            "values": values,
        },
    )


# ============================================================
# 4. 结果转换
# ============================================================

def _props_to_text(label: str, props: Dict[str, Any]) -> str:
    """
    当没有找到 EvidenceChunk 时，把结构化节点属性转成可读文本。

    这样即使 LithologyLayer / Borehole 没有挂 EvidenceChunk，
    fact_query 也能返回一条可用证据。
    """

    if not props:
        return f"{label} 节点存在，但没有可展示的属性。"

    preferred_keys = [
        "layer_id",
        "borehole_id",
        "lithology_major_v2",
        "lithology_minor_v3",
        "lithology_major",
        "lithology_minor",
        "top_depth",
        "bottom_depth",
        "layer_top",
        "layer_bottom",
        "depth_top",
        "depth_bottom",
        "start_depth",
        "end_depth",
        "thickness",
        "elevation",
    ]

    lines: List[str] = [f"{label} 结构化记录："]

    used = set()

    for key in preferred_keys:
        if key in props and props.get(key) is not None:
            lines.append(f"- {key}: {props.get(key)}")
            used.add(key)

    for key in sorted(props.keys()):
        if key in used:
            continue

        value = props.get(key)

        if value is None:
            continue

        lines.append(f"- {key}: {value}")

    return "\n".join(lines)


def _source_record_to_synthetic_chunk(
    record: Dict[str, Any],
    exact_match_type: str,
    exact_score: float,
) -> Dict[str, Any]:
    """
    把 LithologyLayer / Borehole 源节点转换成 synthetic chunk。

    synthetic chunk 表示：
    - 没有找到真实 EvidenceChunk
    - 但结构化节点已经被精确命中
    - 可以把节点属性作为事实查询上下文
    """

    source_label = safe_text(record.get("matched_label", ""))
    matched_value = normalize_exact_id(record.get("matched_value", ""))
    source_props = record.get("source_props") or {}

    return {
        "chunk_id": f"exact::{exact_match_type}::{matched_value}",
        "chunk_type": f"exact_{exact_match_type}_record",
        "chunk_text": _props_to_text(source_label or exact_match_type, source_props),
        "source_label": source_label,
        "source_id": matched_value,
        "vector_score": 1.0,
        "retrieval_method": "exact",
        "exact_match_type": exact_match_type,
        "exact_match_value": matched_value,
        "exact_score": exact_score,
        "matched_fields": [exact_match_type],
        "source_props": make_json_safe(source_props),
        "is_synthetic": True,
        "graph_expandable": False,
    }


def _chunk_record_to_chunk(
    record: Dict[str, Any],
    exact_match_type: str,
    exact_match_value: str,
    exact_score: float,
) -> Dict[str, Any]:
    """
    把 EvidenceChunk 查询结果转换成统一 retrieved chunk 格式。
    """

    chunk_id = safe_text(record.get("chunk_id", ""))

    if not chunk_id:
        chunk_id = f"exact_chunk::{exact_match_type}::{exact_match_value}"

    return {
        "chunk_id": chunk_id,
        "chunk_type": safe_text(record.get("chunk_type", "")) or "exact_evidence_chunk",
        "chunk_text": safe_text(record.get("chunk_text", "")),
        "source_label": safe_text(record.get("source_label", "")) or "EvidenceChunk",
        "source_id": safe_text(record.get("source_id", "")) or exact_match_value,
        "vector_score": 1.0,
        "retrieval_method": "exact",
        "exact_match_type": exact_match_type,
        "exact_match_value": exact_match_value,
        "exact_score": exact_score,
        "matched_fields": [exact_match_type],
        "source_props": make_json_safe(record.get("chunk_props") or {}),
        "is_synthetic": False,
        "graph_expandable": True,
    }


def _infer_matched_value_from_chunk(
    chunk_record: Dict[str, Any],
    candidate_values: List[str],
) -> str:
    """
    判断某个 EvidenceChunk 命中了哪个 exact value。
    """

    candidates = _normalize_id_list(candidate_values)

    text = " ".join(
        [
            safe_text(chunk_record.get("chunk_id", "")),
            safe_text(chunk_record.get("source_id", "")),
            safe_text(chunk_record.get("chunk_text", "")),
        ]
    ).upper()

    for value in candidates:
        if value in text:
            return value

    return candidates[0] if candidates else ""


def _dedup_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对 exact retrieval 结果去重。

    优先保留：
    1. exact_score 更高的结果
    2. 非 synthetic 的 EvidenceChunk
    3. 先出现的结果
    """

    best_by_key: Dict[str, Dict[str, Any]] = {}

    for chunk in chunks:
        chunk_id = safe_text(chunk.get("chunk_id", ""))
        source_id = safe_text(chunk.get("source_id", ""))
        chunk_text = safe_text(chunk.get("chunk_text", ""))

        key = chunk_id or f"{source_id}::{chunk_text[:80]}"

        if key not in best_by_key:
            best_by_key[key] = chunk
            continue

        old = best_by_key[key]

        old_score = float(old.get("exact_score", 0.0) or 0.0)
        new_score = float(chunk.get("exact_score", 0.0) or 0.0)

        old_is_synthetic = bool(old.get("is_synthetic", False))
        new_is_synthetic = bool(chunk.get("is_synthetic", False))

        if new_score > old_score:
            best_by_key[key] = chunk
            continue

        if new_score == old_score and old_is_synthetic and not new_is_synthetic:
            best_by_key[key] = chunk

    results = list(best_by_key.values())

    results.sort(
        key=lambda x: (
            float(x.get("exact_score", 0.0) or 0.0),
            0 if x.get("is_synthetic") else 1,
        ),
        reverse=True,
    )

    return results


# ============================================================
# 5. 对外精确检索函数
# ============================================================

def exact_search_by_layer_id(
    layer_ids: List[str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    按 layer_id 精确查询。

    稳定版逻辑：
    1. 查 LithologyLayer.layer_id。
    2. 查 EvidenceChunk.source_id / chunk_id / chunk_text。
    3. 如果 EvidenceChunk 存在，优先返回 EvidenceChunk。
    4. 如果 EvidenceChunk 不存在，但 LithologyLayer 存在，则返回 synthetic chunk。
    """

    layer_ids = _normalize_id_list(layer_ids)

    if not layer_ids:
        return []

    source_records = _fetch_layers_by_id(layer_ids, limit=limit)
    chunk_records = _fetch_chunks_by_exact_values(layer_ids, limit=limit)

    chunks: List[Dict[str, Any]] = []

    for record in chunk_records:
        matched_value = _infer_matched_value_from_chunk(record, layer_ids)

        chunks.append(
            _chunk_record_to_chunk(
                record=record,
                exact_match_type="layer_id",
                exact_match_value=matched_value,
                exact_score=100.0,
            )
        )

    # 如果没有 EvidenceChunk，则使用 LithologyLayer 结构化节点兜底
    if not chunks:
        for record in source_records:
            chunks.append(
                _source_record_to_synthetic_chunk(
                    record=record,
                    exact_match_type="layer_id",
                    exact_score=100.0,
                )
            )

    return _dedup_chunks(chunks)[:limit]


def exact_search_by_borehole_id(
    borehole_ids: List[str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    按 borehole_id 精确查询。

    稳定版逻辑：
    1. 查 Borehole.borehole_id。
    2. 查 EvidenceChunk.source_id / chunk_id / chunk_text。
    3. 如果 EvidenceChunk 存在，优先返回 EvidenceChunk。
    4. 如果 EvidenceChunk 不存在，但 Borehole 存在，则返回 synthetic chunk。
    """

    borehole_ids = _normalize_id_list(borehole_ids)

    if not borehole_ids:
        return []

    source_records = _fetch_boreholes_by_id(borehole_ids, limit=limit)
    chunk_records = _fetch_chunks_by_exact_values(borehole_ids, limit=limit)

    chunks: List[Dict[str, Any]] = []

    for record in chunk_records:
        matched_value = _infer_matched_value_from_chunk(record, borehole_ids)

        chunks.append(
            _chunk_record_to_chunk(
                record=record,
                exact_match_type="borehole_id",
                exact_match_value=matched_value,
                exact_score=85.0,
            )
        )

    if not chunks:
        for record in source_records:
            chunks.append(
                _source_record_to_synthetic_chunk(
                    record=record,
                    exact_match_type="borehole_id",
                    exact_score=85.0,
                )
            )

    return _dedup_chunks(chunks)[:limit]


def exact_retrieve_chunks(
    user_question: str,
    entities: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 10,
    include_borehole_context_when_layer_found: bool = False,
) -> List[Dict[str, Any]]:
    """
    fact_query 专用精确检索入口。

    推荐后续在 qa_pipeline.py 中使用：

    if question_analysis.get("问题意图") == "fact_query":
        retrieved_chunks = exact_retrieve_chunks(
            user_question=user_question,
            entities=question_analysis.get("核心实体", []),
            top_k=10,
        )
    else:
        retrieved_chunks = vector_search_chunks(...)

    检索优先级：
    1. 如果有 layer_id，优先查 layer_id。
    2. 如果没有 layer_id，但有 borehole_id，则查 borehole_id。
    3. 如果 layer_id 查不到结果，再 fallback 到 borehole_id。
    """

    exact_ids = extract_exact_ids(
        user_question=user_question,
        entities=entities,
    )

    layer_ids = exact_ids["layer_ids"]
    borehole_ids = exact_ids["borehole_ids"]

    chunks: List[Dict[str, Any]] = []

    if layer_ids:
        layer_chunks = exact_search_by_layer_id(
            layer_ids=layer_ids,
            limit=top_k,
        )
        chunks.extend(layer_chunks)

        # 默认不加入 borehole 背景，避免 CHGC001_2 的精确查询被 CHGC001 的大量 chunk 稀释。
        # 如后续需要给 LLM 更多上下文，可把该参数设为 True。
        if include_borehole_context_when_layer_found and len(chunks) < top_k:
            borehole_chunks = exact_search_by_borehole_id(
                borehole_ids=borehole_ids,
                limit=top_k - len(chunks),
            )
            chunks.extend(borehole_chunks)

        # layer_id 明确存在但查不到结果时，才用 borehole_id 兜底
        if not chunks and borehole_ids:
            borehole_chunks = exact_search_by_borehole_id(
                borehole_ids=borehole_ids,
                limit=top_k,
            )
            chunks.extend(borehole_chunks)

    elif borehole_ids:
        borehole_chunks = exact_search_by_borehole_id(
            borehole_ids=borehole_ids,
            limit=top_k,
        )
        chunks.extend(borehole_chunks)

    chunks = _dedup_chunks(chunks)

    return chunks[:top_k]