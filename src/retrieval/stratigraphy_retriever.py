from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from configs.settings import NEO4J_DATABASE
from src.utils.env import get_driver
from src.utils.text import make_json_safe, safe_text, unique_keep_order


STRATIGRAPHY_ROUTE = "stratigraphy_exact_retrieval"

STRAT_SPECS = [
    {
        "field": "geologic_period",
        "entity_type": "地质纪",
        "label": "GeologicPeriod",
        "value_keys": ["name", "period_name"],
    },
    {
        "field": "geologic_epoch",
        "entity_type": "地质世",
        "label": "GeologicEpoch",
        "value_keys": ["name", "epoch_name"],
    },
    {
        "field": "strat_group",
        "entity_type": "地层组",
        "label": "StratGroup",
        "value_keys": ["name", "group_name"],
    },
    {
        "field": "strat_member",
        "entity_type": "地层段",
        "label": "StratMember",
        "value_keys": ["name", "member_name"],
    },
]

FACT_RETURN_KEYWORDS = [
    "给出",
    "列出",
    "显示",
    "查询",
    "有哪些",
    "所有",
    "对应",
    "钻孔",
    "孔号",
    "坐标",
    "位置",
    "分层",
    "层位",
    "地层",
]

REASONING_KEYWORDS = [
    "为什么",
    "原因",
    "影响",
    "控制因素",
    "推断",
    "预测",
    "估计",
    "解释",
    "透水",
    "渗透率",
]

_VOCAB_CACHE: Optional[Dict[str, List[str]]] = None


def _safe_limit(limit: int, default: int = 200, max_limit: int = 500) -> int:
    try:
        value = int(limit)
    except Exception:
        value = default

    if value <= 0:
        value = default

    return min(value, max_limit)


def _run_cypher(cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    driver = get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        return session.run(cypher, params or {}).data()


def _load_stratigraphy_vocab() -> Dict[str, List[str]]:
    global _VOCAB_CACHE

    if _VOCAB_CACHE is not None:
        return _VOCAB_CACHE

    vocab: Dict[str, List[str]] = {}

    for spec in STRAT_SPECS:
        value_expr = " + ".join(
            f"collect(DISTINCT toString(n.{key}))" for key in spec["value_keys"]
        )
        cypher = f"""
        MATCH (n:{spec['label']})
        RETURN {value_expr} AS values
        """
        records = _run_cypher(cypher)
        values: List[str] = []
        for record in records:
            for value in record.get("values") or []:
                text = safe_text(value)
                if text and text.lower() != "none":
                    values.append(text)
        vocab[spec["field"]] = sorted(set(values), key=lambda x: len(x), reverse=True)

    _VOCAB_CACHE = vocab
    return _VOCAB_CACHE


def extract_stratigraphy_terms(user_question: str) -> Dict[str, List[str]]:
    question = safe_text(user_question)
    vocab = _load_stratigraphy_vocab()

    matches: Dict[str, List[str]] = {}

    for spec in STRAT_SPECS:
        field = spec["field"]
        values = []
        for term in vocab.get(field, []):
            if term and term in question:
                values.append(term)
        matches[field] = unique_keep_order(values)

    return matches


def is_stratigraphy_query(
    user_question: str,
    question_analysis: Optional[Dict[str, Any]] = None,
) -> bool:
    question = safe_text(user_question)

    if not question:
        return False

    terms = extract_stratigraphy_terms(question)
    has_strat_term = any(values for values in terms.values())

    if not has_strat_term:
        return False

    # 解释、预测、渗透率类问题仍交给原 GraphRAG/水文规则链路。
    if any(keyword in question for keyword in REASONING_KEYWORDS):
        return False

    return any(keyword in question for keyword in FACT_RETURN_KEYWORDS)


def _format_layer_context(row: Dict[str, Any]) -> str:
    borehole_id = safe_text(row.get("borehole_id"))
    x_value = safe_text(row.get("x"))
    y_value = safe_text(row.get("y"))
    ground_elevation = safe_text(row.get("ground_elevation"))
    layer_id = safe_text(row.get("layer_id"))
    top_depth = safe_text(row.get("top_depth"))
    bottom_depth = safe_text(row.get("bottom_depth"))
    thickness = safe_text(row.get("thickness"))
    lithology_major = safe_text(row.get("lithology_major_v2"))
    lithology_minor = safe_text(row.get("lithology_minor_v3")) or "未知"
    period = safe_text(row.get("geologic_period"))
    epoch = safe_text(row.get("geologic_epoch"))
    group = safe_text(row.get("strat_group"))
    member = safe_text(row.get("strat_member"))

    return (
        f"钻孔 {borehole_id} 坐标 X={x_value}, Y={y_value}, 地面高程={ground_elevation}；"
        f"分层 {layer_id} 位于 {top_depth}m 至 {bottom_depth}m，厚度 {thickness}m；"
        f"地层信息：纪={period}，世={epoch}，组={group}，段={member}；"
        f"岩性大类={lithology_major}，岩性小类={lithology_minor}。"
    )


def _row_to_chunk(row: Dict[str, Any], matched_fields: List[str]) -> Dict[str, Any]:
    layer_id = safe_text(row.get("layer_id"))
    chunk_id = safe_text(row.get("chunk_id")) or f"stratigraphy::{layer_id}"
    chunk_text = _format_layer_context(row)

    source_props = {
        "borehole_id": row.get("borehole_id"),
        "x": row.get("x"),
        "y": row.get("y"),
        "ground_elevation": row.get("ground_elevation"),
        "layer_id": row.get("layer_id"),
        "top_depth": row.get("top_depth"),
        "bottom_depth": row.get("bottom_depth"),
        "thickness": row.get("thickness"),
        "lithology_major_v2": row.get("lithology_major_v2"),
        "lithology_minor_v3": row.get("lithology_minor_v3"),
        "geologic_period": row.get("geologic_period"),
        "geologic_epoch": row.get("geologic_epoch"),
        "strat_group": row.get("strat_group"),
        "strat_member": row.get("strat_member"),
    }

    return {
        "chunk_id": chunk_id,
        "chunk_type": "stratigraphy_layer_profile_chunk",
        "chunk_text": chunk_text,
        "source_label": "LithologyLayer",
        "source_id": safe_text(row.get("layer_node_id")) or f"Layer:{layer_id}",
        "vector_score": 1.0,
        "retrieval_method": "stratigraphy_exact",
        "exact_match_type": "stratigraphy",
        "exact_match_value": ", ".join(
            safe_text(row.get(field)) for field in matched_fields if safe_text(row.get(field))
        ),
        "exact_score": 100.0,
        "matched_fields": matched_fields,
        "source_props": make_json_safe(source_props),
        "is_synthetic": False,
        "graph_expandable": bool(row.get("chunk_id")),
        "graph_context_text": chunk_text,
    }


def retrieve_stratigraphy_exact(
    user_question: str,
    top_k: int = 200,
) -> List[Dict[str, Any]]:
    terms = extract_stratigraphy_terms(user_question)
    active_fields = [field for field, values in terms.items() if values]

    if not active_fields:
        return []

    safe_limit = _safe_limit(top_k)

    cypher = f"""
    MATCH (b:Borehole)-[:HAS_LAYER]->(l:LithologyLayer)
    WHERE
        (size($geologic_period) = 0 OR toString(l.geologic_period) IN $geologic_period)
        AND (size($geologic_epoch) = 0 OR toString(l.geologic_epoch) IN $geologic_epoch)
        AND (size($strat_group) = 0 OR toString(l.strat_group) IN $strat_group)
        AND (size($strat_member) = 0 OR toString(l.strat_member) IN $strat_member)
    OPTIONAL MATCH (l)-[:HAS_EVIDENCE_CHUNK]->(chunk:EvidenceChunk)
    WITH b, l, head(collect(chunk)) AS chunk
    RETURN
        toString(b.borehole_id) AS borehole_id,
        b.x AS x,
        b.y AS y,
        b.ground_elevation AS ground_elevation,
        toString(l.id) AS layer_node_id,
        toString(l.layer_id) AS layer_id,
        l.top_depth AS top_depth,
        l.bottom_depth AS bottom_depth,
        l.thickness AS thickness,
        l.lithology_major_v2 AS lithology_major_v2,
        l.lithology_minor_v3 AS lithology_minor_v3,
        l.geologic_period AS geologic_period,
        l.geologic_epoch AS geologic_epoch,
        l.strat_group AS strat_group,
        l.strat_member AS strat_member,
        toString(coalesce(chunk.chunk_id, '')) AS chunk_id
    ORDER BY borehole_id, toFloat(l.top_depth), layer_id
    LIMIT {safe_limit}
    """

    rows = _run_cypher(cypher, terms)
    return [_row_to_chunk(row, active_fields) for row in rows]


def _format_depth(value: Any) -> str:
    text = safe_text(value)
    return text if text else "-"


def build_stratigraphy_direct_answer(
    user_question: str,
    chunks: List[Dict[str, Any]],
) -> str:
    terms = extract_stratigraphy_terms(user_question)
    matched_terms = [
        value
        for field in ["geologic_period", "geologic_epoch", "strat_group", "strat_member"]
        for value in terms.get(field, [])
    ]
    target = "、".join(matched_terms) if matched_terms else "指定地层"

    if not chunks:
        return f"当前知识图谱中未检索到 {target} 对应的钻孔分层记录。"

    grouped: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        props = chunk.get("source_props") or {}
        borehole_id = safe_text(props.get("borehole_id"))
        if borehole_id not in grouped:
            grouped[borehole_id] = {
                "x": props.get("x"),
                "y": props.get("y"),
                "ground_elevation": props.get("ground_elevation"),
                "layers": [],
            }
        grouped[borehole_id]["layers"].append(props)

    lines = [
        f"检索到 {target} 相关分层 {len(chunks)} 条，覆盖钻孔 {len(grouped)} 个。",
        "",
        "| 钻孔 | X | Y | 地面高程 | 对应分层 |",
        "|---|---:|---:|---:|---|",
    ]

    for borehole_id in sorted(grouped.keys()):
        item = grouped[borehole_id]
        layer_parts = []
        for layer in sorted(
            item["layers"],
            key=lambda p: (
                float(p.get("top_depth") or 0),
                safe_text(p.get("layer_id")),
            ),
        ):
            lithology_minor = safe_text(layer.get("lithology_minor_v3")) or "未知"
            layer_parts.append(
                f"{safe_text(layer.get('layer_id'))}"
                f"({ _format_depth(layer.get('top_depth'))}-{ _format_depth(layer.get('bottom_depth'))}m,"
                f"{safe_text(layer.get('strat_group'))}/{safe_text(layer.get('strat_member'))},"
                f"{safe_text(layer.get('lithology_major_v2'))}-{lithology_minor})"
            )

        lines.append(
            "| "
            + " | ".join(
                [
                    borehole_id,
                    safe_text(item.get("x")),
                    safe_text(item.get("y")),
                    safe_text(item.get("ground_elevation")),
                    "<br>".join(layer_parts),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def stratigraphy_rerank(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for rank, chunk in enumerate(chunks, start=1):
        item = dict(chunk)
        item.update(
            {
                "rank": rank,
                "final_score": 100.0,
                "c_score": 100.0,
                "score": 100.0,
                "use_for_context": True,
                "rerank_route": "stratigraphy_rule",
                "scoring_method": "stratigraphy_exact_match",
                "rerank_reason": "精确命中纪/世/组/段地层字段，并返回钻孔坐标与分层属性。",
            }
        )
        results.append(item)

    return results
