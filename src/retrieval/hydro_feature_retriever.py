from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from configs.settings import NEO4J_DATABASE
from src.utils.env import get_driver
from src.utils.text import make_json_safe, safe_text


FEATURE_EFFECT_RE = re.compile(
    r"水文地质特征[“\"](?P<feature>[^”\"]+)[”\"].*?"
    r"产生[“\"](?P<effect>[^”\"]+)[”\"]",
)


MECHANISM_BY_FEATURE_TYPE = {
    "GrainSizeFeature": (
        "机制解释：粒径越粗，颗粒骨架形成的孔隙半径通常越大，"
        "渗流通道更容易贯通，水流阻力降低，因此渗透率倾向升高；"
        "细粒组分增多时则可能堵塞孔隙并削弱通道连通。"
    ),
    "ClayContentFeature": (
        "机制解释：黏土或泥质成分会填充颗粒间孔隙，细粒吸附水和膨胀性会"
        "压缩有效渗流通道，降低孔隙连通性，因此渗透率倾向降低。"
    ),
    "SortingFeature": (
        "机制解释：分选性影响孔隙结构均一性。分选较好时颗粒排列更稳定，"
        "有效孔隙和渗流通道更连续；分选差或细粒充填时通道更易被堵塞。"
    ),
    "DensityFeature": (
        "机制解释：密实度升高通常意味着压密增强、孔隙率降低、孔喉变窄，"
        "有效过水断面和通道连通性下降，因此渗透率倾向降低。"
    ),
    "CementationFeature": (
        "机制解释：胶结物会占据或封堵颗粒间孔隙。泥质胶结通常降低有效孔隙"
        "和孔喉连通；胶结差则保留更多开放孔隙，可能提高局部渗透能力。"
    ),
    "InterlayerFeature": (
        "机制解释：夹砂层或透镜体会造成非均质结构，砂层内部颗粒较粗、孔隙"
        "较开放，可能形成局部高渗通道，但空间连续性仍受夹层展布控制。"
    ),
}

DEFAULT_MECHANISM = (
    "机制解释：该水文地质特征通过改变颗粒组成、细粒含量、孔隙尺度、"
    "孔喉结构、压密或胶结状态，影响有效渗流通道和介质连通性，从而改变渗透率。"
)

MECHANISM_TERMS = [
    "机制解释",
    "粒径",
    "颗粒",
    "细粒",
    "黏土",
    "泥质",
    "胶结",
    "压密",
    "密实",
    "分选",
    "夹层",
    "非均质",
    "孔隙",
    "孔喉",
    "通道",
    "渗流",
    "连通",
    "水流阻力",
    "有效过水断面",
]


def extract_hydro_feature_query(user_question: str) -> Dict[str, str]:
    question = safe_text(user_question)
    match = FEATURE_EFFECT_RE.search(question)

    if match:
        return {
            "feature_value": safe_text(match.group("feature")),
            "permeability_effect": safe_text(match.group("effect")),
        }

    quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", question)

    return {
        "feature_value": safe_text(quoted[0]) if quoted else "",
        "permeability_effect": safe_text(quoted[1]) if len(quoted) > 1 else "",
    }


def is_hydro_feature_causal_query(
    user_question: str,
    question_analysis: Optional[Dict[str, Any]] = None,
) -> bool:
    question = safe_text(user_question)

    if not question:
        return False

    extracted = extract_hydro_feature_query(question)

    if not extracted["feature_value"] or not extracted["permeability_effect"]:
        return False

    intent = ""
    if question_analysis:
        intent = safe_text(question_analysis.get("问题意图", ""))

    return (
        "为什么" in question
        and "水文地质特征" in question
        and ("渗透率" in question or "透水" in question)
        and (not intent or intent == "分析渗透率变化原因")
    )


def _run_cypher(
    driver: Any,
    database: str,
    cypher: str,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    with driver.session(database=database) as session:
        return session.run(cypher, params).data()


def _light_props(props: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in (props or {}).items()
        if key not in {"embedding", "embedding_vector"}
    }


def build_mechanism_text(source_props: Dict[str, Any]) -> str:
    explicit_parts: List[str] = []
    for key in ["mechanism", "explanation", "pore_connectivity"]:
        value = safe_text(source_props.get(key, ""))
        if value and value not in explicit_parts:
            explicit_parts.append(value)

    if explicit_parts:
        return "机制解释：" + " ".join(explicit_parts)

    feature_type = safe_text(source_props.get("feature_type", ""))
    return MECHANISM_BY_FEATURE_TYPE.get(feature_type, DEFAULT_MECHANISM)


def fetch_hydro_feature_candidates(
    feature_value: str,
    permeability_effect: str,
    driver: Any = None,
    database: str = NEO4J_DATABASE,
    include_near: bool = True,
) -> Dict[str, Any]:
    feature = safe_text(feature_value)
    effect = safe_text(permeability_effect)

    if not feature or not effect:
        return {
            "exact": [],
            "near": [],
            "feature_value": feature,
            "permeability_effect": effect,
        }

    driver = driver or get_driver()

    exact_rows = _run_cypher(
        driver,
        database,
        """
        MATCH (hf:HydroFeature)
        WHERE hf.feature_value = $feature_value
          AND hf.permeability_effect = $permeability_effect
        OPTIONAL MATCH (hf)-[:HAS_EVIDENCE_CHUNK]->(chunk:EvidenceChunk)
        RETURN
            properties(hf) AS source_props,
            properties(chunk) AS chunk_props
        ORDER BY
            CASE hf.effect_weight WHEN '高' THEN 3 WHEN '中' THEN 2 ELSE 1 END DESC,
            CASE hf.confidence WHEN '高' THEN 3 WHEN '中' THEN 2 ELSE 1 END DESC
        LIMIT 10
        """,
        {
            "feature_value": feature,
            "permeability_effect": effect,
        },
    )

    near_rows: List[Dict[str, Any]] = []

    if include_near and not exact_rows:
        near_rows = _run_cypher(
            driver,
            database,
            """
            MATCH (hf:HydroFeature)
            WHERE hf.feature_value = $feature_value
               OR hf.permeability_effect = $permeability_effect
               OR hf.feature_value CONTAINS $feature_value
               OR $feature_value CONTAINS hf.feature_value
               OR hf.permeability_effect CONTAINS $permeability_effect
               OR $permeability_effect CONTAINS hf.permeability_effect
            OPTIONAL MATCH (hf)-[:HAS_EVIDENCE_CHUNK]->(chunk:EvidenceChunk)
            RETURN
                properties(hf) AS source_props,
                properties(chunk) AS chunk_props
            LIMIT 10
            """,
            {
                "feature_value": feature,
                "permeability_effect": effect,
            },
        )

    return {
        "exact": exact_rows,
        "near": near_rows,
        "feature_value": feature,
        "permeability_effect": effect,
    }


def _source_props_to_text(props: Dict[str, Any]) -> str:
    feature_type = safe_text(props.get("feature_type", ""))
    feature = safe_text(props.get("feature_value", ""))
    effect = safe_text(props.get("permeability_effect", ""))
    weight = safe_text(props.get("effect_weight", ""))
    confidence = safe_text(props.get("confidence", ""))
    mechanism = safe_text(props.get("mechanism", ""))

    return (
        "水文地质特征规则："
        f"特征类型为 {feature_type}，特征取值为“{feature}”，"
        f"对渗透率的影响为“{effect}”，影响权重为{weight}，"
        f"置信度为{confidence}。{mechanism}"
    )


def _record_to_chunk(record: Dict[str, Any]) -> Dict[str, Any]:
    source_props = _light_props(record.get("source_props") or {})
    chunk_props = _light_props(record.get("chunk_props") or {})
    mechanism = build_mechanism_text(source_props)
    source_props["mechanism"] = mechanism
    source_props["mechanism_keywords"] = [
        term
        for term in MECHANISM_TERMS
        if term in mechanism
    ]

    source_id = safe_text(source_props.get("id", ""))
    if not source_id:
        source_id = f"HydroFeature:{safe_text(source_props.get('feature_value', ''))}"

    chunk_id = safe_text(chunk_props.get("chunk_id", "")) or safe_text(chunk_props.get("id", ""))
    if not chunk_id:
        chunk_id = f"exact::hydro_feature::{source_id}"

    chunk_text = safe_text(chunk_props.get("chunk_text", ""))
    if chunk_text:
        chunk_text = f"{chunk_text} {mechanism}"
    else:
        chunk_text = _source_props_to_text(source_props)

    return {
        "chunk_id": chunk_id,
        "chunk_type": safe_text(chunk_props.get("chunk_type", "")) or "hydro_feature_rule_chunk",
        "chunk_text": chunk_text,
        "source_label": "HydroFeature",
        "source_id": source_id,
        "vector_score": 1.0,
        "retrieval_method": "hydro_feature_exact",
        "exact_match_type": "feature_value_permeability_effect",
        "exact_match_value": f"{source_props.get('feature_value')}|{source_props.get('permeability_effect')}",
        "exact_score": 100.0,
        "matched_fields": ["feature_value", "permeability_effect", "mechanism"],
        "source_props": make_json_safe(source_props),
        "chunk_props": make_json_safe(chunk_props),
        "is_synthetic": not bool(chunk_props),
        "graph_expandable": bool(chunk_props),
    }


def retrieve_hydro_feature_exact(
    feature_value: str,
    permeability_effect: str,
    driver: Any = None,
    database: str = NEO4J_DATABASE,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = fetch_hydro_feature_candidates(
        feature_value=feature_value,
        permeability_effect=permeability_effect,
        driver=driver,
        database=database,
        include_near=True,
    )

    exact_chunks = [_record_to_chunk(row) for row in rows["exact"]]
    near_chunks = [_record_to_chunk(row) for row in rows["near"]]

    return exact_chunks, near_chunks


def score_mechanism_coverage(text: str) -> Dict[str, Any]:
    combined = safe_text(text)
    hits = [term for term in MECHANISM_TERMS if term in combined]

    has_mechanism_field = any(
        key in combined
        for key in ["mechanism", "机制解释", "explanation", "pore_connectivity"]
    )
    has_process = any(
        term in combined
        for term in ["填充", "封堵", "压密", "胶结", "分选", "非均质", "颗粒", "细粒"]
    )
    has_pathway = any(
        term in combined
        for term in ["孔隙", "孔喉", "通道", "渗流", "连通", "有效过水断面"]
    )

    score = 0.0
    if has_mechanism_field:
        score += 0.35
    if has_process:
        score += 0.3
    if has_pathway:
        score += 0.3
    if len(hits) >= 4:
        score += 0.05

    return {
        "mechanism_score": round(min(score, 1.0), 2),
        "mechanism_hit": bool(score >= 0.55),
        "mechanism_terms": hits,
    }


def hydro_feature_rerank(
    retrieved_chunks: List[Dict[str, Any]],
    feature_value: str,
    permeability_effect: str,
    final_top_k: int = 10,
) -> List[Dict[str, Any]]:
    target_feature = safe_text(feature_value)
    target_effect = safe_text(permeability_effect)
    results: List[Dict[str, Any]] = []

    for chunk in retrieved_chunks:
        source_props = chunk.get("source_props") or {}
        chunk_feature = safe_text(source_props.get("feature_value", ""))
        chunk_effect = safe_text(source_props.get("permeability_effect", ""))
        mechanism = safe_text(source_props.get("mechanism", ""))
        text = " ".join(
            [
                safe_text(chunk.get("chunk_text", "")),
                mechanism,
                str(make_json_safe(source_props)),
            ]
        )
        mechanism_eval = score_mechanism_coverage(text)

        feature_exact = chunk_feature == target_feature
        effect_exact = chunk_effect == target_effect
        final_score = 0.0

        if feature_exact:
            final_score += 40.0
        if effect_exact:
            final_score += 35.0
        final_score += mechanism_eval["mechanism_score"] * 25.0

        use_for_context = bool(feature_exact and effect_exact and mechanism_eval["mechanism_hit"])

        results.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "source_id": chunk.get("source_id"),
                "final_score": round(final_score, 2),
                "semantic_score": 0.0,
                "rule_score": round(75.0 if feature_exact and effect_exact else final_score, 2),
                "evidence_score": round(mechanism_eval["mechanism_score"] * 100.0, 2),
                "matched_rules": [
                    item
                    for item, ok in [
                        ("feature_value_exact", feature_exact),
                        ("permeability_effect_exact", effect_exact),
                        ("mechanism_coverage", mechanism_eval["mechanism_hit"]),
                    ]
                    if ok
                ],
                "conflict_or_uncertainty": "" if use_for_context else "因果解释证据不完整",
                "rerank_reason": (
                    "HydroFeature 精确评分："
                    f"feature_value 命中={feature_exact}，"
                    f"permeability_effect 命中={effect_exact}，"
                    f"mechanism_score={mechanism_eval['mechanism_score']}，"
                    f"mechanism_terms={mechanism_eval['mechanism_terms']}。"
                ),
                "use_for_context": use_for_context,
                "rerank_route": "hydro_feature_rule",
                "scoring_method": "hydro_feature_exact_rule",
                "feature_value_hit": feature_exact,
                "permeability_effect_hit": effect_exact,
                "mechanism_hit": mechanism_eval["mechanism_hit"],
                "mechanism_score": mechanism_eval["mechanism_score"],
                "mechanism_terms": mechanism_eval["mechanism_terms"],
            }
        )

    results.sort(key=lambda x: float(x.get("final_score", 0.0) or 0.0), reverse=True)

    return results[:final_top_k]
