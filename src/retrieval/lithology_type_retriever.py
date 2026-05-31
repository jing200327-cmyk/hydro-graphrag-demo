from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from configs.settings import NEO4J_DATABASE
from src.fallback.entity_guard import extract_lithology_values
from src.utils.env import get_driver
from src.utils.text import make_json_safe, safe_text


def is_lithology_permeability_level_query(
    user_question: str,
    question_analysis: Optional[Dict[str, Any]] = None,
) -> bool:
    question = safe_text(user_question)

    if not question:
        return False

    if "基础透水等级倾向" not in question:
        return False

    if "岩性大类" not in question or "岩性小类" not in question:
        return False

    major_values, minor_values = extract_lithology_values(
        question,
        question_analysis or {},
    )

    return bool(major_values and minor_values)


def extract_lithology_type_query(
    user_question: str,
    question_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    major_values, minor_values = extract_lithology_values(
        safe_text(user_question),
        question_analysis or {},
    )

    return {
        "lithology_major_v2": major_values[0] if major_values else "",
        "lithology_minor_v3": minor_values[0] if minor_values else "",
    }


def _run_cypher(
    driver: Any,
    database: str,
    cypher: str,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    with driver.session(database=database) as session:
        return session.run(cypher, params).data()


def fetch_lithology_type_candidates(
    lithology_major_v2: str,
    lithology_minor_v3: str,
    driver: Any = None,
    database: str = NEO4J_DATABASE,
    include_near: bool = True,
) -> Dict[str, Any]:
    """
    Fetch exact LithologyType evidence first, plus optional near records for
    diagnostics only. Near records must not be used as answer evidence.
    """

    major = safe_text(lithology_major_v2)
    minor = safe_text(lithology_minor_v3)

    if not major or not minor:
        return {
            "exact": [],
            "near": [],
            "lithology_major_v2": major,
            "lithology_minor_v3": minor,
        }

    driver = driver or get_driver()

    exact_rows = _run_cypher(
        driver,
        database,
        """
        MATCH (lt:LithologyType)
        WHERE lt.lithology_major_v2 = $major
          AND lt.lithology_minor_v3 = $minor
        OPTIONAL MATCH (lt)-[:HAS_EVIDENCE_CHUNK]->(chunk:EvidenceChunk)
        RETURN
            properties(lt) AS source_props,
            properties(chunk) AS chunk_props
        LIMIT 10
        """,
        {
            "major": major,
            "minor": minor,
        },
    )

    near_rows: List[Dict[str, Any]] = []

    if include_near and not exact_rows:
        near_rows = _run_cypher(
            driver,
            database,
            """
            MATCH (lt:LithologyType)
            WHERE lt.lithology_major_v2 = $major
               OR lt.lithology_minor_v3 = $minor
               OR lt.lithology_minor_v3 CONTAINS $minor
               OR $minor CONTAINS lt.lithology_minor_v3
            OPTIONAL MATCH (lt)-[:HAS_EVIDENCE_CHUNK]->(chunk:EvidenceChunk)
            RETURN
                properties(lt) AS source_props,
                properties(chunk) AS chunk_props
            LIMIT 10
            """,
            {
                "major": major,
                "minor": minor,
            },
        )

    return {
        "exact": exact_rows,
        "near": near_rows,
        "lithology_major_v2": major,
        "lithology_minor_v3": minor,
    }


def _source_props_to_text(props: Dict[str, Any]) -> str:
    major = safe_text(props.get("lithology_major_v2", ""))
    minor = safe_text(props.get("lithology_minor_v3", ""))
    level = safe_text(props.get("base_permeability_level", ""))

    if major and minor and level:
        return (
            "岩性透水性倾向证据："
            f"岩性大类为 {major}，岩性小类为 {minor}，"
            f"在 demo 知识库中基础透水等级倾向为 {level}。"
        )

    return "LithologyType 结构化记录：" + str(make_json_safe(props))


def _light_props(props: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in (props or {}).items()
        if key not in {"embedding", "embedding_vector"}
    }


def _record_to_chunk(record: Dict[str, Any]) -> Dict[str, Any]:
    source_props = _light_props(record.get("source_props") or {})
    chunk_props = _light_props(record.get("chunk_props") or {})

    source_id = safe_text(source_props.get("id", ""))
    if not source_id:
        source_id = (
            "LithologyType:"
            f"{safe_text(source_props.get('lithology_major_v2', ''))}|"
            f"{safe_text(source_props.get('lithology_minor_v3', ''))}"
        )

    chunk_id = safe_text(chunk_props.get("chunk_id", "")) or safe_text(chunk_props.get("id", ""))

    if not chunk_id:
        chunk_id = f"exact::lithology_type::{source_id}"

    chunk_text = safe_text(chunk_props.get("chunk_text", ""))
    if not chunk_text:
        chunk_text = _source_props_to_text(source_props)

    return {
        "chunk_id": chunk_id,
        "chunk_type": safe_text(chunk_props.get("chunk_type", "")) or "lithology_tendency_chunk",
        "chunk_text": chunk_text,
        "source_label": "LithologyType",
        "source_id": source_id,
        "vector_score": 1.0,
        "retrieval_method": "lithology_type_exact",
        "exact_match_type": "lithology_major_minor",
        "exact_match_value": source_id,
        "exact_score": 100.0,
        "matched_fields": ["lithology_major_v2", "lithology_minor_v3"],
        "source_props": make_json_safe(source_props),
        "chunk_props": make_json_safe(chunk_props),
        "is_synthetic": not bool(chunk_props),
        "graph_expandable": bool(chunk_props),
    }


def retrieve_lithology_type_exact(
    lithology_major_v2: str,
    lithology_minor_v3: str,
    driver: Any = None,
    database: str = NEO4J_DATABASE,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = fetch_lithology_type_candidates(
        lithology_major_v2=lithology_major_v2,
        lithology_minor_v3=lithology_minor_v3,
        driver=driver,
        database=database,
        include_near=True,
    )

    exact_chunks = [_record_to_chunk(row) for row in rows["exact"]]
    near_chunks = [_record_to_chunk(row) for row in rows["near"]]

    return exact_chunks, near_chunks


def lithology_type_rerank(
    retrieved_chunks: List[Dict[str, Any]],
    lithology_major_v2: str,
    lithology_minor_v3: str,
    final_top_k: int = 10,
) -> List[Dict[str, Any]]:
    major = safe_text(lithology_major_v2)
    minor = safe_text(lithology_minor_v3)

    results: List[Dict[str, Any]] = []

    for chunk in retrieved_chunks:
        source_props = chunk.get("source_props") or {}
        chunk_major = safe_text(source_props.get("lithology_major_v2", ""))
        chunk_minor = safe_text(source_props.get("lithology_minor_v3", ""))

        major_exact = chunk_major == major
        minor_exact = chunk_minor == minor
        contains_only = (
            not minor_exact
            and bool(minor)
            and bool(chunk_minor)
            and (minor in chunk_minor or chunk_minor in minor)
        )

        score = 100.0 if major_exact and minor_exact else 0.0

        if contains_only:
            score = min(score, 20.0)

        use_for_context = bool(major_exact and minor_exact)

        if major_exact and minor_exact:
            reason = (
                "LithologyType 精确匹配："
                f"lithology_major_v2={major}，lithology_minor_v3={minor}。"
            )
        elif contains_only:
            reason = (
                "包含式近似匹配被降权："
                f"用户小类={minor}，候选小类={chunk_minor}，不允许作为事实答案。"
            )
        else:
            reason = "LithologyType 未同时精确匹配用户指定岩性大类和岩性小类。"

        results.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "source_id": chunk.get("source_id"),
                "final_score": score,
                "semantic_score": 0.0,
                "rule_score": score,
                "evidence_score": 100.0 if use_for_context else 0.0,
                "matched_rules": ["lithology_major_minor_exact"] if use_for_context else [],
                "conflict_or_uncertainty": "" if use_for_context else "非精确岩性 pair",
                "rerank_reason": reason,
                "use_for_context": use_for_context,
                "rerank_route": "lithology_type_rule",
                "scoring_method": "lithology_type_exact_rule",
            }
        )

    results.sort(key=lambda x: float(x.get("final_score", 0.0) or 0.0), reverse=True)

    return results[:final_top_k]
