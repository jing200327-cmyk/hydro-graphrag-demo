from __future__ import annotations

from typing import Any, Dict, List


def graph_expand_chunks(
    driver,
    database: str,
    chunk_ids: List[str],
    max_neighbors_per_chunk: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """
    根据 EvidenceChunk 的 id / chunk_id 进行图谱扩展。

    注意：
    - driver 从外部传入，不在本模块里创建。
    - database 从外部传入，不在本模块里读取配置。
    - chunk_ids 通常来自 vector_search_chunks 的返回结果。
    """

    if not chunk_ids:
        return {}

    cypher = """
    UNWIND $chunk_ids AS cid
    MATCH (chunk:EvidenceChunk)
    WHERE chunk.id = cid OR chunk.chunk_id = cid

    OPTIONAL MATCH (source)-[:HAS_EVIDENCE_CHUNK]->(chunk)
    OPTIONAL MATCH (source)-[r]-(neighbor)

    WITH cid, chunk, source, r, neighbor

    RETURN
        cid AS chunk_id,
        chunk.chunk_text AS chunk_text,
        chunk.chunk_type AS chunk_type,
        chunk.source_label AS chunk_source_label,
        chunk.source_id AS chunk_source_id,

        CASE
            WHEN source IS NULL THEN []
            ELSE labels(source)
        END AS source_labels,

        CASE
            WHEN source IS NULL THEN null
            ELSE coalesce(
                source.id,
                source.borehole_id,
                source.layer_id,
                source.feature_value,
                source.level_name
            )
        END AS source_id,

        CASE
            WHEN source IS NULL THEN {}
            ELSE properties(source)
        END AS source_properties,

        collect(DISTINCT
            CASE
                WHEN neighbor IS NULL THEN null
                ELSE {
                    relation_type: type(r),
                    neighbor_labels: labels(neighbor),
                    neighbor_id: coalesce(
                        neighbor.id,
                        neighbor.borehole_id,
                        neighbor.layer_id,
                        neighbor.feature_value,
                        neighbor.level_name
                    ),
                    neighbor_properties: properties(neighbor)
                }
            END
        )[0..$max_neighbors_per_chunk] AS neighbors
    """

    with driver.session(database=database) as session:
        records = session.run(
            cypher,
            chunk_ids=chunk_ids,
            max_neighbors_per_chunk=int(max_neighbors_per_chunk),
        ).data()

    result: Dict[str, Dict[str, Any]] = {}

    for record in records:
        cid = record.get("chunk_id")

        if not cid:
            continue

        neighbors: List[Dict[str, Any]] = []

        for neighbor in record.get("neighbors") or []:
            if not neighbor:
                continue

            if (
                neighbor.get("relation_type") is None
                and neighbor.get("neighbor_id") is None
            ):
                continue

            neighbors.append(neighbor)

        result[cid] = {
            "chunk_text": record.get("chunk_text"),
            "chunk_type": record.get("chunk_type"),
            "chunk_source_label": record.get("chunk_source_label"),
            "chunk_source_id": record.get("chunk_source_id"),
            "source_labels": record.get("source_labels") or [],
            "source_id": record.get("source_id"),
            "source_properties": record.get("source_properties") or {},
            "neighbors": neighbors,
        }

    return result


def graph_context_to_text(context: Dict[str, Any]) -> str:
    """
    将 graph_expand_chunks 返回的结构化图谱上下文
    转成 LLM 可读的扁平文本。
    """

    if not context:
        return ""

    important_keys = [
        "borehole_id",
        "layer_id",
        "top_depth",
        "bottom_depth",
        "thickness",
        "mid",
        "lithology_major_v2",
        "lithology_minor_v3",
        "base_permeability_level",
        "feature_type",
        "feature_value",
        "permeability_effect",
        "effect_weight",
        "confidence",
        "k_value",
        "k_log10",
        "k_unit",
        "source",
        "reliability",
        "observed_permeability_level",
        "level_name",
        "explanation",
    ]

    parts: List[str] = []

    source_id = context.get("source_id")
    source_labels = context.get("source_labels") or []
    source_properties = context.get("source_properties") or {}

    if source_id:
        parts.append(f"source_id={source_id}")

    if source_labels:
        parts.append(f"source_labels={'/'.join(source_labels)}")

    for key in important_keys:
        value = source_properties.get(key)

        if value is not None:
            parts.append(f"{key}={value}")

    for neighbor in context.get("neighbors", []):
        relation_type = neighbor.get("relation_type")
        neighbor_labels = neighbor.get("neighbor_labels") or []
        neighbor_id = neighbor.get("neighbor_id")
        neighbor_properties = neighbor.get("neighbor_properties") or {}

        if relation_type:
            parts.append(f"relation={relation_type}")

        if neighbor_id:
            parts.append(f"neighbor_id={neighbor_id}")

        if neighbor_labels:
            parts.append(f"neighbor_labels={'/'.join(neighbor_labels)}")

        for key in important_keys:
            value = neighbor_properties.get(key)

            if value is not None:
                parts.append(f"{key}={value}")

    return "；".join([str(part) for part in parts if part])