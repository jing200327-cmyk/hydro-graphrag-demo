from __future__ import annotations

from typing import Any, Dict, List


def vector_search_chunks(
    driver,
    database: str,
    index_name: str,
    query_embedding: List[float],
    top_k: int = 30,
) -> List[Dict[str, Any]]:
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
    YIELD node, score
    RETURN
        coalesce(node.id, node.chunk_id) AS chunk_id,
        node.chunk_id AS chunk_id_property,
        node.chunk_type AS chunk_type,
        node.chunk_text AS chunk_text,
        node.source_label AS source_label,
        node.source_id AS source_id,
        score AS vector_score
    ORDER BY vector_score DESC
    """

    with driver.session(database=database) as session:
        return session.run(
            cypher,
            index_name=index_name,
            top_k=int(top_k),
            query_embedding=query_embedding,
        ).data()