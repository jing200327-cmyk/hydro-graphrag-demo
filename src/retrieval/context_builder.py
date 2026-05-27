from __future__ import annotations

from typing import Any, Dict, List

from src.retrieval.graph_expander import graph_context_to_text


def merge_vector_and_graph_context(
    vector_results: List[Dict[str, Any]],
    graph_expansion_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    retrieved_chunks: List[Dict[str, Any]] = []

    for item in vector_results:
        cid = item.get("chunk_id")
        graph_context = graph_expansion_map.get(cid, {})
        graph_context_text = graph_context_to_text(graph_context)

        retrieved_chunks.append({
            "chunk_id": cid,
            "chunk_type": item.get("chunk_type"),
            "chunk_text": item.get("chunk_text"),
            "source_label": item.get("source_label"),
            "source_id": item.get("source_id"),
            "vector_score": float(item.get("vector_score", 0.0) or 0.0),
            "graph_context": graph_context,
            "graph_context_text": graph_context_text,
        })

    return retrieved_chunks