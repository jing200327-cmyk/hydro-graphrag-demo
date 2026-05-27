from __future__ import annotations

from typing import Any, Dict, List


def get_top_score(items: List[Dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0

    try:
        return float(items[0].get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def normalize_0_100(value: float) -> float:
    if 0 <= value <= 1:
        return value * 100
    return max(0.0, min(100.0, value))


def compute_final_confidence(
    retrieved_chunks: List[Dict[str, Any]],
    b_score_results: List[Dict[str, Any]],
    c_rerank_results: List[Dict[str, Any]],
) -> float:
    top_vector_score = 0.0
    if retrieved_chunks:
        top_vector_score = normalize_0_100(float(retrieved_chunks[0].get("vector_score", 0.0) or 0.0))

    top_b_score = get_top_score(b_score_results, "score")
    top_c_score = get_top_score(c_rerank_results, "final_score")

    evidence_score = 0.0
    if c_rerank_results:
        evidence_score = get_top_score(c_rerank_results, "evidence_score")

    graph_support_score = 0.0
    if retrieved_chunks:
        with_graph = sum(1 for x in retrieved_chunks[:5] if x.get("graph_context_text"))
        graph_support_score = min(with_graph / 5.0 * 100.0, 100.0)

    confidence = (
        0.25 * top_vector_score
        + 0.25 * top_b_score
        + 0.30 * top_c_score
        + 0.10 * evidence_score
        + 0.10 * graph_support_score
    )

    return round(max(0.0, min(100.0, confidence)), 2)