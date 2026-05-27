from __future__ import annotations

from typing import Any, Dict, List

from src.rerank.scorer import match_hydro_rules, evidence_completeness_score
from src.utils.text import safe_text, coerce_score_to_100


def c_rerank_chunks(
    user_question: str,
    retrieved_chunks: List[Dict[str, Any]],
    b_score_results: List[Dict[str, Any]],
    context_threshold: float = 60.0,
    final_top_k: int = 10,
) -> List[Dict[str, Any]]:
    if not b_score_results:
        return []

    chunk_map = {
        item.get("chunk_id"): item
        for item in retrieved_chunks
        if item.get("chunk_id")
    }

    results: List[Dict[str, Any]] = []

    for b_item in b_score_results:
        cid = b_item.get("chunk_id")
        chunk = chunk_map.get(cid, {})

        text = (
            f"{user_question}；"
            f"{safe_text(chunk.get('chunk_text'))}；"
            f"{safe_text(chunk.get('graph_context_text'))}；"
            f"{safe_text(b_item.get('reason'))}"
        )

        semantic_score = coerce_score_to_100(b_item.get("score"))
        rule_score, matched_rules, uncertainty_flags, _detail = match_hydro_rules(text)
        evidence_score, evidence_hits = evidence_completeness_score(chunk)

        final_score = round(
            0.50 * semantic_score
            + 0.35 * rule_score
            + 0.15 * evidence_score,
            2,
        )

        rerank_reason = (
            f"综合 B 语义/实体/因素评分 {semantic_score:.1f}，"
            f"水文规则匹配 {rule_score:.1f}，"
            f"证据完整性 {evidence_score:.1f}。"
        )

        if evidence_hits:
            rerank_reason += "证据完整性依据：" + "；".join(evidence_hits[:5]) + "。"

        results.append(
            {
                "chunk_id": cid,
                "final_score": final_score,
                "semantic_score": round(semantic_score, 2),
                "rule_score": round(rule_score, 2),
                "evidence_score": round(evidence_score, 2),
                "matched_rules": matched_rules,
                "conflict_or_uncertainty": "；".join(uncertainty_flags),
                "rerank_reason": rerank_reason,
                "use_for_context": bool(final_score >= context_threshold),
            }
        )

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    return results[:final_top_k]