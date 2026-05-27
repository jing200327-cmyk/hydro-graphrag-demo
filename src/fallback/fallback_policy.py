from __future__ import annotations

from typing import Any, Dict, List


def should_abstain(
    confidence: float,
    retrieved_chunks: List[Dict[str, Any]],
    c_rerank_results: List[Dict[str, Any]],
    min_confidence: float = 45.0,
) -> bool:
    if not retrieved_chunks:
        return True

    if not c_rerank_results:
        return True

    if confidence < min_confidence:
        return True

    return False


def build_fallback_answer(reason: str = "") -> str:
    base = "根据当前知识库检索结果，未找到足够可靠的证据支撑该问题的确定性回答。"

    if reason:
        return f"{base}\n\n原因：{reason}"

    return base


def apply_fallback_policy(
    confidence: float,
    retrieved_chunks: List[Dict[str, Any]],
    c_rerank_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not retrieved_chunks:
        return {
            "fallback_triggered": True,
            "fallback_answer": build_fallback_answer("向量检索未召回相关 EvidenceChunk。"),
        }

    if not c_rerank_results:
        return {
            "fallback_triggered": True,
            "fallback_answer": build_fallback_answer("重排序阶段未得到可用于回答的证据。"),
        }

    if confidence < 45:
        return {
            "fallback_triggered": True,
            "fallback_answer": build_fallback_answer(f"综合置信度较低，confidence={confidence}。"),
        }

    return {
        "fallback_triggered": False,
        "fallback_answer": "",
    }