from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.query.intent_classifier import classify_question_intent
from src.rerank.fact_scorer import fact_rerank_chunks
from src.rerank.scorer import match_hydro_rules, evidence_completeness_score
from src.utils.text import safe_text, coerce_score_to_100


FACT_QUERY_INTENT = "fact_query"


def _resolve_question_intent(
    user_question: str,
    intent: Optional[str] = None,
    question_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """
    解析当前问题意图。

    优先级：
    1. 显式传入 intent
    2. question_analysis["问题意图"]
    3. question_analysis["intent"]
    4. 重新调用 classify_question_intent(user_question)

    这样做的好处：
    - 不强依赖 qa_pipeline.py 已经改造完成
    - 即使旧 pipeline 没传 question_analysis，也能在 reranker 层自动识别 fact_query
    """

    if intent:
        return safe_text(intent)

    if question_analysis:
        for key in ["问题意图", "intent", "question_intent"]:
            value = safe_text(question_analysis.get(key, ""))
            if value:
                return value

    return classify_question_intent(user_question)


def _resolve_entities(
    entities: Optional[List[Dict[str, Any]]] = None,
    question_analysis: Optional[Dict[str, Any]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    解析 query translation 阶段抽取出的实体。

    兼容：
    - entities 显式传入
    - question_analysis["核心实体"]
    - question_analysis["entities"]
    """

    if entities is not None:
        return entities

    if not question_analysis:
        return None

    for key in ["核心实体", "entities", "core_entities"]:
        value = question_analysis.get(key)
        if isinstance(value, list):
            return value

    return None


def _normalize_fact_rerank_result(
    fact_item: Dict[str, Any],
    context_threshold: float,
) -> Dict[str, Any]:
    """
    将 fact_scorer.py 的输出转换成与 C rerank 尽量兼容的结构。

    原 C rerank 主要字段：
    - chunk_id
    - final_score
    - semantic_score
    - rule_score
    - evidence_score
    - matched_rules
    - conflict_or_uncertainty
    - rerank_reason
    - use_for_context

    fact_query 没有水文规则评分，所以：
    - rule_score 固定为 0
    - evidence_score 使用 final_score 近似表示事实证据质量
    - matched_rules 使用 matched_fields 表示命中的事实字段
    """

    final_score = coerce_score_to_100(fact_item.get("final_score"))

    matched_fields = fact_item.get("matched_fields", [])
    if not isinstance(matched_fields, list):
        matched_fields = []

    fact_match_details = fact_item.get("fact_match_details", [])
    if not isinstance(fact_match_details, list):
        fact_match_details = []

    rerank_reason = safe_text(fact_item.get("rerank_reason", ""))

    if not rerank_reason:
        rerank_reason = "事实查询评分：按 layer_id、borehole_id、深度、岩性、语义相似度进行加权排序。"

    return {
        # 原 C rerank 兼容字段
        "chunk_id": fact_item.get("chunk_id"),
        "final_score": round(final_score, 2),
        "semantic_score": round(float(fact_item.get("semantic_score", 0.0) or 0.0), 2),
        "rule_score": 0.0,
        "evidence_score": round(final_score, 2),
        "matched_rules": matched_fields,
        "conflict_or_uncertainty": "",
        "rerank_reason": rerank_reason,
        "use_for_context": bool(final_score >= context_threshold),

        # 新增：路由和事实查询评分字段
        "rerank_route": "fact_query",
        "scoring_method": fact_item.get("scoring_method", "fact_query_rule_score"),
        "fact_raw_score": fact_item.get("fact_raw_score", 0.0),
        "active_weight": fact_item.get("active_weight", 0.0),
        "active_dimensions": fact_item.get("active_dimensions", []),
        "layer_id_score": fact_item.get("layer_id_score", 0.0),
        "borehole_id_score": fact_item.get("borehole_id_score", 0.0),
        "depth_score": fact_item.get("depth_score", 0.0),
        "lithology_score": fact_item.get("lithology_score", 0.0),
        "fact_semantic_score": fact_item.get("semantic_score", 0.0),
        "matched_fields": matched_fields,
        "fact_match_details": fact_match_details,

        # 保留部分检索来源信息，方便前端或调试展示
        "source_id": fact_item.get("source_id"),
        "source_label": fact_item.get("source_label"),
        "chunk_type": fact_item.get("chunk_type"),
        "retrieval_method": fact_item.get("retrieval_method"),
        "exact_match_type": fact_item.get("exact_match_type"),
        "exact_match_value": fact_item.get("exact_match_value"),
        "is_synthetic": fact_item.get("is_synthetic", False),
    }


def _c_rerank_hydro_chunks(
    user_question: str,
    retrieved_chunks: List[Dict[str, Any]],
    b_score_results: List[Dict[str, Any]],
    context_threshold: float = 60.0,
    final_top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    原有 C rerank 逻辑。

    用于：
    - 渗透率查询
    - 透水等级查询
    - 因果解释
    - 多条件水文地质问题
    """

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

                # 新增：标记这是原水文规则重排路线
                "rerank_route": "hydro_rule",
                "scoring_method": "c_hydro_rule_rerank",
            }
        )

    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    return results[:final_top_k]


def _c_rerank_fact_chunks(
    user_question: str,
    retrieved_chunks: List[Dict[str, Any]],
    entities: Optional[List[Dict[str, Any]]] = None,
    context_threshold: float = 60.0,
    final_top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    fact_query 专用 rerank 路由。

    这里调用 fact_scorer.py，而不是原来的 B/C 水文规则评分。

    核心原则：
    - layer_id 精确匹配权重最高
    - borehole_id 次之
    - 深度和岩性字段用于事实查询补强
    - 语义相似度只作为弱辅助信号
    """

    if not retrieved_chunks:
        return []

    fact_results = fact_rerank_chunks(
        user_question=user_question,
        chunks=retrieved_chunks,
        entities=entities,
        top_k=final_top_k,
    )

    normalized_results = [
        _normalize_fact_rerank_result(
            fact_item=item,
            context_threshold=context_threshold,
        )
        for item in fact_results
    ]

    normalized_results = sorted(
        normalized_results,
        key=lambda x: float(x.get("final_score", 0.0) or 0.0),
        reverse=True,
    )

    return normalized_results[:final_top_k]


def c_rerank_chunks(
    user_question: str,
    retrieved_chunks: List[Dict[str, Any]],
    b_score_results: Optional[List[Dict[str, Any]]] = None,
    context_threshold: float = 60.0,
    final_top_k: int = 10,
    question_analysis: Optional[Dict[str, Any]] = None,
    intent: Optional[str] = None,
    entities: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    C rerank 总入口：按 intent 路由。

    路由规则：
    1. fact_query:
       - 不走原 B/C 水文规则评分
       - 直接使用 fact_scorer.py 的事实查询评分
       - 可在 b_score_results 为空时正常工作

    2. 非 fact_query:
       - 保持原有逻辑
       - 使用 B 综合评分 + 水文规则评分 + 证据完整性评分

    参数兼容：
    - 旧调用方式仍然可用：
      c_rerank_chunks(user_question, retrieved_chunks, b_score_results)

    - 新推荐调用方式：
      c_rerank_chunks(
          user_question=user_question,
          retrieved_chunks=retrieved_chunks,
          b_score_results=b_score_results,
          question_analysis=question_analysis,
      )
    """

    resolved_intent = _resolve_question_intent(
        user_question=user_question,
        intent=intent,
        question_analysis=question_analysis,
    )

    resolved_entities = _resolve_entities(
        entities=entities,
        question_analysis=question_analysis,
    )

    if resolved_intent == FACT_QUERY_INTENT:
        return _c_rerank_fact_chunks(
            user_question=user_question,
            retrieved_chunks=retrieved_chunks,
            entities=resolved_entities,
            context_threshold=context_threshold,
            final_top_k=final_top_k,
        )

    return _c_rerank_hydro_chunks(
        user_question=user_question,
        retrieved_chunks=retrieved_chunks,
        b_score_results=b_score_results or [],
        context_threshold=context_threshold,
        final_top_k=final_top_k,
    )