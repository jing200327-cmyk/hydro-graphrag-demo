from __future__ import annotations

from typing import Any, Dict

from src.query.entity_extractor import extract_core_entities, identify_hydro_factors
from src.query.intent_classifier import classify_question_intent
from src.query.query_expander import generate_search_keywords, build_retrieval_direction
from src.utils.text import safe_text


def analyze_user_question(user_question: str) -> Dict[str, Any]:
    question = safe_text(user_question)

    if not question:
        raise ValueError("user_question 不能为空。")

    entities = extract_core_entities(question)
    intent = classify_question_intent(question)
    hydro_factors = identify_hydro_factors(question, entities)
    search_keywords = generate_search_keywords(
        user_question=question,
        entities=entities,
        intent=intent,
        hydro_factors=hydro_factors,
    )
    retrieval_direction = build_retrieval_direction(
        intent=intent,
        entities=entities,
        hydro_factors=hydro_factors,
    )

    return {
        "核心实体": entities,
        "问题意图": intent,
        "检索关键词": search_keywords,
        "需要关注的水文地质因素": hydro_factors,
        "建议检索方向": retrieval_direction,
    }


analyze_user_question_for_retrieval = analyze_user_question