from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.query.entity_extractor import (
    HYDRO_FACTOR_KEYWORDS,
    PERMEABILITY_DIRECT_KEYWORDS,
    CAUSAL_KEYWORDS,
    load_vocab_cache,
    find_vocab_matches,
)
from src.utils.text import (
    safe_text,
    unique_keep_order,
)


def collect_question_entity_values(question_analysis: Dict[str, Any]) -> List[str]:
    entities = question_analysis.get("核心实体", [])
    values: List[str] = []

    for entity in entities:
        if isinstance(entity, dict) and entity.get("实体值"):
            values.append(entity["实体值"])

    return unique_keep_order(values)


def match_entities_in_text(
    user_question: str,
    text: str,
    question_analysis: Dict[str, Any],
) -> List[str]:
    vocab = load_vocab_cache()
    candidate_terms: List[str] = []
    candidate_terms.extend(collect_question_entity_values(question_analysis))

    for key in [
        "borehole",
        "layer",
        "lithology_major",
        "lithology_minor",
        "feature_value",
        "permeability_level",
    ]:
        candidate_terms.extend(
            find_vocab_matches(user_question, vocab.get(key, []), max_count=30)
        )

    combined_text = f"{user_question} {text}"

    return unique_keep_order(
        [
            term
            for term in unique_keep_order(candidate_terms)
            if term and term in combined_text
        ]
    )


def match_permeability_factors_in_text(user_question: str, text: str) -> List[str]:
    combined_text = f"{user_question} {text}"
    matched: List[str] = []

    for factor_name, keywords in HYDRO_FACTOR_KEYWORDS.items():
        for kw in keywords:
            if kw in combined_text:
                matched.extend([factor_name, kw])

    for kw in PERMEABILITY_DIRECT_KEYWORDS:
        if kw in combined_text:
            matched.append(kw)

    return unique_keep_order(matched)


def match_causal_terms(text: str) -> List[str]:
    return unique_keep_order([kw for kw in CAUSAL_KEYWORDS if kw in text])


def normalize_vector_score(vector_score: Optional[float]) -> float:
    if vector_score is None:
        return 0.0

    try:
        value = float(vector_score)
    except Exception:
        return 0.0

    return max(0.0, min(1.0, value))


def keyword_overlap_score(
    user_question: str,
    text: str,
    keywords: List[str],
) -> float:
    if not keywords:
        return 0.0

    matched = [
        kw
        for kw in keywords
        if kw and (kw in text or kw in user_question)
    ]

    return max(0.0, min(1.0, len(set(matched)) / max(len(set(keywords)), 1)))