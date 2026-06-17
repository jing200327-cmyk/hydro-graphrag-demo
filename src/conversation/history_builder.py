from __future__ import annotations

from typing import Any, Dict, List

from src.utils.text import safe_text, truncate_text, unique_keep_order


ENTITY_KEYS = [
    "borehole_id",
    "layer_id",
    "lithology_major_v2",
    "lithology_minor_v3",
    "geologic_period",
    "geologic_epoch",
    "strat_group",
    "strat_member",
]


def _append_entity(entities: Dict[str, List[str]], key: str, value: Any) -> None:
    text = safe_text(value)
    if not text:
        return
    entities.setdefault(key, []).append(text)


def _collect_entities_from_result(result: Dict[str, Any]) -> Dict[str, List[str]]:
    entities: Dict[str, List[str]] = {}

    question_analysis = result.get("question_analysis") or {}
    for item in question_analysis.get("核心实体") or []:
        field_name = safe_text(item.get("字段名"))
        entity_type = safe_text(item.get("实体类型"))
        entity_value = safe_text(item.get("实体值"))

        if field_name:
            _append_entity(entities, field_name, entity_value)
        elif entity_type:
            _append_entity(entities, entity_type, entity_value)

    for chunk in result.get("vector_search_results") or []:
        props = chunk.get("source_props") or {}
        for key in ENTITY_KEYS:
            _append_entity(entities, key, props.get(key))

    return {
        key: unique_keep_order(values)
        for key, values in entities.items()
        if unique_keep_order(values)
    }


def build_history_context(
    turns: List[Dict[str, Any]],
    summary: str = "",
    max_turns: int = 5,
    max_answer_chars: int = 500,
) -> Dict[str, Any]:
    """
    Build a compact, RAG-safe conversation context.

    History is used for coreference and query rewriting only. It is not treated
    as knowledge evidence; the current turn must still retrieve from Neo4j and
    EvidenceChunk.
    """

    recent_turns = turns[-max_turns:] if max_turns > 0 else []
    compact_turns: List[Dict[str, Any]] = []
    merged_entities: Dict[str, List[str]] = {}

    for turn in recent_turns:
        result = turn.get("retrieval_snapshot") or {}
        turn_entities = _collect_entities_from_result(result)

        for key, values in turn_entities.items():
            merged_entities.setdefault(key, []).extend(values)

        compact_turns.append(
            {
                "turn_index": turn.get("turn_index"),
                "user_question": safe_text(turn.get("user_question")),
                "rewritten_question": safe_text(turn.get("rewritten_question")),
                "question_intent": safe_text(turn.get("question_intent")),
                "retrieval_route": safe_text(turn.get("retrieval_route")),
                "final_answer_summary": truncate_text(
                    turn.get("final_answer"),
                    max_answer_chars,
                ),
                "entities": turn_entities,
            }
        )

    return {
        "summary": safe_text(summary),
        "recent_turns": compact_turns,
        "recent_entities": {
            key: unique_keep_order(values)
            for key, values in merged_entities.items()
            if unique_keep_order(values)
        },
        "usage": "history_for_coreference_only",
    }
