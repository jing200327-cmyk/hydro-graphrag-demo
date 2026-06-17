from __future__ import annotations

from typing import Any, Dict, List

from src.utils.text import safe_text, unique_keep_order


COREFERENCE_TERMS = [
    "这些",
    "上述",
    "上面",
    "前面",
    "该",
    "其",
    "它",
    "其中",
    "这些钻孔",
    "这些分层",
    "该段",
    "该组",
    "上一轮",
]


def _first(values: List[str]) -> str:
    return values[0] if values else ""


def _limited_join(values: List[str], limit: int = 12) -> str:
    values = unique_keep_order(values)

    if not values:
        return ""

    shown = values[:limit]
    suffix = f"等{len(values)}个" if len(values) > limit else ""
    return "、".join(shown) + suffix


def rewrite_user_question(
    user_question: str,
    history_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Deterministically rewrite follow-up questions into standalone queries.

    This avoids an extra LLM call in the MVP. It only adds explicit constraints
    when the current question contains clear coreference terms.
    """

    question = safe_text(user_question)
    history_context = history_context or {}

    if not question:
        return {
            "rewritten_question": "",
            "rewritten": False,
            "rewrite_reason": "empty_question",
        }

    if not any(term in question for term in COREFERENCE_TERMS):
        return {
            "rewritten_question": question,
            "rewritten": False,
            "rewrite_reason": "no_coreference_term",
        }

    entities = history_context.get("recent_entities") or {}
    constraints: List[str] = []

    strat_member = _first(entities.get("strat_member") or [])
    strat_group = _first(entities.get("strat_group") or [])
    geologic_epoch = _first(entities.get("geologic_epoch") or [])
    geologic_period = _first(entities.get("geologic_period") or [])
    boreholes = _limited_join(entities.get("borehole_id") or [])
    layers = _limited_join(entities.get("layer_id") or [])

    if strat_member and strat_member not in question:
        constraints.append(f"地层段={strat_member}")

    if strat_group and strat_group not in question:
        constraints.append(f"地层组={strat_group}")

    if geologic_epoch and geologic_epoch not in question:
        constraints.append(f"地质世={geologic_epoch}")

    if geologic_period and geologic_period not in question:
        constraints.append(f"地质纪={geologic_period}")

    if "钻孔" in question and boreholes and not any(b in question for b in entities.get("borehole_id", [])):
        constraints.append(f"相关钻孔={boreholes}")

    if "分层" in question and layers and not any(l in question for l in entities.get("layer_id", [])):
        constraints.append(f"相关分层={layers}")

    if not constraints:
        return {
            "rewritten_question": question,
            "rewritten": False,
            "rewrite_reason": "no_reusable_entities",
        }

    rewritten_question = f"在{'，'.join(constraints)}的上下文中，{question}"

    return {
        "rewritten_question": rewritten_question,
        "rewritten": rewritten_question != question,
        "rewrite_reason": "coreference_constraints_added",
        "applied_constraints": constraints,
    }
