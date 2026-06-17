from __future__ import annotations

from typing import Any, Dict, List

from src.utils.text import safe_text, truncate_text


def summarize_turns(
    turns: List[Dict[str, Any]],
    max_turns: int = 8,
    max_chars: int = 1200,
) -> str:
    """
    Deterministic conversation summary placeholder.

    This keeps the module boundary stable for later LLM-based summarization while
    avoiding an extra model call in the first persistence rollout.
    """

    recent_turns = turns[-max_turns:] if max_turns > 0 else []
    lines: List[str] = []

    for turn in recent_turns:
        user_question = safe_text(turn.get("user_question"))
        final_answer = truncate_text(turn.get("final_answer"), 180)
        retrieval_route = safe_text(turn.get("retrieval_route"))

        if not user_question and not final_answer:
            continue

        lines.append(
            f"- 用户问：{user_question}；路由：{retrieval_route or 'unknown'}；回答摘要：{final_answer}"
        )

    return truncate_text("\n".join(lines), max_chars)
