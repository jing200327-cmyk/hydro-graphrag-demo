# src/generation/llm_client.py

from __future__ import annotations

from typing import Any, Dict, Optional

from configs.settings import DEEPSEEK_MODEL
from src.generation.prompt_builder import FINAL_ANSWER_SYSTEM_PROMPT


def call_deepseek_final_answer(
    client: Any,
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    if model is None:
        model = DEEPSEEK_MODEL

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": FINAL_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        max_tokens=max_tokens,
        temperature=0.2,
    )

    message = response.choices[0].message

    usage = None
    if getattr(response, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

    return {
        "model": model,
        "answer": message.content,
        "usage": usage,
    }