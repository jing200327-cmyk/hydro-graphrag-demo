from __future__ import annotations

from typing import Any, Dict

from configs.settings import DEEPSEEK_MODEL
from src.generation.llm_client import call_deepseek_final_answer
from src.utils.env import get_deepseek_client


def generate_answer(
    final_prompt: str,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    client = get_deepseek_client()

    return call_deepseek_final_answer(
        client=client,
        prompt=final_prompt,
        model=DEEPSEEK_MODEL,
        max_tokens=max_tokens,
    )