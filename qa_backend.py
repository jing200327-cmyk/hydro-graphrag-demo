# qa_backend.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent

REAL_RAG_DIR = Path(
    os.getenv("HYDRO_GRAPHRAG_PROJECT_DIR", str(APP_DIR))
).resolve()

if str(REAL_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_RAG_DIR))

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

load_dotenv(REAL_RAG_DIR / ".env", override=True)
load_dotenv(APP_DIR / ".env", override=True)


def flatten_graph_context(
    vector_search_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for item in vector_search_results or []:
        rows.append(
            {
                "chunk_id": item.get("chunk_id"),
                "graph_context_text": item.get("graph_context_text", ""),
                "graph_context": item.get("graph_context", {}),
            }
        )

    return rows


def normalize_result(raw_result: Any) -> Dict[str, Any]:
    if isinstance(raw_result, str):
        return {
            "final_answer": raw_result,
            "question_analysis": {},
            "retrieved_chunks": [],
            "graph_context": [],
            "b_score_results": [],
            "c_rerank_results": [],
            "final_prompt": "",
            "deepseek_usage": None,
            "confidence": None,
            "fallback_triggered": None,
            "raw": raw_result,
        }

    if not isinstance(raw_result, dict):
        return {
            "final_answer": str(raw_result),
            "question_analysis": {},
            "retrieved_chunks": [],
            "graph_context": [],
            "b_score_results": [],
            "c_rerank_results": [],
            "final_prompt": "",
            "deepseek_usage": None,
            "confidence": None,
            "fallback_triggered": None,
            "raw": raw_result,
        }

    vector_search_results = raw_result.get("vector_search_results", [])

    return {
        "final_answer": raw_result.get("final_answer", "未生成最终答案。"),
        "question_analysis": raw_result.get("question_analysis", {}),
        "retrieved_chunks": vector_search_results,
        "graph_context": flatten_graph_context(vector_search_results),
        "b_score_results": raw_result.get("b_score_results", []),
        "c_rerank_results": raw_result.get("c_rerank_results", []),
        "final_prompt": raw_result.get("final_prompt", ""),
        "deepseek_usage": raw_result.get("deepseek_usage"),
        "confidence": raw_result.get("confidence"),
        "fallback_triggered": raw_result.get("fallback_triggered"),
        "raw": raw_result,
    }


def run_qa(
    user_question: str,
    top_k: int = 10,
    raw_top_k: int = 30,
    b_keep_threshold: float = 60.0,
    c_context_threshold: float = 60.0,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    from src.pipeline.qa_pipeline import run_end_to_end_graphrag_qa

    print("=" * 100)
    print("[Streamlit] Python:", sys.executable)
    print("[Streamlit] APP_DIR:", APP_DIR)
    print("[Streamlit] REAL_RAG_DIR:", REAL_RAG_DIR)
    print("[Streamlit] user_question:", user_question)
    print("[Streamlit] raw_top_k:", raw_top_k)
    print("[Streamlit] final_top_k:", top_k)
    print("=" * 100)

    raw_result = run_end_to_end_graphrag_qa(
        user_question=user_question,
        raw_top_k=raw_top_k,
        final_top_k=top_k,
        b_keep_threshold=b_keep_threshold,
        c_context_threshold=c_context_threshold,
        max_tokens=max_tokens,
        save_outputs=True,
        enable_llm=True,
    )

    return normalize_result(raw_result)


def answer_question(
    question: str,
    top_k: int = 10,
    raw_top_k: int = 30,
    b_keep_threshold: float = 60.0,
    c_context_threshold: float = 60.0,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """
    兼容旧接口。

    如果旧代码里调用 answer_question(...)，
    这里统一转发到 run_qa(...)。
    """

    return run_qa(
        user_question=question,
        top_k=top_k,
        raw_top_k=raw_top_k,
        b_keep_threshold=b_keep_threshold,
        c_context_threshold=c_context_threshold,
        max_tokens=max_tokens,
    )


__all__ = [
    "run_qa",
    "answer_question",
    "normalize_result",
    "flatten_graph_context",
]