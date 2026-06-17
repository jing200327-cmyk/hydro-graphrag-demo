# qa_backend.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "conversation_id": raw_result.get("conversation_id", ""),
        "turn_id": raw_result.get("turn_id", ""),
        "original_user_question": raw_result.get("original_user_question", raw_result.get("user_question", "")),
        "rewritten_question": raw_result.get("rewritten_question", raw_result.get("user_question", "")),
        "history_context": raw_result.get("history_context", {}),
        "query_rewrite": raw_result.get("query_rewrite", {}),
        "raw": raw_result,
    }


def run_qa(
    user_question: str,
    top_k: int = 10,
    raw_top_k: int = 30,
    b_keep_threshold: float = 60.0,
    c_context_threshold: float = 60.0,
    max_tokens: int = 2048,
    conversation_id: Optional[str] = None,
    persist_history: bool = True,
) -> Dict[str, Any]:
    from src.conversation.query_rewriter import rewrite_user_question
    from src.conversation.service import ConversationService, timed_ms
    from src.pipeline.qa_pipeline import run_end_to_end_graphrag_qa

    started_at = time.perf_counter()
    conversation_service = ConversationService()
    effective_conversation_id = conversation_service.ensure_conversation(conversation_id)
    history_context = conversation_service.build_history_context(effective_conversation_id)
    rewrite_result = rewrite_user_question(
        user_question=user_question,
        history_context=history_context,
    )
    rewritten_question = rewrite_result.get("rewritten_question") or user_question

    print("=" * 100)
    print("[Streamlit] Python:", sys.executable)
    print("[Streamlit] APP_DIR:", APP_DIR)
    print("[Streamlit] REAL_RAG_DIR:", REAL_RAG_DIR)
    print("[Streamlit] user_question:", user_question)
    print("[Streamlit] rewritten_question:", rewritten_question)
    print("[Streamlit] conversation_id:", effective_conversation_id)
    print("[Streamlit] raw_top_k:", raw_top_k)
    print("[Streamlit] final_top_k:", top_k)
    print("=" * 100)

    raw_result = run_end_to_end_graphrag_qa(
        user_question=rewritten_question,
        original_user_question=user_question,
        history_context=history_context,
        query_rewrite=rewrite_result,
        raw_top_k=raw_top_k,
        final_top_k=top_k,
        b_keep_threshold=b_keep_threshold,
        c_context_threshold=c_context_threshold,
        max_tokens=max_tokens,
        save_outputs=True,
        enable_llm=True,
        enable_rankgpt=False,
    )

    if isinstance(raw_result, dict):
        raw_result["conversation_id"] = effective_conversation_id
        raw_result["original_user_question"] = user_question
        raw_result["rewritten_question"] = rewritten_question
        raw_result["history_context"] = history_context
        raw_result["query_rewrite"] = rewrite_result

        if persist_history:
            turn_id = conversation_service.append_turn(
                conversation_id=effective_conversation_id,
                user_question=user_question,
                rewritten_question=rewritten_question,
                result=raw_result,
                query_rewrite=rewrite_result,
                history_context=history_context,
                latency_ms=timed_ms(started_at),
            )
            raw_result["turn_id"] = turn_id

    return normalize_result(raw_result)


def answer_question(
    question: str,
    top_k: int = 10,
    raw_top_k: int = 30,
    b_keep_threshold: float = 60.0,
    c_context_threshold: float = 60.0,
    max_tokens: int = 2048,
    conversation_id: Optional[str] = None,
    persist_history: bool = True,
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
        conversation_id=conversation_id,
        persist_history=persist_history,
    )


__all__ = [
    "run_qa",
    "answer_question",
    "normalize_result",
    "flatten_graph_context",
]
