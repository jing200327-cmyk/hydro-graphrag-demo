from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from configs.settings import (
    NEO4J_DATABASE,
    VECTOR_INDEX_NAME,
    OUTPUT_DIR,
)
from src.query.parser import analyze_user_question
from src.retrieval.embedder import encode_query
from src.retrieval.neo4j_vector_retriever import vector_search_chunks
from src.retrieval.graph_expander import graph_expand_chunks
from src.retrieval.context_builder import merge_vector_and_graph_context
from src.rerank.scorer import b_score_chunks
from src.rerank.reranker import c_rerank_chunks
from src.generation.prompt_builder import build_final_prompt
from src.generation.answer_generator import generate_answer
from src.fallback.confidence import compute_final_confidence
from src.fallback.fallback_policy import apply_fallback_policy
from src.utils.env import get_driver
from src.utils.text import safe_text, make_json_safe


def save_qa_outputs(result: Dict[str, Any], output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_result = make_json_safe(result)

    with open(output_dir / "block3_end_to_end_result.json", "w", encoding="utf-8") as f:
        json.dump(safe_result, f, ensure_ascii=False, indent=2)

    with open(output_dir / "block3_final_prompt.txt", "w", encoding="utf-8") as f:
        f.write(result.get("final_prompt", "") or "")

    with open(output_dir / "block3_final_answer.md", "w", encoding="utf-8") as f:
        f.write(result.get("final_answer", "") or "")

    pd.DataFrame(make_json_safe(result.get("vector_search_results", []))).to_excel(
        output_dir / "block3_vector_and_graph_results.xlsx",
        index=False,
    )

    pd.DataFrame(make_json_safe(result.get("b_score_results", []))).to_excel(
        output_dir / "block3_b_score_results.xlsx",
        index=False,
    )

    pd.DataFrame(make_json_safe(result.get("c_rerank_results", []))).to_excel(
        output_dir / "block3_c_rerank_results.xlsx",
        index=False,
    )


def run_end_to_end_graphrag_qa(
    user_question: str,
    raw_top_k: int = 30,
    final_top_k: int = 10,
    b_keep_threshold: float = 60.0,
    c_context_threshold: float = 60.0,
    max_tokens: int = 2048,
    save_outputs: bool = True,
    enable_llm: bool = True,
) -> Dict[str, Any]:
    user_question = safe_text(user_question)

    if not user_question:
        raise ValueError("user_question 不能为空。")

    driver = get_driver()

    question_analysis = analyze_user_question(user_question)

    query_embedding = encode_query(user_question)

    vector_results = vector_search_chunks(
        driver=driver,
        database=NEO4J_DATABASE,
        index_name=VECTOR_INDEX_NAME,
        query_embedding=query_embedding,
        top_k=raw_top_k,
    )

    if not vector_results:
        result = {
            "user_question": user_question,
            "question_analysis": question_analysis,
            "vector_search_results": [],
            "graph_expansion_results": {},
            "b_score_results": [],
            "c_rerank_results": [],
            "final_prompt": "",
            "final_answer": "根据当前知识库信息，无法检索到与该问题相关的知识片段，因此无法给出可靠判断。",
            "deepseek_usage": None,
            "confidence": 0.0,
            "fallback_triggered": True,
        }
        if save_outputs:
            save_qa_outputs(result)
        return result

    chunk_ids = [item["chunk_id"] for item in vector_results if item.get("chunk_id")]

    graph_expansion_map = graph_expand_chunks(
        driver=driver,
        database=NEO4J_DATABASE,
        chunk_ids=chunk_ids,
        max_neighbors_per_chunk=30,
    )

    retrieved_chunks = merge_vector_and_graph_context(
        vector_results=vector_results,
        graph_expansion_map=graph_expansion_map,
    )

    b_score_results = b_score_chunks(
        user_question=user_question,
        retrieved_chunks=retrieved_chunks,
        question_analysis=question_analysis,
        keep_threshold=b_keep_threshold,
    )[:final_top_k]

    c_rerank_results = c_rerank_chunks(
        user_question=user_question,
        retrieved_chunks=retrieved_chunks,
        b_score_results=b_score_results,
        context_threshold=c_context_threshold,
        final_top_k=final_top_k,
    )

    confidence = compute_final_confidence(
        retrieved_chunks=retrieved_chunks,
        b_score_results=b_score_results,
        c_rerank_results=c_rerank_results,
    )

    fallback_result = apply_fallback_policy(
        confidence=confidence,
        retrieved_chunks=retrieved_chunks,
        c_rerank_results=c_rerank_results,
    )

    final_prompt = build_final_prompt(
        user_question=user_question,
        retrieved_chunks=retrieved_chunks,
        b_score_results=b_score_results,
        c_rerank_results=c_rerank_results,
        final_top_k=final_top_k,
    )

    deepseek_usage = None

    if fallback_result["fallback_triggered"]:
        final_answer = fallback_result["fallback_answer"]
    elif enable_llm:
        llm_result = generate_answer(
            final_prompt=final_prompt,
            max_tokens=max_tokens,
        )
        final_answer = llm_result["answer"]
        deepseek_usage = llm_result.get("usage")
    else:
        final_answer = ""

    result = {
        "user_question": user_question,
        "question_analysis": question_analysis,
        "vector_search_results": retrieved_chunks,
        "graph_expansion_results": graph_expansion_map,
        "b_score_results": b_score_results,
        "c_rerank_results": c_rerank_results,
        "final_prompt": final_prompt,
        "final_answer": final_answer,
        "deepseek_usage": deepseek_usage,
        "confidence": confidence,
        "fallback_triggered": fallback_result["fallback_triggered"],
    }

    if save_outputs:
        save_qa_outputs(result)

    return result