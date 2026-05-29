from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.retrieval.exact_retriever import exact_retrieve_chunks
from src.rerank.scorer import b_score_chunks
from src.rerank.reranker import c_rerank_chunks
from src.rerank.rankgpt_reranker import rankgpt_rerank_chunks
from src.generation.prompt_builder import build_final_prompt
from src.generation.answer_generator import generate_answer
from src.fallback.confidence import compute_final_confidence
from src.fallback.fallback_policy import apply_fallback_policy
from src.utils.env import get_driver
from src.utils.text import safe_text, make_json_safe


FACT_QUERY_INTENT = "fact_query"


# ============================================================
# 1. 输出保存
# ============================================================

def save_qa_outputs(result: Dict[str, Any], output_dir: Path = OUTPUT_DIR) -> None:
    """
    保存端到端 QA 结果。

    兼容旧输出：
    - block3_end_to_end_result.json
    - block3_final_prompt.txt
    - block3_final_answer.md
    - block3_vector_and_graph_results.xlsx
    - block3_b_score_results.xlsx
    - block3_c_rerank_results.xlsx

    新增输出：
    - block3_rankgpt_results.xlsx
    - block3_effective_rerank_results.xlsx
    """

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

    pd.DataFrame(make_json_safe(result.get("rankgpt_results", []))).to_excel(
        output_dir / "block3_rankgpt_results.xlsx",
        index=False,
    )

    pd.DataFrame(make_json_safe(result.get("effective_rerank_results", []))).to_excel(
        output_dir / "block3_effective_rerank_results.xlsx",
        index=False,
    )


# ============================================================
# 2. 小工具函数
# ============================================================

def _get_question_intent(question_analysis: Dict[str, Any]) -> str:
    """
    从 question_analysis 中获取问题意图。

    兼容字段：
    - 问题意图
    - intent
    - question_intent
    """

    for key in ["问题意图", "intent", "question_intent"]:
        value = safe_text(question_analysis.get(key, ""))
        if value:
            return value

    return ""


def _get_question_entities(question_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 question_analysis 中获取实体列表。

    兼容字段：
    - 核心实体
    - entities
    - core_entities
    """

    for key in ["核心实体", "entities", "core_entities"]:
        value = question_analysis.get(key)
        if isinstance(value, list):
            return value

    return []


def _is_fact_query(question_analysis: Dict[str, Any]) -> bool:
    return _get_question_intent(question_analysis) == FACT_QUERY_INTENT


def _extract_chunk_ids_for_graph_expand(
    retrieved_chunks: List[Dict[str, Any]],
) -> List[str]:
    """
    提取可用于图谱扩展的 chunk_id。

    注意：
    fact_query 的 exact_retrieve_chunks 可能返回 synthetic chunk。
    synthetic chunk 是结构化节点拼接出的证据，不一定是 Neo4j 中真实 EvidenceChunk。
    因此只有 graph_expandable 不为 False 且不是 synthetic 的 chunk 才进入 graph_expand_chunks。
    """

    chunk_ids: List[str] = []

    for item in retrieved_chunks:
        if item.get("is_synthetic") is True:
            continue

        if item.get("graph_expandable") is False:
            continue

        chunk_id = safe_text(item.get("chunk_id", ""))

        if chunk_id:
            chunk_ids.append(chunk_id)

    return chunk_ids


def _maybe_graph_expand_chunks(
    driver: Any,
    retrieved_chunks: List[Dict[str, Any]],
    enable_graph_expand: bool = True,
    max_neighbors_per_chunk: int = 30,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    对 retrieved_chunks 做可选图谱扩展。

    非 fact_query：
    - 正常图谱扩展

    fact_query：
    - 默认可以不做图谱扩展
    - 因为 exact retrieval 已经返回结构化证据
    - 如果后续需要更多上下文，可通过 enable_fact_graph_expand=True 开启
    """

    if not retrieved_chunks or not enable_graph_expand:
        return retrieved_chunks, {}

    chunk_ids = _extract_chunk_ids_for_graph_expand(retrieved_chunks)

    if not chunk_ids:
        return retrieved_chunks, {}

    graph_expansion_map = graph_expand_chunks(
        driver=driver,
        database=NEO4J_DATABASE,
        chunk_ids=chunk_ids,
        max_neighbors_per_chunk=max_neighbors_per_chunk,
    )

    merged_chunks = merge_vector_and_graph_context(
        vector_results=retrieved_chunks,
        graph_expansion_map=graph_expansion_map,
    )

    return merged_chunks, graph_expansion_map


def _build_empty_retrieval_result(
    user_question: str,
    question_analysis: Dict[str, Any],
    retrieval_route: str,
    save_outputs: bool,
) -> Dict[str, Any]:
    """
    统一构造无检索结果时的返回。
    """

    result = {
        "user_question": user_question,
        "question_analysis": question_analysis,
        "retrieval_route": retrieval_route,
        "rerank_route": "",
        "vector_search_results": [],
        "graph_expansion_results": {},
        "b_score_results": [],
        "c_rerank_results": [],
        "rankgpt_results": [],
        "effective_rerank_results": [],
        "final_prompt": "",
        "final_answer": "根据当前知识库信息，无法检索到与该问题相关的知识片段，因此无法给出可靠判断。",
        "deepseek_usage": None,
        "confidence": 0.0,
        "fallback_triggered": True,
        "fallback_reason": "no_retrieval_results",
    }

    if save_outputs:
        save_qa_outputs(result)

    return result


def _compute_fact_query_confidence(
    retrieved_chunks: List[Dict[str, Any]],
    effective_rerank_results: List[Dict[str, Any]],
) -> float:
    """
    fact_query 专用置信度。

    原 compute_final_confidence 更适合原始 B/C 水文规则链路。
    fact_query 会跳过 B 评分，如果继续完全依赖原置信度逻辑，可能被误判为低置信度。

    这里使用：
    - effective_rerank_results 的 final_score
    - retrieved_chunks 的 exact_score / vector_score
    进行保守估计。
    """

    if not retrieved_chunks or not effective_rerank_results:
        return 0.0

    top_final_score = 0.0
    for item in effective_rerank_results:
        try:
            top_final_score = max(top_final_score, float(item.get("final_score", 0.0) or 0.0))
        except Exception:
            continue

    top_exact_score = 0.0
    for item in retrieved_chunks:
        try:
            top_exact_score = max(top_exact_score, float(item.get("exact_score", 0.0) or 0.0))
        except Exception:
            continue

    top_vector_score = 0.0
    for item in retrieved_chunks:
        try:
            value = float(item.get("vector_score", 0.0) or 0.0)
            if value <= 1.0:
                value = value * 100.0
            top_vector_score = max(top_vector_score, value)
        except Exception:
            continue

    confidence = max(
        top_final_score,
        0.8 * top_exact_score,
        0.7 * top_vector_score,
    )

    return round(min(max(confidence, 0.0), 100.0), 2)


def _run_rankgpt_if_needed(
    user_question: str,
    retrieved_chunks: List[Dict[str, Any]],
    c_rerank_results: List[Dict[str, Any]],
    question_analysis: Dict[str, Any],
    final_top_k: int,
    enable_rankgpt: bool,
    rankgpt_candidate_top_k: int,
    rankgpt_fail_open: bool,
) -> List[Dict[str, Any]]:
    """
    在规则重排之后调用 RankGPT。

    位置：
    - Fact Rule Rerank / C Rule Rerank
    - RankGPT
    - Top10 Context

    如果 RankGPT 关闭或失败，会保留规则重排顺序。
    """

    if not c_rerank_results:
        return []

    if not enable_rankgpt:
        return []

    return rankgpt_rerank_chunks(
        user_question=user_question,
        reranked_results=c_rerank_results,
        retrieved_chunks=retrieved_chunks,
        question_analysis=question_analysis,
        candidate_top_k=rankgpt_candidate_top_k,
        final_top_k=final_top_k,
        enabled=enable_rankgpt,
        fail_open=rankgpt_fail_open,
    )


# ============================================================
# 3. fact_query 新链路
# ============================================================

def _run_fact_query_chain(
    user_question: str,
    question_analysis: Dict[str, Any],
    driver: Any,
    raw_top_k: int,
    final_top_k: int,
    c_context_threshold: float,
    enable_rankgpt: bool,
    rankgpt_candidate_top_k: int,
    rankgpt_fail_open: bool,
    enable_fact_graph_expand: bool,
) -> Dict[str, Any]:
    """
    fact_query 专用链路。

    新链路：
    1. Query Translation 已在外层完成 analyze_user_question()
    2. Exact Retrieval：exact_retrieve_chunks()
    3. 可选 Graph Expand
    4. Fact Rule Rerank：c_rerank_chunks() 内部按 intent 路由到 fact_scorer
    5. RankGPT Rerank：rankgpt_rerank_chunks()
    6. Top10 Context 使用 RankGPT 后的结果

    注意：
    - 跳过 encode_query()
    - 跳过 vector_search_chunks()
    - 跳过 b_score_chunks()
    - 不影响原水文规则链路
    """

    entities = _get_question_entities(question_analysis)

    retrieved_chunks = exact_retrieve_chunks(
        user_question=user_question,
        entities=entities,
        top_k=max(raw_top_k, final_top_k),
    )

    retrieved_chunks, graph_expansion_map = _maybe_graph_expand_chunks(
        driver=driver,
        retrieved_chunks=retrieved_chunks,
        enable_graph_expand=enable_fact_graph_expand,
        max_neighbors_per_chunk=30,
    )

    b_score_results: List[Dict[str, Any]] = []

    c_rerank_results = c_rerank_chunks(
        user_question=user_question,
        retrieved_chunks=retrieved_chunks,
        b_score_results=b_score_results,
        context_threshold=c_context_threshold,
        final_top_k=final_top_k,
        question_analysis=question_analysis,
        intent=FACT_QUERY_INTENT,
        entities=entities,
    )

    rankgpt_results = _run_rankgpt_if_needed(
        user_question=user_question,
        retrieved_chunks=retrieved_chunks,
        c_rerank_results=c_rerank_results,
        question_analysis=question_analysis,
        final_top_k=final_top_k,
        enable_rankgpt=enable_rankgpt,
        rankgpt_candidate_top_k=rankgpt_candidate_top_k,
        rankgpt_fail_open=rankgpt_fail_open,
    )

    effective_rerank_results = rankgpt_results if rankgpt_results else c_rerank_results

    confidence = _compute_fact_query_confidence(
        retrieved_chunks=retrieved_chunks,
        effective_rerank_results=effective_rerank_results,
    )

    return {
        "retrieval_route": "fact_exact_retrieval",
        "rerank_route": "fact_rule",
        "retrieved_chunks": retrieved_chunks,
        "graph_expansion_map": graph_expansion_map,
        "b_score_results": b_score_results,
        "c_rerank_results": c_rerank_results,
        "rankgpt_results": rankgpt_results,
        "effective_rerank_results": effective_rerank_results,
        "confidence": confidence,
    }


# ============================================================
# 4. 原水文 GraphRAG 链路
# ============================================================

def _run_hydro_graphrag_chain(
    user_question: str,
    question_analysis: Dict[str, Any],
    driver: Any,
    raw_top_k: int,
    final_top_k: int,
    b_keep_threshold: float,
    c_context_threshold: float,
    enable_rankgpt: bool,
    rankgpt_candidate_top_k: int,
    rankgpt_fail_open: bool,
) -> Dict[str, Any]:
    """
    原有非 fact_query GraphRAG 链路。

    保持原运行逻辑：
    1. encode_query()
    2. vector_search_chunks()
    3. graph_expand_chunks()
    4. merge_vector_and_graph_context()
    5. b_score_chunks()
    6. c_rerank_chunks()

    新增：
    7. RankGPT Rerank，作为规则重排后的语义排序层。
    """

    query_embedding = encode_query(user_question)

    vector_results = vector_search_chunks(
        driver=driver,
        database=NEO4J_DATABASE,
        index_name=VECTOR_INDEX_NAME,
        query_embedding=query_embedding,
        top_k=raw_top_k,
    )

    if not vector_results:
        return {
            "retrieval_route": "vector_graph_retrieval",
            "rerank_route": "hydro_rule",
            "retrieved_chunks": [],
            "graph_expansion_map": {},
            "b_score_results": [],
            "c_rerank_results": [],
            "rankgpt_results": [],
            "effective_rerank_results": [],
            "confidence": 0.0,
        }

    chunk_ids = [
        item["chunk_id"]
        for item in vector_results
        if item.get("chunk_id")
    ]

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
        question_analysis=question_analysis,
    )

    rankgpt_results = _run_rankgpt_if_needed(
        user_question=user_question,
        retrieved_chunks=retrieved_chunks,
        c_rerank_results=c_rerank_results,
        question_analysis=question_analysis,
        final_top_k=final_top_k,
        enable_rankgpt=enable_rankgpt,
        rankgpt_candidate_top_k=rankgpt_candidate_top_k,
        rankgpt_fail_open=rankgpt_fail_open,
    )

    effective_rerank_results = rankgpt_results if rankgpt_results else c_rerank_results

    confidence = compute_final_confidence(
        retrieved_chunks=retrieved_chunks,
        b_score_results=b_score_results,
        c_rerank_results=effective_rerank_results,
    )

    return {
        "retrieval_route": "vector_graph_retrieval",
        "rerank_route": "hydro_rule",
        "retrieved_chunks": retrieved_chunks,
        "graph_expansion_map": graph_expansion_map,
        "b_score_results": b_score_results,
        "c_rerank_results": c_rerank_results,
        "rankgpt_results": rankgpt_results,
        "effective_rerank_results": effective_rerank_results,
        "confidence": confidence,
    }


# ============================================================
# 5. 端到端 QA 主入口
# ============================================================

def run_end_to_end_graphrag_qa(
    user_question: str,
    raw_top_k: int = 30,
    final_top_k: int = 10,
    b_keep_threshold: float = 60.0,
    c_context_threshold: float = 60.0,
    max_tokens: int = 2048,
    save_outputs: bool = True,
    enable_llm: bool = True,
    enable_rankgpt: bool = False,
    rankgpt_candidate_top_k: int = 10,
    rankgpt_fail_open: bool = True,
    enable_fact_graph_expand: bool = False,
) -> Dict[str, Any]:
    """
    在线 GraphRAG QA 主函数。

    增强后的路由：

    A. fact_query：
        analyze_user_question
        -> exact_retrieve_chunks
        -> c_rerank_chunks 内部路由到 fact_scorer
        -> rankgpt_rerank_chunks
        -> build_final_prompt
        -> generate_answer

    B. 非 fact_query：
        analyze_user_question
        -> encode_query
        -> vector_search_chunks
        -> graph_expand_chunks
        -> merge_vector_and_graph_context
        -> b_score_chunks
        -> c_rerank_chunks 原水文规则重排
        -> rankgpt_rerank_chunks
        -> build_final_prompt
        -> generate_answer

    不影响原链路：
    - 非 fact_query 仍然使用原有向量检索、图谱扩展、B 评分、C 重排。
    - RankGPT 失败时 fail_open，保留规则重排顺序。
    """

    user_question = safe_text(user_question)

    if not user_question:
        raise ValueError("user_question 不能为空。")

    driver = get_driver()

    # 1. Query Translation
    question_analysis = analyze_user_question(user_question)
    question_intent = _get_question_intent(question_analysis)

    # 2. RankGPT 是否启用
    # 注意：RankGPT 本身调用 LLM。
    # 如果 enable_llm=False，通常用于离线检索/重排测试，这里也默认不调用 RankGPT。
    effective_enable_rankgpt = bool(enable_rankgpt and enable_llm)

    # 3. Route：fact_query 走新链路，其他问题走原链路
    if _is_fact_query(question_analysis):
        chain_result = _run_fact_query_chain(
            user_question=user_question,
            question_analysis=question_analysis,
            driver=driver,
            raw_top_k=raw_top_k,
            final_top_k=final_top_k,
            c_context_threshold=c_context_threshold,
            enable_rankgpt=effective_enable_rankgpt,
            rankgpt_candidate_top_k=rankgpt_candidate_top_k,
            rankgpt_fail_open=rankgpt_fail_open,
            enable_fact_graph_expand=enable_fact_graph_expand,
        )
    else:
        chain_result = _run_hydro_graphrag_chain(
            user_question=user_question,
            question_analysis=question_analysis,
            driver=driver,
            raw_top_k=raw_top_k,
            final_top_k=final_top_k,
            b_keep_threshold=b_keep_threshold,
            c_context_threshold=c_context_threshold,
            enable_rankgpt=effective_enable_rankgpt,
            rankgpt_candidate_top_k=rankgpt_candidate_top_k,
            rankgpt_fail_open=rankgpt_fail_open,
        )

    retrieved_chunks = chain_result["retrieved_chunks"]
    graph_expansion_map = chain_result["graph_expansion_map"]
    b_score_results = chain_result["b_score_results"]
    c_rerank_results = chain_result["c_rerank_results"]
    rankgpt_results = chain_result["rankgpt_results"]
    effective_rerank_results = chain_result["effective_rerank_results"]
    confidence = chain_result["confidence"]

    # 4. 无检索结果兜底
    if not retrieved_chunks:
        return _build_empty_retrieval_result(
            user_question=user_question,
            question_analysis=question_analysis,
            retrieval_route=chain_result["retrieval_route"],
            save_outputs=save_outputs,
        )

    # 5. Fallback
    # 注意：这里使用 effective_rerank_results。
    # 如果 RankGPT 成功，则用 RankGPT 后的顺序作为最终上下文依据。
    # 如果 RankGPT 未启用或失败，则 effective_rerank_results 会退回 C/Fact Rule 结果。
    fallback_result = apply_fallback_policy(
        confidence=confidence,
        retrieved_chunks=retrieved_chunks,
        c_rerank_results=effective_rerank_results,
    )

    # 6. Final Prompt
    # 注意：为了不改 prompt_builder.py，这里把 effective_rerank_results
    # 作为 c_rerank_results 传入。
    final_prompt = build_final_prompt(
        user_question=user_question,
        retrieved_chunks=retrieved_chunks,
        b_score_results=b_score_results,
        c_rerank_results=effective_rerank_results,
        final_top_k=final_top_k,
    )

    # 7. Final Answer
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

    # 8. 组装结果
    result = {
        "user_question": user_question,
        "question_analysis": question_analysis,
        "question_intent": question_intent,
        "retrieval_route": chain_result["retrieval_route"],
        "rerank_route": chain_result["rerank_route"],

        # 为兼容旧前端字段，仍然叫 vector_search_results。
        # fact_query 时这里实际是 exact retrieval 结果。
        "vector_search_results": retrieved_chunks,

        "graph_expansion_results": graph_expansion_map,
        "b_score_results": b_score_results,

        # 规则重排结果：
        # - fact_query: fact rule rerank
        # - 非 fact_query: 原 C hydro rerank
        "c_rerank_results": c_rerank_results,

        # RankGPT 结果
        "rankgpt_results": rankgpt_results,

        # 最终用于 Top10 Context 的重排结果
        "effective_rerank_results": effective_rerank_results,

        "final_prompt": final_prompt,
        "final_answer": final_answer,
        "deepseek_usage": deepseek_usage,
        "confidence": confidence,
        "fallback_triggered": fallback_result["fallback_triggered"],
        "fallback_answer": fallback_result.get("fallback_answer", ""),
        "rankgpt_enabled": effective_enable_rankgpt,
        "enable_fact_graph_expand": enable_fact_graph_expand,
    }

    if save_outputs:
        save_qa_outputs(result)

    return result
