from __future__ import annotations

import json
from typing import Any, Dict, List

from src.utils.text import truncate_text


FINAL_ANSWER_SYSTEM_PROMPT = """
你是一名水文地质、地下水渗流、知识图谱和 GraphRAG 问答专家。
你必须严格基于提供的检索证据、图谱上下文、B 评分结果和 C 水文规则重排序结果回答。
不要编造不存在的钻孔、层位、渗透率数值、单位、来源或观测结论。

必须区分：
1. 直接证据，例如 k_value、k_log10、k_unit、source、reliability；
2. 规则推断，例如岩性、粒径、黏粒含量、胶结、裂隙、埋深、夹层等推断；
3. 不确定性，例如证据不足、正负因素并存、缺少实测值。

回答结构建议：
【结论】
【依据】
【水文逻辑解释】
【补充说明】
""".strip()


END_TO_END_PROMPT = """
一、用户原始问题
{user_question}

二、Neo4j 向量检索结果
{vector_search_results}

三、图谱扩展结果
{graph_expansion_results}

四、B 综合相关性评分结果
{b_score_results}

五、C 水文规则重排序结果
{c_rerank_results}

六、回答生成要求
1. 只能基于上述上下文回答。
2. 不得编造渗透率数值、钻孔编号、层位编号。
3. 如果有 k_value、k_log10、k_unit、source、reliability，要优先作为直接证据。
4. 如果没有直接渗透率记录，要明确说明属于规则推断。
5. 如果证据不足或正负因素并存，要明确说明不确定性。
6. 用中文回答，面向水文地质/工程地质用户，表达清晰。
""".strip()


def format_vector_search_results(
    retrieved_chunks: List[Dict[str, Any]],
    top_n: int = 10,
) -> str:
    rows = []

    for idx, item in enumerate(retrieved_chunks[:top_n], start=1):
        rows.append(
            {
                "rank": idx,
                "chunk_id": item.get("chunk_id"),
                "chunk_type": item.get("chunk_type"),
                "source_label": item.get("source_label"),
                "source_id": item.get("source_id"),
                "vector_score": round(float(item.get("vector_score", 0.0) or 0.0), 4),
                "chunk_text": truncate_text(item.get("chunk_text"), 800),
            }
        )

    return json.dumps(rows, ensure_ascii=False, indent=2)


def format_graph_expansion_results(
    retrieved_chunks: List[Dict[str, Any]],
    top_n: int = 10,
) -> str:
    rows = []

    for idx, item in enumerate(retrieved_chunks[:top_n], start=1):
        rows.append(
            {
                "rank": idx,
                "chunk_id": item.get("chunk_id"),
                "chunk_type": item.get("chunk_type"),
                "source_label": item.get("source_label"),
                "source_id": item.get("source_id"),
                "graph_context_text": truncate_text(item.get("graph_context_text"), 1200),
            }
        )

    return json.dumps(rows, ensure_ascii=False, indent=2)


def format_b_score_results(
    b_score_results: List[Dict[str, Any]],
    top_n: int = 10,
) -> str:
    return json.dumps(
        b_score_results[:top_n],
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def format_c_rerank_results(
    c_rerank_results: List[Dict[str, Any]],
    top_n: int = 10,
) -> str:
    return json.dumps(
        c_rerank_results[:top_n],
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def build_final_prompt(
    user_question: str,
    retrieved_chunks: List[Dict[str, Any]],
    b_score_results: List[Dict[str, Any]],
    c_rerank_results: List[Dict[str, Any]],
    final_top_k: int = 10,
) -> str:
    vector_search_results_text = format_vector_search_results(
        retrieved_chunks=retrieved_chunks,
        top_n=final_top_k,
    )

    graph_expansion_results_text = format_graph_expansion_results(
        retrieved_chunks=retrieved_chunks,
        top_n=final_top_k,
    )

    b_score_results_text = format_b_score_results(
        b_score_results=b_score_results,
        top_n=final_top_k,
    )

    c_rerank_results_text = format_c_rerank_results(
        c_rerank_results=c_rerank_results,
        top_n=final_top_k,
    )

    return END_TO_END_PROMPT.format(
        user_question=user_question,
        vector_search_results=vector_search_results_text,
        graph_expansion_results=graph_expansion_results_text,
        b_score_results=b_score_results_text,
        c_rerank_results=c_rerank_results_text,
    )