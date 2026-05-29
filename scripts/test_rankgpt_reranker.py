from __future__ import annotations

import json
import sys
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List


# ============================================================
# 1. 加入项目根目录
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ============================================================
# 2. 加载 .env
# ============================================================

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_DIR / ".env")
except Exception:
    pass


# ============================================================
# 3. 导入待测试模块
# ============================================================

from src.rerank.rankgpt_reranker import (
    build_rankgpt_candidates,
    build_rankgpt_messages,
    validate_rankgpt_ids,
    apply_rankgpt_order,
    fallback_rankgpt_order,
    rankgpt_rerank_chunks,
)


# ============================================================
# 4. Mock 数据
# ============================================================

MOCK_RETRIEVED_CHUNKS: List[Dict[str, Any]] = [
    {
        "chunk_id": "EvidenceChunk:Layer|CHGC001_1",
        "chunk_type": "lithology_layer_fact",
        "chunk_text": "钻孔 CHGC001 的分层 CHGC001_1 位于 0.0m 至 2.8m，岩性大类为砂土，岩性小类为细砂。",
        "source_label": "LithologyLayer",
        "source_id": "CHGC001_1",
        "vector_score": 0.96,
        "retrieval_method": "exact",
        "exact_match_type": "layer_id",
        "exact_match_value": "CHGC001_1",
    },
    {
        "chunk_id": "EvidenceChunk:Layer|CHGC001_2",
        "chunk_type": "lithology_layer_fact",
        "chunk_text": "钻孔 CHGC001 的分层 CHGC001_2 位于 2.8m 至 6.2m，岩性大类为黏性土，岩性小类为粉质黏土。",
        "source_label": "LithologyLayer",
        "source_id": "CHGC001_2",
        "vector_score": 0.82,
        "retrieval_method": "exact",
        "exact_match_type": "layer_id",
        "exact_match_value": "CHGC001_2",
    },
    {
        "chunk_id": "EvidenceChunk:Borehole|CHGC001",
        "chunk_type": "borehole_fact",
        "chunk_text": "钻孔 CHGC001 包含多个分层，包括 CHGC001_1、CHGC001_2、CHGC001_3。",
        "source_label": "Borehole",
        "source_id": "CHGC001",
        "vector_score": 0.88,
        "retrieval_method": "exact",
        "exact_match_type": "borehole_id",
        "exact_match_value": "CHGC001",
    },
    {
        "chunk_id": "EvidenceChunk:Noise|Clay",
        "chunk_type": "semantic_noise",
        "chunk_text": "粉质黏土一般具有较低渗透性，和黏粒含量、孔隙结构、压实程度有关。",
        "source_label": "EvidenceChunk",
        "source_id": "unknown",
        "vector_score": 0.99,
        "retrieval_method": "vector",
    },
]


# 模拟规则重排后的结果。
# 注意这里故意把 CHGC001_1 排在 CHGC001_2 前面，
# 用于测试 RankGPT 是否能根据问题把 CHGC001_2 调到第一。
MOCK_RULE_RERANK_RESULTS: List[Dict[str, Any]] = [
    {
        "rank": 1,
        "chunk_id": "EvidenceChunk:Layer|CHGC001_1",
        "final_score": 98.0,
        "semantic_score": 4.8,
        "rule_score": 0.0,
        "evidence_score": 98.0,
        "matched_fields": ["borehole_id", "lithology", "semantic"],
        "rerank_route": "fact_query",
        "scoring_method": "fact_query_rule_score",
        "rerank_reason": "同钻孔 CHGC001，包含岩性字段，但不是目标分层 CHGC001_2。",
        "use_for_context": True,
    },
    {
        "rank": 2,
        "chunk_id": "EvidenceChunk:Layer|CHGC001_2",
        "final_score": 92.0,
        "semantic_score": 4.1,
        "rule_score": 0.0,
        "evidence_score": 92.0,
        "matched_fields": ["layer_id", "borehole_id", "lithology", "semantic"],
        "rerank_route": "fact_query",
        "scoring_method": "fact_query_rule_score",
        "rerank_reason": "精确命中目标分层 CHGC001_2，并包含深度范围和岩性字段。",
        "use_for_context": True,
    },
    {
        "rank": 3,
        "chunk_id": "EvidenceChunk:Borehole|CHGC001",
        "final_score": 70.0,
        "semantic_score": 4.4,
        "rule_score": 0.0,
        "evidence_score": 70.0,
        "matched_fields": ["borehole_id", "semantic"],
        "rerank_route": "fact_query",
        "scoring_method": "fact_query_rule_score",
        "rerank_reason": "命中钻孔 CHGC001，但只是钻孔概览，不是具体分层证据。",
        "use_for_context": True,
    },
    {
        "rank": 4,
        "chunk_id": "EvidenceChunk:Noise|Clay",
        "final_score": 35.0,
        "semantic_score": 5.0,
        "rule_score": 0.0,
        "evidence_score": 35.0,
        "matched_fields": ["semantic"],
        "rerank_route": "fact_query",
        "scoring_method": "fact_query_rule_score",
        "rerank_reason": "语义上提到粉质黏土，但没有目标钻孔或目标分层。",
        "use_for_context": False,
    },
]


MOCK_QUESTION = "钻孔 CHGC001 的分层 CHGC001_2 位于哪个深度范围？岩性大类和岩性小类分别是什么？"

MOCK_QUESTION_ANALYSIS: Dict[str, Any] = {
    "问题意图": "fact_query",
    "核心实体": [
        {
            "实体类型": "层位",
            "实体值": "CHGC001_2",
            "来源": "Regex.layer_id",
            "字段名": "layer_id",
        },
        {
            "实体类型": "钻孔",
            "实体值": "CHGC001",
            "来源": "Regex.borehole_id",
            "字段名": "borehole_id",
        },
    ],
    "检索关键词": ["CHGC001_2", "CHGC001", "深度范围", "岩性大类", "岩性小类"],
}


# ============================================================
# 5. 打印工具
# ============================================================

def print_title(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def print_ranked_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("没有排序结果。")
        return

    for item in results:
        print("\n" + "-" * 120)
        print(f"rank: {item.get('rank')}")
        print(f"rankgpt_rank: {item.get('rankgpt_rank')}")
        print(f"chunk_id: {item.get('chunk_id')}")
        print(f"source_id: {item.get('source_id')}")
        print(f"final_score: {item.get('final_score')}")
        print(f"rankgpt_used: {item.get('rankgpt_used')}")
        print(f"rank_before_rankgpt: {item.get('rank_before_rankgpt')}")
        print(f"score_before_rankgpt: {item.get('score_before_rankgpt')}")
        print(f"rankgpt_reason: {item.get('rankgpt_reason')}")
        print(f"chunk_text: {str(item.get('chunk_text', ''))[:180]}")


# ============================================================
# 6. 测试 1：构造 RankGPT candidates，不调用 LLM
# ============================================================

def test_build_rankgpt_candidates() -> None:
    print_title("测试 1：build_rankgpt_candidates，不调用 LLM")

    candidates = build_rankgpt_candidates(
        reranked_results=MOCK_RULE_RERANK_RESULTS,
        retrieved_chunks=MOCK_RETRIEVED_CHUNKS,
        candidate_top_k=10,
        max_chunk_chars=300,
    )

    print("候选数量:", len(candidates))
    pprint(candidates, width=140)

    assert len(candidates) == 4
    assert candidates[0]["chunk_id"] == "EvidenceChunk:Layer|CHGC001_1"
    assert "chunk_text" in candidates[0]

    print("\nPASS：候选构造正常。")


# ============================================================
# 7. 测试 2：构造 messages，不调用 LLM
# ============================================================

def test_build_rankgpt_messages() -> None:
    print_title("测试 2：build_rankgpt_messages，不调用 LLM")

    candidates = build_rankgpt_candidates(
        reranked_results=MOCK_RULE_RERANK_RESULTS,
        retrieved_chunks=MOCK_RETRIEVED_CHUNKS,
        candidate_top_k=10,
        max_chunk_chars=300,
    )

    messages = build_rankgpt_messages(
        user_question=MOCK_QUESTION,
        candidates=candidates,
        question_analysis=MOCK_QUESTION_ANALYSIS,
    )

    print("messages 数量:", len(messages))
    pprint(messages, width=140)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "CHGC001_2" in messages[1]["content"]

    print("\nPASS：RankGPT messages 构造正常。")


# ============================================================
# 8. 测试 3：校验 RankGPT 返回 ID，不调用 LLM
# ============================================================

def test_validate_rankgpt_ids() -> None:
    print_title("测试 3：validate_rankgpt_ids，不调用 LLM")

    candidates = build_rankgpt_candidates(
        reranked_results=MOCK_RULE_RERANK_RESULTS,
        retrieved_chunks=MOCK_RETRIEVED_CHUNKS,
        candidate_top_k=10,
        max_chunk_chars=300,
    )

    ranked_chunk_ids = [
        "EvidenceChunk:Layer|CHGC001_2",
        "EvidenceChunk:Layer|CHGC001_1",
        "not_exists_chunk_id",
    ]

    valid_ids, invalid_ids = validate_rankgpt_ids(
        ranked_chunk_ids=ranked_chunk_ids,
        candidates=candidates,
    )

    print("valid_ids:")
    pprint(valid_ids, width=120)

    print("invalid_ids:")
    pprint(invalid_ids, width=120)

    assert valid_ids == [
        "EvidenceChunk:Layer|CHGC001_2",
        "EvidenceChunk:Layer|CHGC001_1",
    ]
    assert invalid_ids == ["not_exists_chunk_id"]

    print("\nPASS：RankGPT ID 校验正常。")


# ============================================================
# 9. 测试 4：模拟 RankGPT 输出并应用排序，不调用 LLM
# ============================================================

def test_apply_rankgpt_order_with_mock_output() -> None:
    print_title("测试 4：apply_rankgpt_order，模拟 RankGPT 输出，不调用 LLM")

    candidates = build_rankgpt_candidates(
        reranked_results=MOCK_RULE_RERANK_RESULTS,
        retrieved_chunks=MOCK_RETRIEVED_CHUNKS,
        candidate_top_k=10,
        max_chunk_chars=300,
    )

    mock_rankgpt_output = {
        "ranked_chunk_ids": [
            "EvidenceChunk:Layer|CHGC001_2",
            "EvidenceChunk:Layer|CHGC001_1",
            "EvidenceChunk:Borehole|CHGC001",
            "EvidenceChunk:Noise|Clay",
        ],
        "reason": "问题明确询问 CHGC001_2，因此包含该分层 ID、深度范围和岩性字段的 chunk 应排第一。",
        "raw_response": json.dumps(
            {
                "ranked_chunk_ids": [
                    "EvidenceChunk:Layer|CHGC001_2",
                    "EvidenceChunk:Layer|CHGC001_1",
                    "EvidenceChunk:Borehole|CHGC001",
                    "EvidenceChunk:Noise|Clay",
                ],
                "reason": "问题明确询问 CHGC001_2，因此包含该分层 ID 的 chunk 应排第一。",
            },
            ensure_ascii=False,
        ),
        "success": True,
    }

    results = apply_rankgpt_order(
        reranked_results=MOCK_RULE_RERANK_RESULTS,
        candidates=candidates,
        rankgpt_output=mock_rankgpt_output,
        retrieved_chunks=MOCK_RETRIEVED_CHUNKS,
        top_k=10,
    )

    print_ranked_results(results)

    assert results[0]["chunk_id"] == "EvidenceChunk:Layer|CHGC001_2"
    assert results[0]["rankgpt_used"] is True
    assert results[0]["rank_before_rankgpt"] == 2

    print("\nPASS：模拟 RankGPT 输出后，CHGC001_2 被排到第一。")


# ============================================================
# 10. 测试 5：Fallback，不调用 LLM
# ============================================================

def test_fallback_rankgpt_order() -> None:
    print_title("测试 5：fallback_rankgpt_order，不调用 LLM")

    results = fallback_rankgpt_order(
        reranked_results=MOCK_RULE_RERANK_RESULTS,
        retrieved_chunks=MOCK_RETRIEVED_CHUNKS,
        reason="测试 fallback，保留规则重排顺序。",
        top_k=10,
    )

    print_ranked_results(results)

    assert results[0]["chunk_id"] == "EvidenceChunk:Layer|CHGC001_1"
    assert results[0]["rankgpt_used"] is False

    print("\nPASS：Fallback 保留原规则排序。")


# ============================================================
# 11. 测试 6：rankgpt_rerank_chunks enabled=False，不调用 LLM
# ============================================================

def test_rankgpt_disabled() -> None:
    print_title("测试 6：rankgpt_rerank_chunks enabled=False，不调用 LLM")

    results = rankgpt_rerank_chunks(
        user_question=MOCK_QUESTION,
        reranked_results=MOCK_RULE_RERANK_RESULTS,
        retrieved_chunks=MOCK_RETRIEVED_CHUNKS,
        question_analysis=MOCK_QUESTION_ANALYSIS,
        candidate_top_k=10,
        final_top_k=10,
        enabled=False,
    )

    print_ranked_results(results)

    assert results[0]["chunk_id"] == "EvidenceChunk:Layer|CHGC001_1"
    assert results[0]["rankgpt_used"] is False

    print("\nPASS：RankGPT disabled 时保留原规则排序。")


# ============================================================
# 12. 测试 7：真实调用 LLM，可选
# ============================================================

def test_rankgpt_with_real_llm() -> None:
    """
    真实调用 DeepSeek / OpenAI-compatible LLM。

    前提：
    1. .env 中 DEEPSEEK_API_KEY 正确
    2. DEEPSEEK_BASE_URL 正确
    3. DEEPSEEK_MODEL 正确
    4. src.utils.env.get_deepseek_client() 可正常创建 client

    如果你暂时不想消耗 API，可以不在 main() 中调用这个函数。
    """

    print_title("测试 7：rankgpt_rerank_chunks 真实调用 LLM")

    results = rankgpt_rerank_chunks(
        user_question=MOCK_QUESTION,
        reranked_results=MOCK_RULE_RERANK_RESULTS,
        retrieved_chunks=MOCK_RETRIEVED_CHUNKS,
        question_analysis=MOCK_QUESTION_ANALYSIS,
        candidate_top_k=10,
        final_top_k=10,
        enabled=True,
        fail_open=True,
    )

    print_ranked_results(results)

    if results:
        print("\nTop1:", results[0].get("chunk_id"))
        print("rankgpt_used:", results[0].get("rankgpt_used"))
        print("rankgpt_reason:", results[0].get("rankgpt_reason"))

    # 注意：真实 LLM 有不确定性，这里不做强 assert。
    # 只建议人工确认 Top1 是否为 EvidenceChunk:Layer|CHGC001_2。


# ============================================================
# 13. main
# ============================================================

def main() -> None:
    test_build_rankgpt_candidates()
    test_build_rankgpt_messages()
    test_validate_rankgpt_ids()
    test_apply_rankgpt_order_with_mock_output()
    test_fallback_rankgpt_order()
    test_rankgpt_disabled()

    # 默认不调用真实 LLM，避免每次测试都消耗 API。
    # 如果要测试真实 RankGPT，取消下面这一行注释。
    # test_rankgpt_with_real_llm()


if __name__ == "__main__":
    main()