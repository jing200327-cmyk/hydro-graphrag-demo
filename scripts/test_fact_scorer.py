from __future__ import annotations

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

from src.query.intent_classifier import classify_question_intent
from src.query.entity_extractor import extract_core_entities
from src.retrieval.exact_retriever import exact_retrieve_chunks
from src.rerank.fact_scorer import (
    collect_fact_query_constraints,
    score_single_fact_chunk,
    fact_score_chunks,
    fact_rerank_chunks,
)


# ============================================================
# 4. Mock 数据：不依赖 Neo4j，用于先验证 fact_scorer 逻辑
# ============================================================

MOCK_CHUNKS: List[Dict[str, Any]] = [
    {
        "chunk_id": "mock_chunk_001",
        "chunk_type": "lithology_layer_fact",
        "chunk_text": "CHGC001_2 层位记录：岩性大类为黏性土，岩性小类为粉质黏土，层顶深度 12.0m，层底深度 18.0m。",
        "source_label": "LithologyLayer",
        "source_id": "CHGC001_2",
        "vector_score": 0.72,
        "retrieval_method": "exact",
        "exact_match_type": "layer_id",
        "exact_match_value": "CHGC001_2",
        "exact_score": 100.0,
        "source_props": {
            "layer_id": "CHGC001_2",
            "borehole_id": "CHGC001",
            "lithology_major_v2": "黏性土",
            "lithology_minor_v3": "粉质黏土",
            "top_depth": 12.0,
            "bottom_depth": 18.0,
            "thickness": 6.0,
        },
        "is_synthetic": False,
    },
    {
        "chunk_id": "mock_chunk_002",
        "chunk_type": "lithology_layer_fact",
        "chunk_text": "CHGC001_1 层位记录：岩性小类为细砂，层顶深度 0.0m，层底深度 12.0m。",
        "source_label": "LithologyLayer",
        "source_id": "CHGC001_1",
        "vector_score": 0.95,
        "retrieval_method": "exact",
        "exact_match_type": "layer_id",
        "exact_match_value": "CHGC001_1",
        "exact_score": 100.0,
        "source_props": {
            "layer_id": "CHGC001_1",
            "borehole_id": "CHGC001",
            "lithology_major_v2": "砂土",
            "lithology_minor_v3": "细砂",
            "top_depth": 0.0,
            "bottom_depth": 12.0,
            "thickness": 12.0,
        },
        "is_synthetic": False,
    },
    {
        "chunk_id": "mock_chunk_003",
        "chunk_type": "borehole_fact",
        "chunk_text": "CHGC001 钻孔包含多个层位，包括 CHGC001_1 和 CHGC001_2。",
        "source_label": "Borehole",
        "source_id": "CHGC001",
        "vector_score": 0.88,
        "retrieval_method": "exact",
        "exact_match_type": "borehole_id",
        "exact_match_value": "CHGC001",
        "exact_score": 85.0,
        "source_props": {
            "borehole_id": "CHGC001",
        },
        "is_synthetic": False,
    },
    {
        "chunk_id": "mock_chunk_004",
        "chunk_type": "semantic_noise",
        "chunk_text": "该段文本讨论粉质黏土的工程性质，但没有明确层位编号。",
        "source_label": "EvidenceChunk",
        "source_id": "unknown",
        "vector_score": 0.99,
        "retrieval_method": "vector",
        "source_props": {},
        "is_synthetic": False,
    },
]


TEST_QUESTIONS = [
    "CHGC001_2 的岩性是什么？",
    "CHGC001_1 的层顶和层底分别是多少？",
    "CHGC001 在 12m 到 18m 的岩性是什么？",
    "chgc001 _ 2 的岩性小类是什么？",
]


# ============================================================
# 5. 打印工具
# ============================================================

def print_scored_chunk_summary(chunks: List[Dict[str, Any]], max_text_len: int = 180) -> None:
    if not chunks:
        print("没有评分结果。")
        return

    print(f"共 {len(chunks)} 条评分结果：")

    for chunk in chunks:
        chunk_text = str(chunk.get("chunk_text", "")).replace("\n", " ")

        if len(chunk_text) > max_text_len:
            chunk_text = chunk_text[:max_text_len] + "..."

        print("\n" + "-" * 120)
        print(f"rank: {chunk.get('rank')}")
        print(f"chunk_id: {chunk.get('chunk_id')}")
        print(f"source_id: {chunk.get('source_id')}")
        print(f"exact_match_type: {chunk.get('exact_match_type')}")
        print(f"exact_match_value: {chunk.get('exact_match_value')}")
        print(f"final_score: {chunk.get('final_score')}")
        print(f"fact_raw_score: {chunk.get('fact_raw_score')}")
        print(f"active_weight: {chunk.get('active_weight')}")
        print(f"layer_id_score: {chunk.get('layer_id_score')}")
        print(f"borehole_id_score: {chunk.get('borehole_id_score')}")
        print(f"depth_score: {chunk.get('depth_score')}")
        print(f"lithology_score: {chunk.get('lithology_score')}")
        print(f"semantic_score: {chunk.get('semantic_score')}")
        print(f"active_dimensions: {chunk.get('active_dimensions')}")
        print(f"matched_fields: {chunk.get('matched_fields')}")
        print(f"rerank_reason: {chunk.get('rerank_reason')}")
        print(f"chunk_text: {chunk_text}")


def print_title(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


# ============================================================
# 6. 测试 1：只测试约束抽取，不依赖 Neo4j
# ============================================================

def test_collect_fact_query_constraints_without_neo4j() -> None:
    print_title("测试 1：collect_fact_query_constraints，不依赖 Neo4j")

    for question in TEST_QUESTIONS:
        print("\n问题:", question)
        print("意图:", classify_question_intent(question))

        constraints = collect_fact_query_constraints(
            user_question=question,
            entities=None,
        )

        pprint(constraints, width=120)


# ============================================================
# 7. 测试 2：使用 Mock Chunks 测试单条评分，不依赖 Neo4j
# ============================================================

def test_score_single_fact_chunk_with_mock_data() -> None:
    print_title("测试 2：score_single_fact_chunk，使用 Mock Chunks，不依赖 Neo4j")

    question = "CHGC001_2 的岩性是什么？"

    constraints = collect_fact_query_constraints(
        user_question=question,
        entities=None,
    )

    print("\n问题:", question)
    print("约束:")
    pprint(constraints, width=120)

    for chunk in MOCK_CHUNKS:
        scored = score_single_fact_chunk(
            user_question=question,
            chunk=chunk,
            entities=None,
            constraints=constraints,
        )

        print("\n" + "-" * 120)
        print("原始 chunk_id:", chunk.get("chunk_id"))
        print("评分结果:")
        pprint(
            {
                "chunk_id": scored.get("chunk_id"),
                "source_id": scored.get("source_id"),
                "final_score": scored.get("final_score"),
                "fact_raw_score": scored.get("fact_raw_score"),
                "layer_id_score": scored.get("layer_id_score"),
                "borehole_id_score": scored.get("borehole_id_score"),
                "depth_score": scored.get("depth_score"),
                "lithology_score": scored.get("lithology_score"),
                "semantic_score": scored.get("semantic_score"),
                "matched_fields": scored.get("matched_fields"),
                "rerank_reason": scored.get("rerank_reason"),
            },
            width=120,
        )


# ============================================================
# 8. 测试 3：使用 Mock Chunks 测试批量重排序，不依赖 Neo4j
# ============================================================

def test_fact_score_chunks_with_mock_data() -> None:
    print_title("测试 3：fact_score_chunks，使用 Mock Chunks，不依赖 Neo4j")

    question = "CHGC001_2 的岩性是什么？"

    print("\n问题:", question)

    scored_chunks = fact_score_chunks(
        user_question=question,
        chunks=MOCK_CHUNKS,
        entities=None,
        top_k=10,
    )

    print_scored_chunk_summary(scored_chunks)

    if scored_chunks:
        top1 = scored_chunks[0]
        expected_source_id = "CHGC001_2"

        print("\n" + "-" * 120)
        print("Top1 校验:")
        print("期望 source_id:", expected_source_id)
        print("实际 source_id:", top1.get("source_id"))

        if top1.get("source_id") == expected_source_id:
            print("PASS：精确 layer_id 命中结果排在第一。")
        else:
            print("FAIL：Top1 不是精确 layer_id 命中结果，请检查评分逻辑。")


# ============================================================
# 9. 测试 4：深度匹配测试，不依赖 Neo4j
# ============================================================

def test_depth_match_with_mock_data() -> None:
    print_title("测试 4：深度匹配评分，使用 Mock Chunks，不依赖 Neo4j")

    question = "CHGC001 在 12m 到 18m 的岩性是什么？"

    print("\n问题:", question)

    scored_chunks = fact_score_chunks(
        user_question=question,
        chunks=MOCK_CHUNKS,
        entities=None,
        top_k=10,
    )

    print_scored_chunk_summary(scored_chunks)

    if scored_chunks:
        print("\nTop1 校验:")
        print("期望命中深度区间 12.0-18.0 的 CHGC001_2")
        print("实际 Top1 source_id:", scored_chunks[0].get("source_id"))


# ============================================================
# 10. 测试 5：完整链路：Query Translation + Exact Retrieval + Fact Rerank
# ============================================================

def test_full_exact_retrieval_and_fact_rerank() -> None:
    """
    这个测试依赖 Neo4j。

    前提：
    1. Neo4j 已启动
    2. .env 配置正确
    3. 图谱中存在 LithologyLayer / Borehole / EvidenceChunk
    4. exact_retriever.py 测试已经通过
    """

    print_title("测试 5：完整链路 extract_core_entities + exact_retrieve_chunks + fact_score_chunks，依赖 Neo4j")

    for question in TEST_QUESTIONS:
        print("\n" + "#" * 120)
        print("问题:", question)
        print("意图:", classify_question_intent(question))

        try:
            entities = extract_core_entities(question)

            print("\nQuery Translation 实体:")
            pprint(entities, width=120)

            retrieved_chunks = exact_retrieve_chunks(
                user_question=question,
                entities=entities,
                top_k=10,
            )

            print("\nExact Retrieval 原始结果数量:", len(retrieved_chunks))

            scored_chunks = fact_score_chunks(
                user_question=question,
                chunks=retrieved_chunks,
                entities=entities,
                top_k=10,
            )

            print("\nFact Rerank 评分结果:")
            print_scored_chunk_summary(scored_chunks)

        except Exception as exc:
            print("完整链路测试失败。")
            print("错误信息:", repr(exc))


# ============================================================
# 11. 测试 6：别名函数 fact_rerank_chunks
# ============================================================

def test_fact_rerank_alias() -> None:
    print_title("测试 6：fact_rerank_chunks 别名函数")

    question = "CHGC001_2 的岩性是什么？"

    scored_chunks = fact_rerank_chunks(
        user_question=question,
        chunks=MOCK_CHUNKS,
        entities=None,
        top_k=3,
    )

    print_scored_chunk_summary(scored_chunks)


# ============================================================
# 12. main
# ============================================================

def main() -> None:
    test_collect_fact_query_constraints_without_neo4j()
    test_score_single_fact_chunk_with_mock_data()
    test_fact_score_chunks_with_mock_data()
    test_depth_match_with_mock_data()
    test_full_exact_retrieval_and_fact_rerank()
    test_fact_rerank_alias()


if __name__ == "__main__":
    main()