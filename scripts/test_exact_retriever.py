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
# 2. 尝试加载 .env
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
from src.retrieval.exact_retriever import (
    extract_exact_ids,
    exact_search_by_layer_id,
    exact_search_by_borehole_id,
    exact_retrieve_chunks,
)


TEST_QUESTIONS = [
    "CHGC001_2 的岩性是什么？",
    "CHGC001_1 的层顶和层底分别是多少？",
    "CHGC001 在 12m 到 18m 的岩性是什么？",
    "chgc001 _ 2 的岩性小类是什么？",
    "CHGC001 第2层的层底是多少？",
]


def print_chunk_summary(chunks: List[Dict[str, Any]], max_text_len: int = 220) -> None:
    if not chunks:
        print("没有检索到结果。")
        return

    print(f"共检索到 {len(chunks)} 条结果：")

    for idx, chunk in enumerate(chunks, start=1):
        chunk_text = chunk.get("chunk_text", "") or ""
        chunk_text = chunk_text.replace("\n", " ")

        if len(chunk_text) > max_text_len:
            chunk_text = chunk_text[:max_text_len] + "..."

        print("\n" + "-" * 100)
        print(f"Top {idx}")
        print(f"chunk_id: {chunk.get('chunk_id')}")
        print(f"chunk_type: {chunk.get('chunk_type')}")
        print(f"source_label: {chunk.get('source_label')}")
        print(f"source_id: {chunk.get('source_id')}")
        print(f"retrieval_method: {chunk.get('retrieval_method')}")
        print(f"exact_match_type: {chunk.get('exact_match_type')}")
        print(f"exact_match_value: {chunk.get('exact_match_value')}")
        print(f"exact_score: {chunk.get('exact_score')}")
        print(f"vector_score: {chunk.get('vector_score')}")
        print(f"is_synthetic: {chunk.get('is_synthetic')}")
        print(f"graph_expandable: {chunk.get('graph_expandable')}")
        print(f"chunk_text: {chunk_text}")


def test_extract_exact_ids_without_neo4j() -> None:
    """
    测试 1：只测试 ID 抽取，不连接 Neo4j。
    """

    print("\n" + "=" * 120)
    print("测试 1：extract_exact_ids，不连接 Neo4j")
    print("=" * 120)

    for question in TEST_QUESTIONS:
        exact_ids = extract_exact_ids(question)

        print("\n问题:", question)
        print("抽取结果:")
        pprint(exact_ids, width=120)


def test_direct_layer_search() -> None:
    """
    测试 2：直接按 layer_id 精确查询。

    需要：
    1. Neo4j 已启动
    2. .env 中 Neo4j 配置正确
    3. 数据库中存在 LithologyLayer.layer_id = CHGC001_2
    """

    print("\n" + "=" * 120)
    print("测试 2：exact_search_by_layer_id")
    print("=" * 120)

    layer_ids = ["CHGC001_2", "CHGC001_1"]

    try:
        chunks = exact_search_by_layer_id(layer_ids=layer_ids, limit=10)
        print("查询 layer_ids:", layer_ids)
        print_chunk_summary(chunks)
    except Exception as exc:
        print("exact_search_by_layer_id 执行失败。")
        print("请检查 Neo4j 是否启动、.env 是否正确、数据库中是否存在 LithologyLayer.layer_id。")
        print("错误信息:", repr(exc))


def test_direct_borehole_search() -> None:
    """
    测试 3：直接按 borehole_id 精确查询。

    需要：
    1. Neo4j 已启动
    2. .env 中 Neo4j 配置正确
    3. 数据库中存在 Borehole.borehole_id = CHGC001
    """

    print("\n" + "=" * 120)
    print("测试 3：exact_search_by_borehole_id")
    print("=" * 120)

    borehole_ids = ["CHGC001"]

    try:
        chunks = exact_search_by_borehole_id(borehole_ids=borehole_ids, limit=10)
        print("查询 borehole_ids:", borehole_ids)
        print_chunk_summary(chunks)
    except Exception as exc:
        print("exact_search_by_borehole_id 执行失败。")
        print("请检查 Neo4j 是否启动、.env 是否正确、数据库中是否存在 Borehole.borehole_id。")
        print("错误信息:", repr(exc))


def test_pipeline_style_exact_retrieve_without_entities() -> None:
    """
    测试 4：模拟 pipeline 调用，但不传 entities。

    exact_retrieve_chunks 会直接从 user_question 中用正则抽取 layer_id / borehole_id。
    """

    print("\n" + "=" * 120)
    print("测试 4：exact_retrieve_chunks，不传 entities")
    print("=" * 120)

    for question in TEST_QUESTIONS:
        print("\n" + "#" * 120)
        print("问题:", question)
        print("意图:", classify_question_intent(question))

        try:
            chunks = exact_retrieve_chunks(
                user_question=question,
                entities=None,
                top_k=10,
            )
            print_chunk_summary(chunks)
        except Exception as exc:
            print("exact_retrieve_chunks 执行失败。")
            print("错误信息:", repr(exc))


def test_pipeline_style_exact_retrieve_with_entities() -> None:
    """
    测试 5：模拟完整 query translation + exact retrieval。

    这一步会调用 extract_core_entities()，因此也会触发 Neo4j 词表加载。
    """

    print("\n" + "=" * 120)
    print("测试 5：extract_core_entities + exact_retrieve_chunks")
    print("=" * 120)

    for question in TEST_QUESTIONS:
        print("\n" + "#" * 120)
        print("问题:", question)
        print("意图:", classify_question_intent(question))

        try:
            entities = extract_core_entities(question)

            print("\nQuery Translation 抽取实体:")
            pprint(entities, width=120)

            print("\nExact IDs:")
            pprint(extract_exact_ids(question, entities), width=120)

            print("\nExact Retrieval 结果:")
            chunks = exact_retrieve_chunks(
                user_question=question,
                entities=entities,
                top_k=10,
            )
            print_chunk_summary(chunks)

        except Exception as exc:
            print("完整链路测试失败。")
            print("可能原因：Neo4j 未启动、.env 配置错误、数据库名错误、或图谱数据字段不一致。")
            print("错误信息:", repr(exc))


def main() -> None:
    test_extract_exact_ids_without_neo4j()
    test_direct_layer_search()
    test_direct_borehole_search()
    test_pipeline_style_exact_retrieve_without_entities()
    test_pipeline_style_exact_retrieve_with_entities()


if __name__ == "__main__":
    main()