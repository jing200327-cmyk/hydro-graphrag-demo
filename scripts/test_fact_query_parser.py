from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint


# ============================================================
# 1. 自动把项目根目录加入 sys.path
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ============================================================
# 2. 导入待测试函数
# ============================================================

from src.query.intent_classifier import classify_question_intent, is_fact_query
from src.query.entity_extractor import (
    extract_layer_ids,
    extract_borehole_ids,
    extract_layer_ordinal_expressions,
    extract_depth_expressions,
    extract_core_entities,
)


# ============================================================
# 3. 测试问题
# ============================================================

TEST_QUESTIONS = [
    {
        "question": "CHGC001_2 的岩性是什么？",
        "expected_intent": "fact_query",
    },
    {
        "question": "CHGC001_1 的层顶和层底分别是多少？",
        "expected_intent": "fact_query",
    },
    {
        "question": "CHGC001 在 12m 到 18m 的岩性是什么？",
        "expected_intent": "fact_query",
    },
    {
        "question": "CHGC001_2 的透水等级是什么？",
        "expected_intent": "判断某区域或某地层渗透性强弱",
    },
    {
        "question": "为什么粗砂层渗透性较强？",
        "expected_intent": "分析渗透率变化原因",
    },
    {
        "question": "比较 CHGC001_1 和 CHGC001_2 的渗透率差异",
        "expected_intent": "比较不同地层或岩性的渗透率",
    },
    {
        "question": "chgc001 _ 2 的岩性小类是什么？",
        "expected_intent": "fact_query",
    },
    {
        "question": "CHGC001 第2层的层底是多少？",
        "expected_intent": "fact_query",
    },
]


def test_intent_only() -> None:
    """
    只测试意图识别。
    这个测试不依赖 Neo4j。
    """

    print("\n" + "=" * 100)
    print("测试 1：问题意图识别 classify_question_intent")
    print("=" * 100)

    passed = 0
    failed = 0

    for item in TEST_QUESTIONS:
        question = item["question"]
        expected = item["expected_intent"]

        actual = classify_question_intent(question)

        ok = actual == expected

        if ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"\n[{status}]")
        print(f"问题: {question}")
        print(f"期望: {expected}")
        print(f"实际: {actual}")
        print(f"is_fact_query: {is_fact_query(question)}")

    print("\n" + "-" * 100)
    print(f"意图识别测试完成：PASS={passed}, FAIL={failed}")
    print("-" * 100)


def test_regex_entity_only() -> None:
    """
    只测试正则抽取。
    这个测试不依赖 Neo4j。
    """

    print("\n" + "=" * 100)
    print("测试 2：正则实体抽取 layer_id / borehole_id / 深度")
    print("=" * 100)

    for item in TEST_QUESTIONS:
        question = item["question"]

        print("\n问题:", question)
        print("layer_ids:", extract_layer_ids(question))
        print("borehole_ids:", extract_borehole_ids(question))
        print("layer_ordinals:", extract_layer_ordinal_expressions(question))
        print("depth_expressions:", extract_depth_expressions(question))


def test_full_entity_extractor() -> None:
    """
    测试完整 extract_core_entities。

    注意：
    extract_core_entities 会调用 load_vocab_cache()
    load_vocab_cache() 会连接 Neo4j。
    所以这个测试要求：
    1. .env 配置正确
    2. Neo4j 正常启动
    3. 数据库里已有 Borehole / LithologyLayer 等节点
    """

    print("\n" + "=" * 100)
    print("测试 3：完整实体抽取 extract_core_entities")
    print("=" * 100)

    for item in TEST_QUESTIONS:
        question = item["question"]

        print("\n问题:", question)

        try:
            entities = extract_core_entities(question)
            pprint(entities, width=120)
        except Exception as exc:
            print("extract_core_entities 执行失败。")
            print("可能原因：Neo4j 未启动、.env 配置错误、数据库名错误、或图谱中缺少相关节点。")
            print("错误信息:", repr(exc))


def main() -> None:
    test_intent_only()
    test_regex_entity_only()
    test_full_entity_extractor()


if __name__ == "__main__":
    main()