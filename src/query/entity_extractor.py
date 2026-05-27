from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from configs.settings import NEO4J_DATABASE
from src.utils.env import get_driver
from src.utils.text import safe_text, unique_keep_order, contains_any


_VOCAB_CACHE: Optional[Dict[str, List[str]]] = None


HYDRO_FACTOR_KEYWORDS = {
    "粒径": [
        "粒径", "颗粒", "砾石", "卵石", "圆砾", "碎石",
        "粗砂", "中砂", "细砂", "粉砂", "砂", "粉粒", "黏粒", "粘粒"
    ],
    "黏粒含量": [
        "黏粒", "粘粒", "黏土", "粘土", "泥质", "含泥", "粉质黏土",
        "粉质粘土", "黏性土", "粘性土", "淤泥"
    ],
    "分选性": [
        "分选", "分选好", "分选较好", "分选差", "级配", "级配良好", "级配不良"
    ],
    "密实度": [
        "松散", "稍密", "中密", "密实", "很密", "压实", "固结"
    ],
    "胶结程度": [
        "胶结", "泥质胶结", "钙质胶结", "铁质胶结", "胶结差", "胶结较好"
    ],
    "裂隙发育": [
        "裂隙", "节理", "破碎", "裂隙发育", "节理发育",
        "岩体破碎", "裂隙闭合", "充填"
    ],
    "风化程度": [
        "全风化", "强风化", "中风化", "弱风化", "微风化",
        "风化裂隙", "黏土化", "粘土化"
    ],
    "夹层与互层": [
        "夹层", "互层", "薄层", "透镜体", "夹黏土", "夹粘土",
        "夹粉土", "夹砂", "砂黏互层", "砂粘互层"
    ],
    "水力学参数": [
        "渗透率", "渗透系数", "k值", "K值", "水力传导系数",
        "导水性", "透水性", "富水性", "k=", "log10", "k_log10"
    ],
    "空间层位": [
        "钻孔", "层位", "分层", "深度", "层顶", "层底", "厚度",
        "中间深度", "埋深"
    ],
    "透水等级": [
        "强透水", "较强透水", "中等透水", "弱透水", "极弱透水",
        "透水等级", "渗透性强", "渗透性弱"
    ],
}


PERMEABILITY_DIRECT_KEYWORDS = [
    "渗透率", "渗透系数", "透水性", "导水性", "水力传导系数",
    "k=", "K=", "k值", "K值", "k_log10", "log10(k)", "m/s", "cm/s", "m/d"
]


CAUSAL_KEYWORDS = [
    "因为", "由于", "导致", "造成", "影响", "控制", "决定",
    "提高", "增大", "增加", "增强",
    "降低", "减小", "削弱", "阻隔",
    "正向", "负向", "强正向", "强负向",
    "对应", "支持", "倾向", "表明",
    "INCREASES", "REDUCES", "AFFECTS", "SUPPORTS", "TENDS_TO_HAVE",
]


def run_cypher_to_list(driver, database: str, cypher: str, field: str) -> List[str]:
    with driver.session(database=database) as session:
        records = session.run(cypher).data()

    values: List[str] = []

    for record in records:
        value = record.get(field)

        if value is None:
            continue

        if isinstance(value, list):
            values.extend([safe_text(v) for v in value if safe_text(v)])
        else:
            text = safe_text(value)
            if text:
                values.append(text)

    return sorted(set(values), key=lambda x: len(x), reverse=True)


def load_vocab_cache() -> Dict[str, List[str]]:
    global _VOCAB_CACHE

    if _VOCAB_CACHE is not None:
        return _VOCAB_CACHE

    driver = get_driver()

    queries = {
        "borehole": """
            MATCH (n:Borehole)
            RETURN collect(DISTINCT n.borehole_id) AS values
        """,
        "layer": """
            MATCH (n:LithologyLayer)
            RETURN collect(DISTINCT n.layer_id) AS values
        """,
        "lithology_major": """
            MATCH (n:LithologyType)
            RETURN collect(DISTINCT n.lithology_major_v2) AS values
        """,
        "lithology_minor": """
            MATCH (n:LithologyType)
            RETURN collect(DISTINCT n.lithology_minor_v3) AS values
        """,
        "feature_type": """
            MATCH (n:HydroFeature)
            RETURN collect(DISTINCT n.feature_type) AS values
        """,
        "feature_value": """
            MATCH (n:HydroFeature)
            RETURN collect(DISTINCT n.feature_value) AS values
        """,
        "permeability_level": """
            MATCH (n:PermeabilityLevel)
            RETURN collect(DISTINCT n.level_name) AS values
        """,
    }

    _VOCAB_CACHE = {
        key: run_cypher_to_list(driver, NEO4J_DATABASE, cypher, "values")
        for key, cypher in queries.items()
    }

    return _VOCAB_CACHE


def find_vocab_matches(question: str, vocab: List[str], max_count: int = 50) -> List[str]:
    matches: List[str] = []

    for term in vocab:
        if term and term in question:
            matches.append(term)

        if len(matches) >= max_count:
            break

    return unique_keep_order(matches)


def extract_depth_expressions(question: str) -> List[str]:
    patterns = [
        r"\d+\.?\d*\s*[m米]",
        r"\d+\.?\d*\s*[-~至到]\s*\d+\.?\d*\s*[m米]",
        r"\d+\.?\d*\s*米\s*[-~至到]\s*\d+\.?\d*\s*米",
        r"层顶\s*\d+\.?\d*",
        r"层底\s*\d+\.?\d*",
        r"厚度\s*\d+\.?\d*",
        r"埋深\s*\d+\.?\d*",
        r"中间深度\s*\d+\.?\d*",
    ]

    results: List[str] = []

    for pattern in patterns:
        results.extend(re.findall(pattern, question))

    return unique_keep_order(results)


def extract_core_entities(user_question: str) -> List[Dict[str, str]]:
    question = safe_text(user_question)
    vocab = load_vocab_cache()

    entities: List[Dict[str, str]] = []

    mapping = [
        ("钻孔", "Neo4j.Borehole", vocab.get("borehole", [])),
        ("层位", "Neo4j.LithologyLayer", vocab.get("layer", [])),
        ("岩性大类", "Neo4j.LithologyType", vocab.get("lithology_major", [])),
        ("岩性小类", "Neo4j.LithologyType", vocab.get("lithology_minor", [])),
        ("水文地质特征类型", "Neo4j.HydroFeature", vocab.get("feature_type", [])),
        ("水文地质特征", "Neo4j.HydroFeature", vocab.get("feature_value", [])),
        ("透水等级", "Neo4j.PermeabilityLevel", vocab.get("permeability_level", [])),
    ]

    for entity_type, source, terms in mapping:
        for value in find_vocab_matches(question, terms):
            entities.append(
                {
                    "实体类型": entity_type,
                    "实体值": value,
                    "来源": source,
                }
            )

    for factor_name, keywords in HYDRO_FACTOR_KEYWORDS.items():
        for kw in keywords:
            if kw in question:
                entities.append(
                    {
                        "实体类型": factor_name,
                        "实体值": kw,
                        "来源": "HydroExpertDictionary",
                    }
                )

    for value in extract_depth_expressions(question):
        entities.append(
            {
                "实体类型": "深度或层位范围",
                "实体值": value,
                "来源": "Regex",
            }
        )

    dedup: List[Dict[str, str]] = []
    seen = set()

    for entity in entities:
        key = (entity["实体类型"], entity["实体值"])

        if key not in seen:
            seen.add(key)
            dedup.append(entity)

    return dedup


def identify_hydro_factors(
    user_question: str,
    entities: List[Dict[str, str]],
) -> List[str]:
    question = safe_text(user_question)
    factors: List[str] = []

    for factor_name, keywords in HYDRO_FACTOR_KEYWORDS.items():
        if contains_any(question, keywords):
            factors.append(factor_name)

    entity_text = " ".join([e.get("实体值", "") for e in entities])

    if contains_any(entity_text, ["砂", "砾", "卵石", "圆砾", "碎石"]):
        factors.extend(["粒径", "分选性", "密实度", "黏粒含量"])

    if contains_any(entity_text, ["黏土", "粘土", "粉质黏土", "粉质粘土", "淤泥", "粉土"]):
        factors.extend(["黏粒含量", "密实度", "夹层与互层"])

    if contains_any(entity_text, ["岩", "风化", "裂隙", "破碎"]):
        factors.extend(["裂隙发育", "风化程度", "胶结程度"])

    if contains_any(question, ["垂向", "层间", "夹层", "互层"]):
        factors.extend(["夹层与互层", "空间层位"])

    if contains_any(question, ["渗透率", "渗透系数", "k值", "K值", "透水性"]):
        factors.append("水力学参数")

    return unique_keep_order(factors)