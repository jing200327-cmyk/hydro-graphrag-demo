from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.query.parser import analyze_user_question
from src.query.entity_extractor import HYDRO_FACTOR_KEYWORDS
from src.rerank.feature_extractor import (
    collect_question_entity_values,
    match_entities_in_text,
    match_permeability_factors_in_text,
    match_causal_terms,
    normalize_vector_score,
    keyword_overlap_score,
)
from src.utils.text import (
    safe_text,
    unique_keep_order,
    find_hits,
    clamp,
)


# 这里继续放原来的 HYDRO_RULES
HYDRO_RULES = [
    # 直接从原 block3_end_to_end.py 复制过来
]


def score_single_chunk(
    user_question: str,
    chunk: Dict[str, Any],
    question_analysis: Dict[str, Any],
    keep_threshold: float = 60.0,
) -> Dict[str, Any]:
    chunk_id = chunk.get("chunk_id") or chunk.get("id")
    chunk_text = safe_text(chunk.get("chunk_text"))
    chunk_type = safe_text(chunk.get("chunk_type"))
    vector_score = chunk.get("vector_score")
    graph_context_text = safe_text(chunk.get("graph_context_text"))

    scoring_text = f"{chunk_text}；{graph_context_text}"

    retrieval_keywords = question_analysis.get("检索关键词", [])
    question_entities = collect_question_entity_values(question_analysis)
    question_hydro_factors = question_analysis.get("需要关注的水文地质因素", [])

    matched_entities = match_entities_in_text(user_question, scoring_text, question_analysis)
    matched_permeability_factors = match_permeability_factors_in_text(user_question, scoring_text)
    matched_causal_terms = match_causal_terms(scoring_text)

    semantic_raw = normalize_vector_score(vector_score)

    if semantic_raw == 0:
        semantic_raw = keyword_overlap_score(user_question, scoring_text, retrieval_keywords)

    semantic_component = semantic_raw * 40.0

    if question_entities:
        entity_raw = min(len(matched_entities) / max(len(question_entities), 1), 1.0)
    else:
        entity_raw = min(len(matched_entities) / 3.0, 1.0)

    entity_component = entity_raw * 20.0

    if question_hydro_factors:
        matched_factor_names = [
            factor
            for factor in question_hydro_factors
            if factor in matched_permeability_factors
            or any(kw in scoring_text for kw in HYDRO_FACTOR_KEYWORDS.get(factor, []))
        ]
        factor_raw = min(
            len(set(matched_factor_names)) / max(len(set(question_hydro_factors)), 1),
            1.0,
        )
    else:
        factor_raw = min(len(matched_permeability_factors) / 5.0, 1.0)

    if any(kw in scoring_text for kw in ["k=", "k_value", "k_log10", "log10(k)", "m/s", "cm/s", "m/d"]):
        factor_raw = min(factor_raw + 0.25, 1.0)

    permeability_component = factor_raw * 25.0

    causal_raw = 0.0

    if matched_causal_terms:
        causal_raw += 0.6

    if any(kw in scoring_text for kw in ["permeability_effect", "effect_weight", "confidence"]):
        causal_raw += 0.3

    if any(kw in scoring_text for kw in ["TENDS_TO_HAVE", "SUPPORTS", "HAS_FEATURE", "HAS_PERMEABILITY_OBSERVATION"]):
        causal_raw += 0.2

    causal_component = min(causal_raw, 1.0) * 15.0

    total_score = round(
        clamp(
            semantic_component
            + entity_component
            + permeability_component
            + causal_component,
            0.0,
            100.0,
        ),
        2,
    )

    reason_parts = [
        f"语义相关性得分贡献 {semantic_component:.1f}/40",
        f"实体概念得分贡献 {entity_component:.1f}/20",
        f"渗透率因素得分贡献 {permeability_component:.1f}/25",
        f"因果解释得分贡献 {causal_component:.1f}/15",
    ]

    if chunk_type:
        reason_parts.append(f"候选片段类型为 {chunk_type}")

    if matched_entities:
        reason_parts.append(f"命中实体：{'、'.join(matched_entities[:8])}")

    if matched_permeability_factors:
        reason_parts.append(f"命中渗透率相关因素：{'、'.join(matched_permeability_factors[:8])}")

    if matched_causal_terms:
        reason_parts.append(f"包含可解释关系词：{'、'.join(matched_causal_terms[:8])}")

    return {
        "chunk_id": chunk_id,
        "score": total_score,
        "matched_entities": matched_entities,
        "matched_permeability_factors": matched_permeability_factors,
        "keep": bool(total_score >= keep_threshold),
        "reason": "；".join(reason_parts),
    }


def b_score_chunks(
    user_question: str,
    retrieved_chunks: List[Dict[str, Any]],
    question_analysis: Optional[Dict[str, Any]] = None,
    keep_threshold: float = 60.0,
) -> List[Dict[str, Any]]:
    if not retrieved_chunks:
        return []

    if question_analysis is None:
        question_analysis = analyze_user_question(user_question)

    evaluated = [
        score_single_chunk(
            user_question=user_question,
            chunk=chunk,
            question_analysis=question_analysis,
            keep_threshold=keep_threshold,
        )
        for chunk in retrieved_chunks
    ]

    return sorted(evaluated, key=lambda x: x["score"], reverse=True)

# ============================================================
# C 阶段：水文地质规则评分
# ============================================================

HYDRO_RULES = [
    {
        "rule_id": "R1_lithology",
        "rule_name": "岩性规则",
        "weight": 18,
        "positive_keywords": [
            "砾石", "卵石", "圆砾", "碎石",
            "粗砂", "中砂",
            "石灰岩", "白云岩", "岩溶",
            "裂隙发育",
            "强透水", "较强透水",
        ],
        "negative_keywords": [
            "黏土", "粘土",
            "粉质黏土", "粉质粘土",
            "淤泥", "泥岩", "页岩",
            "弱透水", "极弱透水",
        ],
        "explanation": (
            "砾石、粗砂、中砂通常渗透性较强；"
            "黏土、泥岩、页岩通常渗透性弱；"
            "碳酸盐岩在岩溶或裂隙发育时可表现为强透水。"
        ),
    },
    {
        "rule_id": "R2_pore_structure",
        "rule_name": "孔隙结构规则",
        "weight": 12,
        "positive_keywords": [
            "有效孔隙", "孔隙度高", "孔隙连通",
            "连通性好", "孔隙连通性较好",
            "孔隙水", "孔隙发育",
        ],
        "negative_keywords": [
            "孤立孔隙", "孔隙细小",
            "孔隙闭塞", "孔隙被堵塞",
            "孔隙连通性差",
        ],
        "explanation": (
            "有效孔隙度越高、孔隙连通性越好，通常越有利于地下水渗流。"
        ),
    },
    {
        "rule_id": "R3_grain_size_sorting",
        "rule_name": "粒径与分选性规则",
        "weight": 18,
        "positive_keywords": [
            "粒径大", "颗粒较粗",
            "粗砂", "中砂",
            "砾石", "卵石", "圆砾",
            "分选好", "分选较好",
            "级配不良",
        ],
        "negative_keywords": [
            "细砂", "粉砂", "粉粒",
            "黏粒", "粘粒", "细颗粒",
            "细颗粒充填",
            "分选差", "级配复杂",
            "含泥", "泥质充填",
        ],
        "explanation": (
            "粒径越大通常渗透性越强；"
            "分选性越好通常孔隙连通性越好；"
            "细颗粒充填会降低渗透性。"
        ),
    },
    {
        "rule_id": "R4_cementation",
        "rule_name": "胶结程度规则",
        "weight": 12,
        "positive_keywords": [
            "松散", "胶结差", "未胶结", "弱胶结",
        ],
        "negative_keywords": [
            "胶结", "胶结程度强", "强胶结",
            "泥质胶结", "钙质胶结", "铁质胶结",
            "孔隙被胶结",
        ],
        "explanation": (
            "胶结程度越强，孔隙越容易被堵塞，渗透性越弱；"
            "松散沉积物通常更有利于渗透。"
        ),
    },
    {
        "rule_id": "R5_fracture_structure",
        "rule_name": "裂隙与构造规则",
        "weight": 16,
        "positive_keywords": [
            "裂隙", "裂隙发育",
            "节理", "节理发育",
            "断层", "破碎",
            "岩体破碎", "构造裂隙",
        ],
        "negative_keywords": [
            "裂隙闭合", "裂隙充填",
            "泥质充填", "矿物充填",
            "方解石充填", "充填密实",
        ],
        "explanation": (
            "裂隙、断层、节理发育可显著提高岩体渗透性；"
            "若裂隙被泥质或矿物充填，则渗透性可能降低。"
        ),
    },
    {
        "rule_id": "R6_burial_compaction",
        "rule_name": "埋深与压实规则",
        "weight": 10,
        "positive_keywords": [
            "浅部", "浅层", "松散", "稍密", "埋深小",
        ],
        "negative_keywords": [
            "埋深增加", "埋深较大",
            "深部", "压实",
            "密实", "很密", "固结",
        ],
        "explanation": (
            "埋深增加通常导致压实增强、孔隙度降低、渗透率下降；"
            "浅部松散沉积层通常更利于渗透。"
        ),
    },
    {
        "rule_id": "R7_hydro_condition",
        "rule_name": "水文地质条件规则",
        "weight": 14,
        "positive_keywords": [
            "补给条件良好", "径流通畅",
            "径流条件好", "地下水运移",
            "导水通道", "富水性好",
        ],
        "negative_keywords": [
            "隔水层", "弱透水层",
            "垂向阻隔",
            "夹黏土", "夹粘土", "夹粉土",
            "限制垂向渗透",
        ],
        "explanation": (
            "补给条件良好、径流通畅有利于地下水运移；"
            "隔水层或弱透水夹层会限制垂向渗透。"
        ),
    },
]


def match_hydro_rules(text: str) -> Tuple[float, List[str], List[str], Dict[str, Any]]:
    """
    基于水文地质专家规则，对文本进行规则匹配评分。

    返回：
    - rule_score: 0-100
    - matched_rules: 命中的规则解释
    - uncertainty_flags: 冲突或不确定性提示
    - detail: 每条规则的命中详情
    """

    text = safe_text(text)

    total_weight = sum(float(rule["weight"]) for rule in HYDRO_RULES)
    weighted_score = 0.0

    matched_rules: List[str] = []
    uncertainty_flags: List[str] = []
    detail: Dict[str, Any] = {}

    high_perm_terms: List[str] = []
    low_perm_terms: List[str] = []

    for rule in HYDRO_RULES:
        rule_id = rule["rule_id"]
        rule_name = rule["rule_name"]
        weight = float(rule["weight"])

        pos_hits = find_hits(text, rule["positive_keywords"])
        neg_hits = find_hits(text, rule["negative_keywords"])

        if pos_hits or neg_hits:
            if pos_hits and not neg_hits:
                local_score_ratio = 1.0
                matched_rules.append(
                    f"{rule_name}：命中增强渗透性因素 {pos_hits}。"
                    f"{rule['explanation']}"
                )
                high_perm_terms.extend(pos_hits)

            elif neg_hits and not pos_hits:
                local_score_ratio = 1.0
                matched_rules.append(
                    f"{rule_name}：命中降低渗透性因素 {neg_hits}。"
                    f"{rule['explanation']}"
                )
                low_perm_terms.extend(neg_hits)

            else:
                local_score_ratio = 0.9
                matched_rules.append(
                    f"{rule_name}：同时命中增强因素 {pos_hits} "
                    f"和降低因素 {neg_hits}，需要结合上下文判断。"
                    f"{rule['explanation']}"
                )
                high_perm_terms.extend(pos_hits)
                low_perm_terms.extend(neg_hits)

                uncertainty_flags.append(
                    f"{rule_name}存在正负因素并存："
                    f"增强因素{pos_hits}；降低因素{neg_hits}"
                )

            weighted_score += weight * local_score_ratio

        detail[rule_id] = {
            "rule_name": rule_name,
            "positive_hits": pos_hits,
            "negative_hits": neg_hits,
            "weight": weight,
        }

    if high_perm_terms and low_perm_terms:
        uncertainty_flags.append(
            f"片段同时包含强透水指示词{unique_keep_order(high_perm_terms)}"
            f"和弱透水指示词{unique_keep_order(low_perm_terms)}，"
            f"需要区分直接证据和条件推断。"
        )

    rule_score = round(
        clamp(weighted_score / max(total_weight, 1.0) * 100.0, 0.0, 100.0),
        2,
    )

    return (
        rule_score,
        unique_keep_order(matched_rules),
        unique_keep_order(uncertainty_flags),
        detail,
    )


def evidence_completeness_score(chunk: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    评估一个 chunk 的证据完整性。

    返回：
    - evidence_score: 0-100
    - hits: 证据完整性命中项说明
    """

    text = (
        f"{safe_text(chunk.get('chunk_text'))}；"
        f"{safe_text(chunk.get('graph_context_text'))}"
    )

    hits: List[str] = []
    score = 0.0

    if any(kw in text for kw in ["k=", "k_value", "k_log10", "log10(k)"]):
        score += 35
        hits.append("包含直接渗透率或 log10(k) 证据")

    if any(kw in text for kw in ["k_unit", "m/s", "cm/s", "m/d"]):
        score += 15
        hits.append("包含渗透率单位")

    if any(kw in text for kw in ["source", "来源"]):
        score += 10
        hits.append("包含数据来源")

    if any(kw in text for kw in ["reliability", "可靠性"]):
        score += 10
        hits.append("包含可靠性信息")

    if any(
        kw in text
        for kw in [
            "lithology_major_v2",
            "lithology_minor_v3",
            "岩性大类",
            "岩性小类",
        ]
    ):
        score += 15
        hits.append("包含岩性信息")

    if any(
        kw in text
        for kw in [
            "permeability_effect",
            "effect_weight",
            "confidence",
            "HydroFeature",
        ]
    ):
        score += 15
        hits.append("包含水文地质特征规则")

    return round(clamp(score, 0.0, 100.0), 2), hits