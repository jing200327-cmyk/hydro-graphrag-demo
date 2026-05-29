from __future__ import annotations

import re
from typing import Dict, List

from src.utils.text import safe_text, contains_any


FACT_QUERY_INTENT = "fact_query"


# 仍然保留原有中文意图，避免影响已有 parser / UI 展示逻辑
INTENT_RULES: Dict[str, List[str]] = {
    "查询定义": [
        "是什么", "什么意思", "定义", "概念", "含义",
        "解释一下", "什么是", "如何理解",
    ],
    "查询影响因素": [
        "影响因素", "受什么影响", "哪些因素", "为什么影响",
        "控制因素", "主要因素", "决定因素",
    ],
    "比较不同地层或岩性的渗透率": [
        "比较", "哪个更", "谁更", "高于", "低于", "大于",
        "小于", "差异", "区别", "相比", "对比",
    ],
    "判断某区域或某地层渗透性强弱": [
        "强弱", "强不强", "弱不弱", "是否透水",
        "透水性如何", "渗透性如何", "是否强透水",
        "是否弱透水", "等级", "属于什么等级",
    ],
    "分析渗透率变化原因": [
        "为什么", "原因", "变化原因", "降低原因", "升高原因",
        "变大", "变小", "增大", "减小", "异常", "差异原因",
    ],
    "预测或推断水文地质条件": [
        "预测", "推断", "估计", "估算", "判断", "可能",
        "大概", "能否", "是否可能", "水文地质条件", "地下水条件",
    ],
}


# 与“渗透率 / 透水性”强相关的问题，暂时继续走原来的 B/C 水文规则链路
PERMEABILITY_QUERY_KEYWORDS = [
    "渗透率", "渗透系数", "透水性", "导水性", "水力传导系数",
    "k值", "K值", "k=", "K=", "k_log10", "log10", "log10(k)",
    "透水等级", "强透水", "较强透水", "中等透水", "弱透水", "极弱透水",
    "渗透性强", "渗透性弱",
]


# 事实查询通常关注这些结构化字段
FACT_FIELD_KEYWORDS = [
    "钻孔", "孔号", "层位", "分层", "层号", "地层",
    "岩性", "岩性大类", "岩性小类",
    "层顶", "层底", "厚度", "深度", "埋深", "范围",
    "起始深度", "终止深度", "顶板", "底板",
]


# 事实查询常见问法
FACT_QUERY_KEYWORDS = [
    "是什么", "是多少", "有哪些", "哪一层", "哪几层",
    "对应", "位于", "查询", "查一下", "给出", "列出",
    "显示", "找到", "属于", "记录",
]


# 明显不是事实查表，而是解释、比较、推断类的问题
NON_FACT_REASONING_KEYWORDS = [
    "为什么", "原因", "影响因素", "受什么影响", "控制因素",
    "决定因素", "预测", "推断", "估计", "估算", "可能",
    "比较", "哪个更", "谁更", "高于", "低于", "大于", "小于",
    "差异", "区别", "相比", "对比",
]


# 兼容 CHGC001_2、ZK001_3 这类 layer_id
LAYER_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"([A-Za-z]{2,}[A-Za-z0-9-]*\d+[A-Za-z0-9-]*\s*_\s*\d+[A-Za-z0-9-]*)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


# 兼容 CHGC001、ZK001 这类 borehole_id
BOREHOLE_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"([A-Za-z]{2,}[A-Za-z0-9-]*\d+[A-Za-z0-9-]*)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


DEPTH_RE = re.compile(
    r"(\d+\.?\d*\s*(?:m|米)\s*[-~至到]\s*\d+\.?\d*\s*(?:m|米)?)"
    r"|(\d+\.?\d*\s*[-~至到]\s*\d+\.?\d*\s*(?:m|米))"
    r"|(\d+\.?\d*\s*(?:m|米))"
)


def _normalize_structured_id(value: str) -> str:
    """
    统一结构化 ID 格式：
    - 去掉空格
    - 转大写
    """
    return re.sub(r"\s+", "", safe_text(value)).upper()


def _has_layer_id(question: str) -> bool:
    return bool(LAYER_ID_RE.search(question))


def _has_borehole_id(question: str) -> bool:
    """
    判断是否存在独立 borehole_id。

    注意：
    CHGC001_2 中也包含 CHGC001，
    所以这里先移除 layer_id，避免把 layer_id 误判成独立 borehole_id。
    """
    text_without_layer_id = LAYER_ID_RE.sub(" ", question)
    return bool(BOREHOLE_ID_RE.search(text_without_layer_id))


def _has_depth_expression(question: str) -> bool:
    return bool(DEPTH_RE.search(question))


def _is_permeability_related(question: str) -> bool:
    return contains_any(question, PERMEABILITY_QUERY_KEYWORDS)


def is_fact_query(user_question: str) -> bool:
    """
    判断是否是事实查询。

    事实查询的典型特征：
    1. 问题中出现明确的 layer_id，例如 CHGC001_2。
    2. 问题中出现明确的 borehole_id，例如 CHGC001。
    3. 问题关注岩性、层顶、层底、厚度、深度、埋深等结构化字段。
    4. 不涉及渗透率、透水等级、原因解释、比较、推断等任务。

    设计原则：
    - fact_query 优先服务于精确字段查询。
    - 涉及渗透率 / 透水性 / 透水等级的问题，暂时不归入 fact_query。
    """

    question = safe_text(user_question)

    if not question:
        return False

    # 渗透率相关问题继续走原有 B/C 水文规则链路
    if _is_permeability_related(question):
        return False

    # 解释、比较、推断类问题不进入事实查询链路
    if contains_any(question, NON_FACT_REASONING_KEYWORDS):
        return False

    has_layer_id = _has_layer_id(question)
    has_borehole_id = _has_borehole_id(question)
    has_depth = _has_depth_expression(question)

    has_fact_field = contains_any(question, FACT_FIELD_KEYWORDS)
    has_fact_query_word = contains_any(question, FACT_QUERY_KEYWORDS)

    # 最强事实查询信号：精确 layer_id + 结构化字段
    if has_layer_id and (has_fact_field or has_fact_query_word):
        return True

    # borehole_id + 深度范围 + 岩性/层位字段
    if has_borehole_id and has_depth and has_fact_field:
        return True

    # borehole_id + 查询字段，例如：CHGC001 有哪些分层？
    if has_borehole_id and has_fact_field and has_fact_query_word:
        return True

    # 没有显式 ID，但问的是结构化字段，也可作为弱 fact_query
    # 例如：粉质黏土层的层顶层底是多少？
    if has_fact_field and has_fact_query_word and has_depth:
        return True

    return False


def classify_question_intent(user_question: str) -> str:
    """
    问题意图识别入口。

    新增优先级：
    1. 先识别 fact_query。
    2. 如果不是 fact_query，再走原有水文地质意图体系。
    """

    question = safe_text(user_question)

    # 新增：事实查询优先识别
    if is_fact_query(question):
        return FACT_QUERY_INTENT

    scores: Dict[str, int] = {}

    for intent, keywords in INTENT_RULES.items():
        scores[intent] = sum(1 for kw in keywords if kw in question)

    if contains_any(question, ["为什么", "原因", "变化原因", "降低", "升高", "变大", "变小"]):
        scores["分析渗透率变化原因"] += 2

    if contains_any(question, ["比较", "哪个更", "谁更", "高于", "低于", "大于", "小于", "差异"]):
        scores["比较不同地层或岩性的渗透率"] += 2

    if contains_any(question, ["预测", "推断", "估计", "估算", "大概", "可能"]):
        scores["预测或推断水文地质条件"] += 2

    if contains_any(question, ["强弱", "等级", "透水性如何", "渗透性如何", "是否透水"]):
        scores["判断某区域或某地层渗透性强弱"] += 2

    if contains_any(question, ["影响因素", "控制因素", "决定因素"]):
        scores["查询影响因素"] += 2

    best_intent, best_score = max(scores.items(), key=lambda x: x[1])

    if best_score == 0:
        if contains_any(question, ["渗透率", "渗透系数", "透水性", "k值", "K值"]):
            return "判断某区域或某地层渗透性强弱"

        return "预测或推断水文地质条件"

    return best_intent