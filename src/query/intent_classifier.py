from __future__ import annotations

from src.utils.text import safe_text, contains_any


INTENT_RULES = {
    "查询定义": ["是什么", "什么意思", "定义", "概念", "含义", "解释一下", "什么是", "如何理解"],
    "查询影响因素": ["影响因素", "受什么影响", "哪些因素", "为什么影响", "控制因素", "主要因素", "决定因素"],
    "比较不同地层或岩性的渗透率": ["比较", "哪个更", "谁更", "高于", "低于", "大于", "小于", "差异", "区别", "相比", "对比"],
    "判断某区域或某地层渗透性强弱": ["强弱", "强不强", "弱不弱", "是否透水", "透水性如何", "渗透性如何", "是否强透水", "是否弱透水", "等级", "属于什么等级"],
    "分析渗透率变化原因": ["为什么", "原因", "变化原因", "降低原因", "升高原因", "变大", "变小", "增大", "减小", "异常", "差异原因"],
    "预测或推断水文地质条件": ["预测", "推断", "估计", "估算", "判断", "可能", "大概", "能否", "是否可能", "水文地质条件", "地下水条件"],
}


def classify_question_intent(user_question: str) -> str:
    question = safe_text(user_question)
    scores = {}

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