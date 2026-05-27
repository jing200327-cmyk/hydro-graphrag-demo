from __future__ import annotations

from typing import Dict, List

from src.utils.text import safe_text, unique_keep_order


def generate_search_keywords(
    user_question: str,
    entities: List[Dict[str, str]],
    intent: str,
    hydro_factors: List[str],
) -> List[str]:
    keywords: List[str] = [safe_text(user_question)]
    keywords.extend([e.get("实体值", "") for e in entities])
    keywords.extend(hydro_factors)

    intent_keywords = {
        "查询定义": ["定义", "概念", "解释", "水文地质意义"],
        "查询影响因素": ["影响因素", "控制因素", "粒径", "黏粒含量", "分选性", "密实度", "裂隙发育", "胶结程度"],
        "比较不同地层或岩性的渗透率": ["渗透率比较", "透水性差异", "岩性透水等级", "k_log10", "PermeabilityObservation"],
        "判断某区域或某地层渗透性强弱": ["透水等级", "强透水", "较强透水", "中等透水", "弱透水", "极弱透水", "渗透率记录"],
        "分析渗透率变化原因": ["渗透率变化原因", "粒径变化", "黏粒含量", "夹层", "裂隙", "风化", "密实度", "胶结"],
        "预测或推断水文地质条件": ["渗透率估计", "透水性推断", "水文地质条件", "相似岩性", "相似分层", "实测渗透率"],
    }

    keywords.extend(intent_keywords.get(intent, []))
    keywords.extend(["渗透率", "渗透系数", "透水性", "岩性", "水文地质特征"])

    return unique_keep_order(keywords)


def build_retrieval_direction(
    intent: str,
    entities: List[Dict[str, str]],
    hydro_factors: List[str],
) -> str:
    entity_types = {e["实体类型"] for e in entities}
    directions = [
        "首先对 EvidenceChunk.chunk_text 进行向量检索，召回与用户问题语义相近的证据。"
    ]

    if "钻孔" in entity_types:
        directions.append("若问题包含钻孔编号，应从 Borehole 沿 HAS_LAYER 扩展到 LithologyLayer。")

    if "层位" in entity_types or "深度或层位范围" in entity_types:
        directions.append("若问题包含层位或深度，应定位 LithologyLayer，并扩展岩性、特征和渗透率观测。")

    if "岩性大类" in entity_types or "岩性小类" in entity_types:
        directions.append("若问题包含岩性，应从 LithologyType 出发，检索同类分层和基础透水等级。")

    if "水力学参数" in hydro_factors:
        directions.append("涉及渗透率或透水性时，应优先关注 k_value、k_log10、k_unit、source、reliability。")

    if intent == "比较不同地层或岩性的渗透率":
        directions.append("比较类问题应分别召回不同岩性或层位证据并进行对比。")
    elif intent == "分析渗透率变化原因":
        directions.append("原因分析类问题应重点检索导致渗透率升高或降低的 HydroFeature。")
    elif intent == "预测或推断水文地质条件":
        directions.append("预测推断类问题应优先检索相似岩性、相似特征和带渗透率标签的样本。")
    elif intent == "判断某区域或某地层渗透性强弱":
        directions.append("强弱判断类问题应综合 LithologyType 倾向、HydroFeature 规则和 PermeabilityObservation。")

    return "".join(directions)