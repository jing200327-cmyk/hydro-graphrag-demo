from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class Entity(TypedDict, total=False):
    实体类型: str
    实体值: str
    来源: str


class QuestionAnalysis(TypedDict, total=False):
    核心实体: List[Entity]
    问题意图: str
    检索关键词: List[str]
    需要关注的水文地质因素: List[str]
    建议检索方向: str


class RetrievedChunk(TypedDict, total=False):
    chunk_id: str
    chunk_type: str
    chunk_text: str
    source_label: str
    source_id: str
    vector_score: float
    graph_context: Dict[str, Any]
    graph_context_text: str


class BScoreResult(TypedDict, total=False):
    chunk_id: str
    score: float
    matched_entities: List[str]
    matched_permeability_factors: List[str]
    keep: bool
    reason: str


class CRerankResult(TypedDict, total=False):
    chunk_id: str
    final_score: float
    semantic_score: float
    rule_score: float
    evidence_score: float
    matched_rules: List[str]
    conflict_or_uncertainty: str
    rerank_reason: str
    use_for_context: bool


class QAResult(TypedDict, total=False):
    user_question: str
    question_analysis: QuestionAnalysis
    vector_search_results: List[RetrievedChunk]
    graph_expansion_results: Dict[str, Any]
    b_score_results: List[BScoreResult]
    c_rerank_results: List[CRerankResult]
    final_prompt: str
    final_answer: str
    deepseek_usage: Optional[Dict[str, Any]]
    confidence: Optional[float]
    fallback_triggered: Optional[bool]