from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Conversation:
    conversation_id: str
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    user_id: str = ""
    project_name: str = ""
    active: bool = True
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    turn_id: str
    conversation_id: str
    turn_index: int
    user_question: str
    rewritten_question: str = ""
    question_intent: str = ""
    retrieval_route: str = ""
    rerank_route: str = ""
    final_answer: str = ""
    confidence: Optional[float] = None
    fallback_triggered: Optional[bool] = None
    fallback_reason: str = ""
    created_at: str = ""
    latency_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnArtifact:
    artifact_id: str
    turn_id: str
    artifact_type: str
    payload: Dict[str, Any] | List[Any] | str | None = None
    created_at: str = ""


@dataclass
class ConversationFeedback:
    feedback_id: str
    turn_id: str
    rating: Optional[int] = None
    tag: str = ""
    comment: str = ""
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
