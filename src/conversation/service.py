from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.conversation.history_builder import build_history_context
from src.conversation.summarizer import summarize_turns
from src.conversation.store import ConversationStore
from src.utils.text import make_json_safe, safe_text, truncate_text


DEFAULT_PROJECT_NAME = "hydro_graphrag_streamlit"

CHUNK_PROP_KEYS = [
    "borehole_id",
    "layer_id",
    "top_depth",
    "bottom_depth",
    "thickness",
    "mid",
    "lithology_major_v2",
    "lithology_minor_v3",
    "geologic_period",
    "geologic_epoch",
    "strat_group",
    "strat_member",
    "k_value",
    "k_log10",
    "k_unit",
    "source",
    "reliability",
    "feature_type",
    "feature_value",
    "permeability_effect",
    "level_name",
]

SECRET_KEYWORDS = [
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
]

PERSISTED_ARTIFACT_TYPES = {
    "retrieval_snapshot",
    "question_analysis",
    "retrieved_chunks",
    "graph_context",
    "b_score_results",
    "c_rerank_results",
    "effective_rerank_results",
    "llm_usage",
    "entity_guard",
    "history_context",
    "query_rewrite",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def dumps_json(value: Any) -> str:
    return json.dumps(make_json_safe(value), ensure_ascii=False, default=str)


def loads_json(value: Any, default: Any = None) -> Any:
    text = safe_text(value)
    if not text:
        return default
    return json.loads(text)


def _contains_secret_key(key: Any) -> bool:
    lowered = safe_text(key).lower()
    return any(keyword in lowered for keyword in SECRET_KEYWORDS)


def _drop_secret_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _drop_secret_keys(item)
            for key, item in value.items()
            if not _contains_secret_key(key)
        }

    if isinstance(value, list):
        return [_drop_secret_keys(item) for item in value]

    return value


def _compact_props(props: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: props.get(key)
        for key in CHUNK_PROP_KEYS
        if key in props and props.get(key) is not None
    }


def _compact_chunk(item: Dict[str, Any]) -> Dict[str, Any]:
    source_props = item.get("source_props") or {}

    return _drop_secret_keys(
        {
            "chunk_id": item.get("chunk_id"),
            "chunk_type": item.get("chunk_type"),
            "source_label": item.get("source_label"),
            "source_id": item.get("source_id"),
            "retrieval_method": item.get("retrieval_method"),
            "vector_score": item.get("vector_score"),
            "exact_score": item.get("exact_score"),
            "final_score": item.get("final_score"),
            "rank": item.get("rank"),
            "matched_fields": item.get("matched_fields"),
            "chunk_text": truncate_text(item.get("chunk_text"), 800),
            "graph_context_text": truncate_text(item.get("graph_context_text"), 1200),
            "source_props": _compact_props(source_props) if isinstance(source_props, dict) else {},
        }
    )


def _compact_score_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return _drop_secret_keys(
        {
            "chunk_id": item.get("chunk_id"),
            "source_id": item.get("source_id"),
            "source_label": item.get("source_label"),
            "rank": item.get("rank"),
            "score": item.get("score"),
            "final_score": item.get("final_score"),
            "b_score": item.get("b_score"),
            "c_score": item.get("c_score"),
            "use_for_context": item.get("use_for_context"),
            "matched_fields": item.get("matched_fields"),
            "matched_rules": item.get("matched_rules"),
            "rerank_reason": truncate_text(item.get("rerank_reason"), 500),
            "reason": truncate_text(item.get("reason"), 500),
        }
    )


def _compact_graph_context(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact_rows: List[Dict[str, Any]] = []

    for item in rows:
        compact_rows.append(
            _drop_secret_keys(
                {
                    "chunk_id": item.get("chunk_id"),
                    "graph_context_text": truncate_text(item.get("graph_context_text"), 1200),
                }
            )
        )

    return compact_rows


def build_persisted_artifacts(
    result: Dict[str, Any],
    history_context: Dict[str, Any],
    query_rewrite: Dict[str, Any],
) -> Dict[str, Any]:
    retrieved_chunks = result.get("vector_search_results") or []
    b_score_results = result.get("b_score_results") or []
    c_rerank_results = result.get("c_rerank_results") or []
    effective_rerank_results = result.get("effective_rerank_results") or []

    retrieval_snapshot = {
        "question_analysis": result.get("question_analysis") or {},
        "question_intent": result.get("question_intent"),
        "retrieval_route": result.get("retrieval_route"),
        "rerank_route": result.get("rerank_route"),
        "confidence": result.get("confidence"),
        "fallback_triggered": result.get("fallback_triggered"),
        "fallback_reason": result.get("fallback_reason"),
        "retrieved_chunks": [_compact_chunk(item) for item in retrieved_chunks[:50]],
        "graph_context": _compact_graph_context(result.get("graph_context") or []),
    }

    return {
        "retrieval_snapshot": _drop_secret_keys(retrieval_snapshot),
        "question_analysis": _drop_secret_keys(result.get("question_analysis") or {}),
        "retrieved_chunks": [_compact_chunk(item) for item in retrieved_chunks[:50]],
        "graph_context": _compact_graph_context(result.get("graph_context") or []),
        "b_score_results": [_compact_score_item(item) for item in b_score_results[:50]],
        "c_rerank_results": [_compact_score_item(item) for item in c_rerank_results[:50]],
        "effective_rerank_results": [
            _compact_score_item(item) for item in effective_rerank_results[:50]
        ],
        "llm_usage": _drop_secret_keys(result.get("deepseek_usage") or {}),
        "entity_guard": _drop_secret_keys(result.get("entity_guard") or {}),
        "history_context": _drop_secret_keys(history_context or {}),
        "query_rewrite": _drop_secret_keys(query_rewrite or {}),
    }


class ConversationService:
    def __init__(self, store: Optional[ConversationStore] = None) -> None:
        self.store = store or ConversationStore()
        self.store.initialize_schema()

    def create_conversation(
        self,
        title: str = "",
        user_id: str = "",
        project_name: str = DEFAULT_PROJECT_NAME,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        conversation_id = make_id("conv")
        now = utc_now_iso()
        clean_title = safe_text(title) or "新会话"

        with self.store.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (
                    conversation_id, title, created_at, updated_at, user_id,
                    project_name, active, summary, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    clean_title,
                    now,
                    now,
                    safe_text(user_id),
                    safe_text(project_name),
                    1,
                    "",
                    dumps_json(metadata or {}),
                ),
            )
            conn.commit()

        return conversation_id

    def ensure_conversation(self, conversation_id: str | None = None) -> str:
        conversation_id = safe_text(conversation_id)

        if conversation_id and self.conversation_exists(conversation_id):
            return conversation_id

        return self.create_conversation()

    def conversation_exists(self, conversation_id: str) -> bool:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return row is not None

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        with self.store.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()

        return dict(row) if row else {}

    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        with self.store.connect() as conn:
            conn.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (safe_text(title) or "新会话", utc_now_iso(), conversation_id),
            )
            conn.commit()

    def delete_conversation(self, conversation_id: str) -> None:
        with self.store.connect() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.commit()

    def list_conversations(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.conversation_id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    c.user_id,
                    c.project_name,
                    c.active,
                    c.summary,
                    COUNT(t.turn_id) AS turn_count
                FROM conversations c
                LEFT JOIN conversation_turns t
                    ON t.conversation_id = c.conversation_id
                WHERE c.active = 1
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        return [dict(row) for row in rows]

    def load_turns(
        self,
        conversation_id: str,
        limit: int = 50,
        include_retrieval_snapshot: bool = True,
    ) -> List[Dict[str, Any]]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY turn_index ASC
                LIMIT ?
                """,
                (conversation_id, int(limit)),
            ).fetchall()

            turns = [dict(row) for row in rows]

            if include_retrieval_snapshot:
                for turn in turns:
                    snapshot = self._load_artifact_payload(
                        conn,
                        turn["turn_id"],
                        "retrieval_snapshot",
                    )
                    turn["retrieval_snapshot"] = snapshot or {}

        return turns

    def load_messages(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        turns = self.load_turns(
            conversation_id=conversation_id,
            limit=limit,
            include_retrieval_snapshot=False,
        )
        messages: List[Dict[str, Any]] = []

        for turn in turns:
            messages.append(
                {
                    "role": "user",
                    "content": safe_text(turn.get("user_question")),
                    "turn_id": turn.get("turn_id"),
                    "turn_index": turn.get("turn_index"),
                    "created_at": turn.get("created_at"),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": safe_text(turn.get("final_answer")),
                    "turn_id": turn.get("turn_id"),
                    "turn_index": turn.get("turn_index"),
                    "created_at": turn.get("created_at"),
                    "retrieval_route": turn.get("retrieval_route"),
                    "confidence": turn.get("confidence"),
                }
            )

        return messages

    def export_conversation(self, conversation_id: str) -> Dict[str, Any]:
        with self.store.connect() as conn:
            conversation = conn.execute(
                """
                SELECT *
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()

            turns = conn.execute(
                """
                SELECT *
                FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY turn_index ASC
                """,
                (conversation_id,),
            ).fetchall()

            turn_ids = [row["turn_id"] for row in turns]
            artifacts_by_turn: Dict[str, List[Dict[str, Any]]] = {}
            feedback_by_turn: Dict[str, List[Dict[str, Any]]] = {}

            for turn_id in turn_ids:
                artifact_rows = conn.execute(
                    """
                    SELECT artifact_id, turn_id, artifact_type, payload_json, created_at
                    FROM turn_artifacts
                    WHERE turn_id = ?
                      AND artifact_type IN (
                        'retrieval_snapshot',
                        'question_analysis',
                        'retrieved_chunks',
                        'graph_context',
                        'b_score_results',
                        'c_rerank_results',
                        'effective_rerank_results',
                        'llm_usage',
                        'entity_guard',
                        'history_context',
                        'query_rewrite'
                      )
                    ORDER BY artifact_type ASC
                    """,
                    (turn_id,),
                ).fetchall()
                artifacts_by_turn[turn_id] = [
                    {
                        "artifact_id": row["artifact_id"],
                        "turn_id": row["turn_id"],
                        "artifact_type": row["artifact_type"],
                        "payload": loads_json(row["payload_json"], default={}),
                        "created_at": row["created_at"],
                    }
                    for row in artifact_rows
                ]

                feedback_rows = conn.execute(
                    """
                    SELECT *
                    FROM conversation_feedback
                    WHERE turn_id = ?
                    ORDER BY created_at ASC
                    """,
                    (turn_id,),
                ).fetchall()
                feedback_by_turn[turn_id] = [
                    {
                        **dict(row),
                        "metadata": loads_json(row["metadata_json"], default={}),
                    }
                    for row in feedback_rows
                ]

        turn_payloads = []
        for row in turns:
            turn = dict(row)
            turn["metadata"] = loads_json(turn.pop("metadata_json", ""), default={})
            turn["artifacts"] = artifacts_by_turn.get(turn["turn_id"], [])
            turn["feedback"] = feedback_by_turn.get(turn["turn_id"], [])
            turn_payloads.append(turn)

        conversation_payload = dict(conversation) if conversation else {}
        if conversation_payload:
            conversation_payload["metadata"] = loads_json(
                conversation_payload.pop("metadata_json", ""),
                default={},
            )

        return {
            "conversation": conversation_payload,
            "turns": turn_payloads,
        }

    def export_conversation_json(self, conversation_id: str) -> str:
        return json.dumps(
            make_json_safe(self.export_conversation(conversation_id)),
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    def export_conversation_markdown(self, conversation_id: str) -> str:
        payload = self.export_conversation(conversation_id)
        conversation = payload.get("conversation") or {}
        turns = payload.get("turns") or []

        lines = [
            f"# {safe_text(conversation.get('title')) or '会话导出'}",
            "",
            f"- conversation_id: `{safe_text(conversation.get('conversation_id'))}`",
            f"- created_at: `{safe_text(conversation.get('created_at'))}`",
            f"- updated_at: `{safe_text(conversation.get('updated_at'))}`",
            "",
        ]

        summary = safe_text(conversation.get("summary"))
        if summary:
            lines.extend(["## 会话摘要", "", summary, ""])

        lines.append("## 对话记录")
        lines.append("")

        for turn in turns:
            lines.extend(
                [
                    f"### 第 {turn.get('turn_index')} 轮",
                    "",
                    "**用户问题**",
                    "",
                    safe_text(turn.get("user_question")),
                    "",
                    "**系统回答**",
                    "",
                    safe_text(turn.get("final_answer")),
                    "",
                    "**检索信息**",
                    "",
                    f"- rewritten_question: `{safe_text(turn.get('rewritten_question'))}`",
                    f"- question_intent: `{safe_text(turn.get('question_intent'))}`",
                    f"- retrieval_route: `{safe_text(turn.get('retrieval_route'))}`",
                    f"- rerank_route: `{safe_text(turn.get('rerank_route'))}`",
                    f"- confidence: `{safe_text(turn.get('confidence'))}`",
                    f"- fallback_triggered: `{safe_text(turn.get('fallback_triggered'))}`",
                    "",
                ]
            )

            feedback = turn.get("feedback") or []
            if feedback:
                lines.append("**反馈**")
                lines.append("")
                for item in feedback:
                    lines.append(
                        f"- rating={item.get('rating')}, tag={safe_text(item.get('tag'))}, comment={safe_text(item.get('comment'))}"
                    )
                lines.append("")

        return "\n".join(lines).strip() + "\n"

    def build_history_context(self, conversation_id: str, max_turns: int = 5) -> Dict[str, Any]:
        with self.store.connect() as conn:
            conv = conn.execute(
                "SELECT summary FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()

        summary = safe_text(conv["summary"]) if conv else ""
        turns = self.load_turns(
            conversation_id=conversation_id,
            limit=max_turns,
            include_retrieval_snapshot=True,
        )

        context = build_history_context(
            turns=turns,
            summary=summary,
            max_turns=max_turns,
        )
        context["conversation_id"] = conversation_id
        return context

    def append_turn(
        self,
        conversation_id: str,
        user_question: str,
        result: Dict[str, Any],
        rewritten_question: str = "",
        query_rewrite: Optional[Dict[str, Any]] = None,
        history_context: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[int] = None,
    ) -> str:
        conversation_id = self.ensure_conversation(conversation_id)
        turn_id = make_id("turn")
        now = utc_now_iso()
        title = truncate_text(user_question, 40)

        question_analysis = result.get("question_analysis") or {}
        turn_metadata = {
            "original_user_question": user_question,
            "query_rewrite": query_rewrite or {},
        }

        with self.store.connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(turn_index), 0) + 1 AS next_index
                FROM conversation_turns
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            turn_index = int(row["next_index"] if row else 1)

            conn.execute(
                """
                INSERT INTO conversation_turns (
                    turn_id, conversation_id, turn_index, user_question,
                    rewritten_question, question_intent, retrieval_route,
                    rerank_route, final_answer, confidence, fallback_triggered,
                    fallback_reason, created_at, latency_ms, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    conversation_id,
                    turn_index,
                    safe_text(user_question),
                    safe_text(rewritten_question),
                    safe_text(result.get("question_intent") or question_analysis.get("问题意图")),
                    safe_text(result.get("retrieval_route")),
                    safe_text(result.get("rerank_route")),
                    safe_text(result.get("final_answer")),
                    result.get("confidence"),
                    self._bool_to_int(result.get("fallback_triggered")),
                    safe_text(result.get("fallback_reason")),
                    now,
                    latency_ms,
                    dumps_json(turn_metadata),
                ),
            )

            artifact_payloads = build_persisted_artifacts(
                result=result,
                history_context=history_context or {},
                query_rewrite=query_rewrite or {},
            )

            for artifact_type, payload in artifact_payloads.items():
                self._insert_artifact(conn, turn_id, artifact_type, payload, now)

            conn.execute(
                """
                UPDATE conversations
                SET
                    updated_at = ?,
                    title = CASE
                        WHEN title = '' OR title = '新会话' THEN ?
                        ELSE title
                    END
                WHERE conversation_id = ?
                """,
                (now, title, conversation_id),
            )
            conn.commit()

        self.refresh_summary(conversation_id)
        return turn_id

    def refresh_summary(self, conversation_id: str) -> str:
        turns = self.load_turns(
            conversation_id=conversation_id,
            limit=20,
            include_retrieval_snapshot=False,
        )
        summary = summarize_turns(turns)

        with self.store.connect() as conn:
            conn.execute(
                """
                UPDATE conversations
                SET summary = ?
                WHERE conversation_id = ?
                """,
                (summary, conversation_id),
            )
            conn.commit()

        return summary

    def add_feedback(
        self,
        turn_id: str,
        rating: Optional[int] = None,
        tag: str = "",
        comment: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        feedback_id = make_id("feedback")

        with self.store.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_feedback (
                    feedback_id, turn_id, rating, tag, comment, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    turn_id,
                    rating,
                    safe_text(tag),
                    safe_text(comment),
                    utc_now_iso(),
                    dumps_json(metadata or {}),
                ),
            )
            conn.commit()

        return feedback_id

    def _insert_artifact(
        self,
        conn: Any,
        turn_id: str,
        artifact_type: str,
        payload: Any,
        created_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO turn_artifacts (
                artifact_id, turn_id, artifact_type, payload_json, created_at
            ) VALUES (
                COALESCE(
                    (
                        SELECT artifact_id
                        FROM turn_artifacts
                        WHERE turn_id = ? AND artifact_type = ?
                    ),
                    ?
                ),
                ?, ?, ?, ?
            )
            """,
            (
                turn_id,
                artifact_type,
                make_id("artifact"),
                turn_id,
                artifact_type,
                dumps_json(payload),
                created_at,
            ),
        )

    def _load_artifact_payload(self, conn: Any, turn_id: str, artifact_type: str) -> Any:
        row = conn.execute(
            """
            SELECT payload_json
            FROM turn_artifacts
            WHERE turn_id = ? AND artifact_type = ?
            """,
            (turn_id, artifact_type),
        ).fetchone()

        if not row:
            return None

        return loads_json(row["payload_json"], default=None)

    @staticmethod
    def _bool_to_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        return 1 if bool(value) else 0


def timed_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)
