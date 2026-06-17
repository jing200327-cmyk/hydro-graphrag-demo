from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from configs.settings import CONVERSATION_DB_PATH


CONVERSATION_SCHEMA_VERSION = 1


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    user_id TEXT NOT NULL DEFAULT '',
    project_name TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    summary TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    turn_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    user_question TEXT NOT NULL,
    rewritten_question TEXT NOT NULL DEFAULT '',
    question_intent TEXT NOT NULL DEFAULT '',
    retrieval_route TEXT NOT NULL DEFAULT '',
    rerank_route TEXT NOT NULL DEFAULT '',
    final_answer TEXT NOT NULL DEFAULT '',
    confidence REAL,
    fallback_triggered INTEGER CHECK (fallback_triggered IS NULL OR fallback_triggered IN (0, 1)),
    fallback_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    latency_ms INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (conversation_id)
        REFERENCES conversations(conversation_id)
        ON DELETE CASCADE,
    UNIQUE (conversation_id, turn_index)
);

CREATE TABLE IF NOT EXISTS turn_artifacts (
    artifact_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (turn_id)
        REFERENCES conversation_turns(turn_id)
        ON DELETE CASCADE,
    UNIQUE (turn_id, artifact_type)
);

CREATE TABLE IF NOT EXISTS conversation_feedback (
    feedback_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    rating INTEGER CHECK (rating IS NULL OR rating BETWEEN -1 AND 5),
    tag TEXT NOT NULL DEFAULT '',
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (turn_id)
        REFERENCES conversation_turns(turn_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
    ON conversations(updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_active_updated_at
    ON conversations(active, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_conversation
    ON conversation_turns(conversation_id, turn_index);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_created_at
    ON conversation_turns(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_turn_artifacts_turn
    ON turn_artifacts(turn_id, artifact_type);

CREATE INDEX IF NOT EXISTS idx_turn_artifacts_type
    ON turn_artifacts(artifact_type);

CREATE INDEX IF NOT EXISTS idx_conversation_feedback_turn
    ON conversation_feedback(turn_id);

CREATE INDEX IF NOT EXISTS idx_conversation_feedback_rating
    ON conversation_feedback(rating);

PRAGMA user_version = 1;
"""


class ConversationStore:
    """
    SQLite storage for persistent RAG conversation history.

    The first version only owns schema initialization. Higher-level services can
    build on this class for append/load/export flows without duplicating SQL
    connection setup.
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path or CONVERSATION_DB_PATH).resolve()

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def initialize_schema(self) -> Path:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        return self.db_path


def initialize_conversation_db(db_path: Optional[Path | str] = None) -> Path:
    return ConversationStore(db_path=db_path).initialize_schema()
