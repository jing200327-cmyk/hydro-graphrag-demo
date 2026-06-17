from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.conversation import CONVERSATION_SCHEMA_VERSION, initialize_conversation_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize the local SQLite database for RAG conversation history.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Optional SQLite database path. Defaults to HYDRO_GRAPHRAG_CONVERSATION_DB_PATH.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = initialize_conversation_db(args.db_path)

    with sqlite3.connect(db_path) as conn:
        user_version = conn.execute("PRAGMA user_version;").fetchone()[0]
        tables = [
            row[0]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name IN (
                    'conversations',
                    'conversation_turns',
                    'turn_artifacts',
                    'conversation_feedback'
                  )
                ORDER BY name
                """
            ).fetchall()
        ]

    print(f"conversation_db_path={db_path}")
    print(f"schema_version={user_version}")
    print(f"expected_schema_version={CONVERSATION_SCHEMA_VERSION}")
    print("tables=" + ", ".join(tables))


if __name__ == "__main__":
    main()
