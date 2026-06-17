# Changelog

## 2026-06-17

- Added local SQLite conversation persistence with `conversations`, `conversation_turns`, `turn_artifacts`, and `conversation_feedback`.
- Added `ConversationService` for conversation creation, message loading, turn persistence, feedback capture, and JSON/Markdown export.
- Connected conversation history to the RAG call chain through deterministic query rewriting and compact history context injection.
- Updated final prompt construction so history is used only for coreference and follow-up understanding, while factual answers still depend on current retrieved evidence.
- Updated Streamlit sidebar with conversation management, history selection, deletion, export, and the stratigraphy example question.
- Added per-answer feedback controls for helpful/inaccurate ratings and badcase attribution tags.
- Added SQLite initialization script and ignored local conversation database files.
