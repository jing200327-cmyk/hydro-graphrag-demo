# Changelog

## 2026-06-23

- Added `evals/scripts/build_stratigraphy_eval_v4.py` to build a 50-case v4 eval dataset without overwriting the original MVP dataset.
- Added `evals/scripts/analyze_stratigraphy_v4.py` to compute Stratigraphy route hit rate, layer-level recall/precision, answer field completeness, hallucination checks, regression count, and Badcase reports.
- Added `evals/datasets/rag_mvp_eval_50_stratigraphy_v4.jsonl`, replacing 10 of the first 15 fact-query cases with real Neo4j-backed stratigraphy fact queries across period, epoch, group, and member dimensions.
- Added v4 Stratigraphy evaluation outputs under `evals/results/v4_stratigraphy/`.
- Documented v4 results in `README.md`: route hit rate 10/10, retrieval recall 82.86%, retrieval precision 100%, answer accuracy 0%, no-hallucination rate 100%, and 0 regressions in the remaining 40 cases.
- Recorded current Badcase findings: large-result stratigraphy queries are capped by the 200-row retrieval limit, and the direct Markdown answer needs explicit table columns for layer ID, depth, thickness, stratigraphy fields, and lithology fields.

## 2026-06-17

- Added local SQLite conversation persistence with `conversations`, `conversation_turns`, `turn_artifacts`, and `conversation_feedback`.
- Added `ConversationService` for conversation creation, message loading, turn persistence, feedback capture, and JSON/Markdown export.
- Connected conversation history to the RAG call chain through deterministic query rewriting and compact history context injection.
- Updated final prompt construction so history is used only for coreference and follow-up understanding, while factual answers still depend on current retrieved evidence.
- Updated Streamlit sidebar with conversation management, history selection, deletion, export, and the stratigraphy example question.
- Added per-answer feedback controls for helpful/inaccurate ratings and badcase attribution tags.
- Added SQLite initialization script and ignored local conversation database files.
