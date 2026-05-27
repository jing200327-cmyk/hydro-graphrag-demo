import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_step(name, fn):
    print(f"\n[CHECK] {name}")
    try:
        result = fn()
        if result is not None:
            print(result)
        print(f"[OK] {name}")
    except Exception as exc:
        print(f"[FAILED] {name}")
        print(type(exc).__name__, exc)
        raise


def main():
    print("Python:", sys.executable)
    print("PROJECT_ROOT:", PROJECT_ROOT)

    check_step(
        "settings import",
        lambda: __import__("configs.settings"),
    )

    check_step(
        "text utils import",
        lambda: __import__("src.utils.text"),
    )

    def check_model_path():
        from src.utils.env import resolve_embedding_model_path
        return resolve_embedding_model_path()

    check_step("embedding model path", check_model_path)

    def check_neo4j():
        from src.utils.env import get_driver
        get_driver()
        return "neo4j connected"

    check_step("neo4j connection", check_neo4j)

    def check_embedding():
        from src.retrieval.embedder import encode_query
        v = encode_query("中砂层为什么通常比粉质黏土透水性强？")
        return f"embedding dim={len(v)}"

    check_step("embedding encode", check_embedding)

    def check_query_parser():
        from src.query.parser import analyze_user_question
        r = analyze_user_question("中砂层为什么通常比粉质黏土透水性强？")
        return json.dumps(r, ensure_ascii=False, indent=2)

    check_step("query parser", check_query_parser)

    def check_pipeline_no_llm():
        from src.pipeline.qa_pipeline import run_end_to_end_graphrag_qa

        r = run_end_to_end_graphrag_qa(
            "中砂层为什么通常比粉质黏土透水性强？",
            raw_top_k=10,
            final_top_k=5,
            save_outputs=False,
            enable_llm=False,
        )

        return {
            "chunks": len(r.get("vector_search_results", [])),
            "confidence": r.get("confidence"),
            "fallback_triggered": r.get("fallback_triggered"),
        }

    check_step("qa pipeline without llm", check_pipeline_no_llm)

    print("\nAll startup checks passed.")


if __name__ == "__main__":
    main()