from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


PROJECT_DIR = Path(
    os.getenv("HYDRO_GRAPHRAG_PROJECT_DIR", Path(__file__).resolve().parents[1])
).resolve()

ENV_PATH = PROJECT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)
else:
    found_env = find_dotenv(usecwd=True)
    if found_env:
        load_dotenv(found_env, override=True)


def env(name: str, default: str = "") -> str:
    value = os.getenv(name)

    if value is None or str(value).strip() == "":
        return default

    return str(value).strip()


def required_env(name: str) -> str:
    value = env(name)

    if not value:
        raise ValueError(f"缺少必需环境变量：{name}。请检查 {ENV_PATH} 或当前环境变量。")

    return value


NEO4J_URI = env("NEO4J_URI")
NEO4J_USERNAME = env("NEO4J_USERNAME")
NEO4J_PASSWORD = env("NEO4J_PASSWORD")
NEO4J_DATABASE = env("NEO4J_DATABASE", "neo4j")

VECTOR_INDEX_NAME = (
    env("VECTOR_INDEX_NAME")
    or env("NEO4J_VECTOR_INDEX")
    or "hydro_lithology_vector_index"
)

EMBEDDING_MODEL_PATH = env("EMBEDDING_MODEL_PATH", "./models/bge-small-zh-v1.5")
EMBEDDING_QUERY_PREFIX = env("EMBEDDING_QUERY_PREFIX", "")
NORMALIZE_EMBEDDINGS = env("NORMALIZE_EMBEDDINGS", "true").lower() not in {
    "0",
    "false",
    "no",
}

DEEPSEEK_API_KEY = env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = env("DEEPSEEK_MODEL", "deepseek-chat")


OUTPUT_DIR = Path(
    env("HYDRO_GRAPHRAG_OUTPUT_DIR", str(PROJECT_DIR / "outputs" / "run_logs"))
).resolve()