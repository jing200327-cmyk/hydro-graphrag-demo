from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.settings import (
    PROJECT_DIR,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    EMBEDDING_MODEL_PATH,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
)


_DRIVER = None
_EMBEDDING_MODEL = None
_DEEPSEEK_CLIENT = None


def get_driver():
    global _DRIVER

    if _DRIVER is not None:
        return _DRIVER

    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        raise ValueError(
            "Neo4j 环境变量不完整。需要 NEO4J_URI、NEO4J_USERNAME、NEO4J_PASSWORD。"
        )

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise ImportError(
            "缺少 neo4j 包。请在当前 Python 环境中安装：pip install neo4j"
        ) from exc

    _DRIVER = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    _DRIVER.verify_connectivity()

    return _DRIVER


def resolve_embedding_model_path() -> Path:
    path = Path(EMBEDDING_MODEL_PATH).expanduser()

    if not path.is_absolute():
        path = PROJECT_DIR / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"本地 embedding 模型路径不存在：{path}\n"
            "请检查 .env 中的 EMBEDDING_MODEL_PATH。"
        )

    return path


def get_embedding_model():
    global _EMBEDDING_MODEL

    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "缺少 sentence-transformers 包。请在当前 Python 环境中安装：pip install sentence-transformers"
        ) from exc

    model_path = resolve_embedding_model_path()

    _EMBEDDING_MODEL = SentenceTransformer(
        str(model_path),
        trust_remote_code=False,
    )

    return _EMBEDDING_MODEL


def get_deepseek_client() -> Any:
    global _DEEPSEEK_CLIENT

    if _DEEPSEEK_CLIENT is not None:
        return _DEEPSEEK_CLIENT

    if not DEEPSEEK_API_KEY:
        raise ValueError("缺少 DEEPSEEK_API_KEY，请在 .env 中配置。")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "缺少 openai 包。请在当前 Python 环境中安装：pip install openai"
        ) from exc

    _DEEPSEEK_CLIENT = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    return _DEEPSEEK_CLIENT


def close_resources() -> None:
    global _DRIVER

    if _DRIVER is not None:
        _DRIVER.close()
        _DRIVER = None