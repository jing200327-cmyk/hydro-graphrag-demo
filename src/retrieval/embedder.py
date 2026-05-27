from __future__ import annotations

from typing import List

from configs.settings import EMBEDDING_QUERY_PREFIX, NORMALIZE_EMBEDDINGS
from src.utils.env import get_embedding_model
from src.utils.text import safe_text


def encode_query(user_question: str) -> List[float]:
    text = safe_text(user_question)

    if not text:
        raise ValueError("user_question 不能为空。")

    if EMBEDDING_QUERY_PREFIX:
        text = EMBEDDING_QUERY_PREFIX + text

    model = get_embedding_model()

    query_embedding = model.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    )

    query_embedding = query_embedding.astype("float32")[0]

    return [float(x) for x in query_embedding.tolist()]