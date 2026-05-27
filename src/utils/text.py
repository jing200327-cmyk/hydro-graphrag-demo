from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd


def safe_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def make_json_safe(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (str, int, bool)):
        return obj

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    if isinstance(obj, pd.Series):
        return obj.to_dict()

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    return str(obj)


def unique_keep_order(items: Optional[Iterable[Any]]) -> List[str]:
    if items is None:
        return []

    seen = set()
    result: List[str] = []

    for item in items:
        text = safe_text(item)

        if not text:
            continue

        if text not in seen:
            seen.add(text)
            result.append(text)

    return result


def contains_any(text: Any, keywords: Optional[Iterable[Any]]) -> bool:
    text = safe_text(text)

    if not text or not keywords:
        return False

    return any(
        safe_text(keyword) in text
        for keyword in keywords
        if safe_text(keyword)
    )


def find_hits(text: Any, keywords: Optional[Iterable[Any]]) -> List[str]:
    text = safe_text(text)

    if not text or not keywords:
        return []

    hits = [
        safe_text(keyword)
        for keyword in keywords
        if safe_text(keyword) and safe_text(keyword) in text
    ]

    return unique_keep_order(hits)


def clamp(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    try:
        number = float(value)
    except Exception:
        return low

    if math.isnan(number) or math.isinf(number):
        return low

    return max(low, min(high, number))


def coerce_score_to_100(value: Any) -> float:
    if value is None:
        return 0.0

    try:
        number = float(value)
    except Exception:
        return 0.0

    if math.isnan(number) or math.isinf(number):
        return 0.0

    if 0 <= number <= 1:
        return round(number * 100.0, 2)

    return round(clamp(number, 0.0, 100.0), 2)


def truncate_text(text: Any, max_chars: int = 1000) -> str:
    text = safe_text(text)

    if max_chars <= 0:
        return ""

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "...[已截断]"