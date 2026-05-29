from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from configs.settings import DEEPSEEK_MODEL
from src.utils.env import get_deepseek_client
from src.utils.text import safe_text, make_json_safe


# ============================================================
# 1. RankGPT Prompt
# ============================================================

RANKGPT_SYSTEM_PROMPT = """
你是一个用于 RAG 系统的候选证据排序器，类似 RankGPT。

你的任务不是回答问题，而是根据用户问题，对候选 chunk 重新排序。

你需要判断：
1. 哪个 chunk 最能直接回答用户问题。
2. 哪些 chunk 是同钻孔但非目标分层。
3. 哪些 chunk 只是语义相似，但不是答案证据。
4. 哪些 chunk 虽然规则分高，但实际不能支撑答案。
5. 哪些 chunk 包含明确的结构化约束，例如 layer_id、borehole_id、深度范围、岩性字段。

排序原则：
- 如果问题中出现明确 layer_id，例如 CHGC001_2，包含该 layer_id 的 chunk 必须优先。
- 同一钻孔但不是目标 layer_id 的 chunk 应该降级。
- 只语义相似但没有目标实体、目标字段或答案证据的 chunk 应该降级。
- 对事实查询，精确 ID、深度、岩性字段优先于泛泛语义相似。
- 对非事实查询，优先选择能解释原因、规则、关系、证据链的 chunk。
- 不要编造候选列表之外的 chunk_id。
- 不要修改 chunk_id。
- 不要输出 Markdown。
- 不要输出解释性前后缀。
- 只输出 JSON。

输出 JSON 格式固定为：
{
  "ranked_chunk_ids": [
    "chunk_id_1",
    "chunk_id_2"
  ],
  "reason": "简要说明排序依据"
}
""".strip()


RANKGPT_USER_PROMPT_TEMPLATE = """
用户问题：
{question}

问题解析信息：
{question_analysis}

候选 chunk 列表：
{candidates_json}

请只根据候选 chunk 的内容和元数据进行排序。

你必须输出 JSON：
{{
  "ranked_chunk_ids": [
    "候选中的 chunk_id"
  ],
  "reason": "排序依据"
}}
""".strip()


# ============================================================
# 2. 配置
# ============================================================

DEFAULT_RANKGPT_TOP_K = 10
DEFAULT_MAX_CHUNK_CHARS = 900
DEFAULT_MAX_REASON_CHARS = 500


# ============================================================
# 3. 通用工具函数
# ============================================================

def _truncate_text(text: Any, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> str:
    value = safe_text(text)

    if max_chars <= 0:
        return value

    if len(value) <= max_chars:
        return value

    return value[:max_chars] + "..."


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _unique_keep_order(values: List[str]) -> List[str]:
    seen = set()
    results: List[str] = []

    for value in values:
        text = safe_text(value)

        if not text:
            continue

        if text in seen:
            continue

        seen.add(text)
        results.append(text)

    return results


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """
    尽量从 LLM 返回中解析 JSON。

    兼容：
    1. 纯 JSON
    2. ```json ... ```
    3. 前后带少量说明文字的 JSON
    """

    content = safe_text(text).strip()

    if not content:
        return None

    # 去掉 Markdown fence
    content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    # 直接解析
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 从文本中截取第一个 {...}
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)

    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def _extract_llm_text(response: Any) -> str:
    """
    兼容 OpenAI-compatible client 的返回对象。
    """

    try:
        return safe_text(response.choices[0].message.content)
    except Exception:
        pass

    try:
        return safe_text(response["choices"][0]["message"]["content"])
    except Exception:
        pass

    return safe_text(response)


# ============================================================
# 4. 候选构造
# ============================================================

def _build_chunk_map(
    retrieved_chunks: Optional[List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    chunk_map: Dict[str, Dict[str, Any]] = {}

    if not retrieved_chunks:
        return chunk_map

    for chunk in retrieved_chunks:
        chunk_id = safe_text(chunk.get("chunk_id", ""))

        if not chunk_id:
            continue

        chunk_map[chunk_id] = chunk

    return chunk_map


def _merge_rerank_item_with_chunk(
    rerank_item: Dict[str, Any],
    chunk_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    合并 C rerank / fact rerank 结果与原始 retrieved chunk。

    rerank_item 通常包含：
    - chunk_id
    - final_score
    - rerank_reason

    retrieved chunk 通常包含：
    - chunk_text
    - source_id
    - source_label
    - graph_context_text
    """

    chunk_id = safe_text(rerank_item.get("chunk_id", ""))
    source_chunk = chunk_map.get(chunk_id, {})

    merged = dict(source_chunk)
    merged.update(rerank_item)

    # 如果 rerank_item 没有 chunk_text，则从原始 retrieved chunk 补充
    if not safe_text(merged.get("chunk_text", "")):
        merged["chunk_text"] = source_chunk.get("chunk_text", "")

    if not safe_text(merged.get("graph_context_text", "")):
        merged["graph_context_text"] = source_chunk.get("graph_context_text", "")

    return merged


def build_rankgpt_candidates(
    reranked_results: List[Dict[str, Any]],
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    candidate_top_k: int = DEFAULT_RANKGPT_TOP_K,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> List[Dict[str, Any]]:
    """
    构造 RankGPT 输入候选。

    输入应该是规则重排后的 Top10，而不是原始全量召回。

    返回示例：
    [
        {
            "rank": 1,
            "chunk_id": "...",
            "chunk_text": "...",
            "rule_final_score": 92.5,
            "source_id": "...",
            "exact_match_value": "...",
            "rerank_reason": "..."
        }
    ]
    """

    if not reranked_results:
        return []

    chunk_map = _build_chunk_map(retrieved_chunks)

    candidates: List[Dict[str, Any]] = []

    for idx, item in enumerate(reranked_results[:candidate_top_k], start=1):
        merged = _merge_rerank_item_with_chunk(item, chunk_map)

        chunk_id = safe_text(merged.get("chunk_id", ""))

        if not chunk_id:
            continue

        chunk_text_parts = [
            safe_text(merged.get("chunk_text", "")),
            safe_text(merged.get("graph_context_text", "")),
        ]

        chunk_text = "\n".join([x for x in chunk_text_parts if x])

        candidate = {
            "rank": idx,
            "chunk_id": chunk_id,
            "chunk_text": _truncate_text(chunk_text, max_chunk_chars),
            "rule_final_score": merged.get("final_score", 0.0),
            "rule_semantic_score": merged.get("semantic_score", 0.0),
            "rule_score": merged.get("rule_score", 0.0),
            "evidence_score": merged.get("evidence_score", 0.0),
            "rerank_route": merged.get("rerank_route", ""),
            "scoring_method": merged.get("scoring_method", ""),
            "matched_rules": merged.get("matched_rules", []),
            "matched_fields": merged.get("matched_fields", []),
            "source_id": merged.get("source_id", ""),
            "source_label": merged.get("source_label", ""),
            "chunk_type": merged.get("chunk_type", ""),
            "retrieval_method": merged.get("retrieval_method", ""),
            "exact_match_type": merged.get("exact_match_type", ""),
            "exact_match_value": merged.get("exact_match_value", ""),
            "rerank_reason": _truncate_text(
                merged.get("rerank_reason", ""),
                DEFAULT_MAX_REASON_CHARS,
            ),
        }

        candidates.append(make_json_safe(candidate))

    return candidates


# ============================================================
# 5. Prompt 构造
# ============================================================

def build_rankgpt_messages(
    user_question: str,
    candidates: List[Dict[str, Any]],
    question_analysis: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    构造 RankGPT messages。
    """

    question_analysis_text = json.dumps(
        make_json_safe(question_analysis or {}),
        ensure_ascii=False,
        indent=2,
    )

    candidates_json = json.dumps(
        make_json_safe(candidates),
        ensure_ascii=False,
        indent=2,
    )

    user_prompt = RANKGPT_USER_PROMPT_TEMPLATE.format(
        question=safe_text(user_question),
        question_analysis=question_analysis_text,
        candidates_json=candidates_json,
    )

    return [
        {
            "role": "system",
            "content": RANKGPT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


# ============================================================
# 6. LLM 调用与 JSON 解析
# ============================================================

def call_rankgpt(
    user_question: str,
    candidates: List[Dict[str, Any]],
    question_analysis: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> Dict[str, Any]:
    """
    调用 DeepSeek / OpenAI-compatible LLM 进行排序。

    返回标准结构：
    {
        "ranked_chunk_ids": [...],
        "reason": "...",
        "raw_response": "...",
        "success": true
    }
    """

    if not candidates:
        return {
            "ranked_chunk_ids": [],
            "reason": "候选列表为空，未调用 RankGPT。",
            "raw_response": "",
            "success": False,
        }

    client = get_deepseek_client()
    used_model = model or DEEPSEEK_MODEL

    messages = build_rankgpt_messages(
        user_question=user_question,
        candidates=candidates,
        question_analysis=question_analysis,
    )

    response = client.chat.completions.create(
        model=used_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    raw_text = _extract_llm_text(response)
    parsed = _safe_json_loads(raw_text)

    if not parsed:
        return {
            "ranked_chunk_ids": [],
            "reason": "RankGPT 返回内容无法解析为 JSON。",
            "raw_response": raw_text,
            "success": False,
        }

    ranked_chunk_ids = parsed.get("ranked_chunk_ids", [])
    reason = safe_text(parsed.get("reason", ""))

    if not isinstance(ranked_chunk_ids, list):
        ranked_chunk_ids = []

    ranked_chunk_ids = [
        safe_text(item)
        for item in ranked_chunk_ids
        if safe_text(item)
    ]

    return {
        "ranked_chunk_ids": ranked_chunk_ids,
        "reason": reason,
        "raw_response": raw_text,
        "success": True,
    }


# ============================================================
# 7. 排序结果校验与应用
# ============================================================

def validate_rankgpt_ids(
    ranked_chunk_ids: List[str],
    candidates: List[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """
    校验 LLM 返回的 chunk_id。

    返回：
    - valid_ids: 确实存在于候选中的 ID
    - invalid_ids: LLM 编造或不存在的 ID
    """

    candidate_ids = [
        safe_text(candidate.get("chunk_id", ""))
        for candidate in candidates
        if safe_text(candidate.get("chunk_id", ""))
    ]

    candidate_id_set = set(candidate_ids)

    valid_ids: List[str] = []
    invalid_ids: List[str] = []

    for chunk_id in ranked_chunk_ids:
        cid = safe_text(chunk_id)

        if not cid:
            continue

        if cid in candidate_id_set:
            valid_ids.append(cid)
        else:
            invalid_ids.append(cid)

    valid_ids = _unique_keep_order(valid_ids)
    invalid_ids = _unique_keep_order(invalid_ids)

    return valid_ids, invalid_ids


def _append_missing_candidate_ids(
    valid_ranked_ids: List[str],
    candidates: List[Dict[str, Any]],
) -> List[str]:
    """
    如果 LLM 返回 ID 不完整，则按原规则顺序补齐剩余候选。
    """

    result = list(valid_ranked_ids)
    seen = set(result)

    for candidate in candidates:
        cid = safe_text(candidate.get("chunk_id", ""))

        if not cid:
            continue

        if cid in seen:
            continue

        result.append(cid)
        seen.add(cid)

    return result


def apply_rankgpt_order(
    reranked_results: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    rankgpt_output: Dict[str, Any],
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    top_k: int = DEFAULT_RANKGPT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    将 RankGPT 输出顺序应用回 reranked_results。

    注意：
    - 保留原 rerank 字段。
    - 新增 rankgpt_rank / rankgpt_reason / rankgpt_used 等字段。
    """

    if not reranked_results:
        return []

    chunk_map = _build_chunk_map(retrieved_chunks)

    result_by_id: Dict[str, Dict[str, Any]] = {}

    for item in reranked_results:
        chunk_id = safe_text(item.get("chunk_id", ""))

        if not chunk_id:
            continue

        merged = _merge_rerank_item_with_chunk(item, chunk_map)
        result_by_id[chunk_id] = merged

    ranked_chunk_ids = rankgpt_output.get("ranked_chunk_ids", [])
    valid_ids, invalid_ids = validate_rankgpt_ids(ranked_chunk_ids, candidates)

    final_ids = _append_missing_candidate_ids(valid_ids, candidates)

    reason = safe_text(rankgpt_output.get("reason", ""))
    success = bool(rankgpt_output.get("success", False))

    output: List[Dict[str, Any]] = []

    for idx, chunk_id in enumerate(final_ids[:top_k], start=1):
        item = dict(result_by_id.get(chunk_id, {}))

        if not item:
            continue

        original_rank = item.get("rank", item.get("fact_rank", item.get("c_rank")))
        original_final_score = item.get("final_score", 0.0)

        item.update(
            {
                "rankgpt_rank": idx,
                "rankgpt_reason": reason,
                "rankgpt_used": success,
                "rankgpt_invalid_ids": invalid_ids,
                "rankgpt_raw_response": rankgpt_output.get("raw_response", ""),
                "rank_before_rankgpt": original_rank,
                "score_before_rankgpt": original_final_score,
                "rank": idx,
                "use_for_context": item.get("use_for_context", True),
            }
        )

        output.append(item)

    return output


def fallback_rankgpt_order(
    reranked_results: List[Dict[str, Any]],
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    reason: str = "RankGPT 未启用或调用失败，保留规则重排顺序。",
    top_k: int = DEFAULT_RANKGPT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    RankGPT 失败时保留原规则顺序。
    """

    chunk_map = _build_chunk_map(retrieved_chunks)

    output: List[Dict[str, Any]] = []

    for idx, item in enumerate(reranked_results[:top_k], start=1):
        merged = _merge_rerank_item_with_chunk(item, chunk_map)

        original_rank = merged.get("rank", merged.get("fact_rank", merged.get("c_rank")))

        merged.update(
            {
                "rankgpt_rank": idx,
                "rankgpt_reason": reason,
                "rankgpt_used": False,
                "rankgpt_invalid_ids": [],
                "rankgpt_raw_response": "",
                "rank_before_rankgpt": original_rank,
                "score_before_rankgpt": merged.get("final_score", 0.0),
                "rank": idx,
            }
        )

        output.append(merged)

    return output


# ============================================================
# 8. 对外主入口
# ============================================================

def rankgpt_rerank_chunks(
    user_question: str,
    reranked_results: List[Dict[str, Any]],
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    question_analysis: Optional[Dict[str, Any]] = None,
    candidate_top_k: int = DEFAULT_RANKGPT_TOP_K,
    final_top_k: int = DEFAULT_RANKGPT_TOP_K,
    enabled: bool = True,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    fail_open: bool = True,
) -> List[Dict[str, Any]]:
    """
    RankGPT rerank 主入口。

    适用位置：
    Fact Rule Rerank / C Rule Rerank
        ↓
    rankgpt_rerank_chunks()
        ↓
    Top10 Context
        ↓
    Final Answer Generation

    参数说明：
    - user_question: 用户问题
    - reranked_results: 规则重排后的结果，通常是 C rerank 或 fact rerank 的 Top10
    - retrieved_chunks: 原始检索 chunk，用于补充 chunk_text / graph_context_text
    - question_analysis: Query Translation 结果
    - candidate_top_k: 输入给 RankGPT 的候选数量，建议 10
    - final_top_k: 最终输出数量，建议 10
    - enabled: 是否启用 RankGPT
    - fail_open: LLM 失败时是否保留原排序。生产环境建议 True。

    返回：
    - 与 reranked_results 兼容的列表
    - 新增 rankgpt_rank / rankgpt_reason / rankgpt_used 等字段
    """

    if not reranked_results:
        return []

    if not enabled:
        return fallback_rankgpt_order(
            reranked_results=reranked_results,
            retrieved_chunks=retrieved_chunks,
            reason="RankGPT disabled，保留规则重排顺序。",
            top_k=final_top_k,
        )

    candidates = build_rankgpt_candidates(
        reranked_results=reranked_results,
        retrieved_chunks=retrieved_chunks,
        candidate_top_k=candidate_top_k,
        max_chunk_chars=max_chunk_chars,
    )

    if not candidates:
        return fallback_rankgpt_order(
            reranked_results=reranked_results,
            retrieved_chunks=retrieved_chunks,
            reason="RankGPT 候选为空，保留规则重排顺序。",
            top_k=final_top_k,
        )

    try:
        rankgpt_output = call_rankgpt(
            user_question=user_question,
            candidates=candidates,
            question_analysis=question_analysis,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        ranked_chunk_ids = rankgpt_output.get("ranked_chunk_ids", [])

        valid_ids, _invalid_ids = validate_rankgpt_ids(
            ranked_chunk_ids=ranked_chunk_ids,
            candidates=candidates,
        )

        if not valid_ids:
            return fallback_rankgpt_order(
                reranked_results=reranked_results,
                retrieved_chunks=retrieved_chunks,
                reason="RankGPT 未返回有效 chunk_id，保留规则重排顺序。",
                top_k=final_top_k,
            )

        return apply_rankgpt_order(
            reranked_results=reranked_results,
            candidates=candidates,
            rankgpt_output=rankgpt_output,
            retrieved_chunks=retrieved_chunks,
            top_k=final_top_k,
        )

    except Exception as exc:
        if not fail_open:
            raise

        return fallback_rankgpt_order(
            reranked_results=reranked_results,
            retrieved_chunks=retrieved_chunks,
            reason=f"RankGPT 调用失败，保留规则重排顺序。错误信息：{repr(exc)}",
            top_k=final_top_k,
        )


# ============================================================
# 9. 别名函数
# ============================================================

def gpt_rerank_chunks(
    user_question: str,
    reranked_results: List[Dict[str, Any]],
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    question_analysis: Optional[Dict[str, Any]] = None,
    candidate_top_k: int = DEFAULT_RANKGPT_TOP_K,
    final_top_k: int = DEFAULT_RANKGPT_TOP_K,
    enabled: bool = True,
) -> List[Dict[str, Any]]:
    """
    rankgpt_rerank_chunks 的短别名。
    """

    return rankgpt_rerank_chunks(
        user_question=user_question,
        reranked_results=reranked_results,
        retrieved_chunks=retrieved_chunks,
        question_analysis=question_analysis,
        candidate_top_k=candidate_top_k,
        final_top_k=final_top_k,
        enabled=enabled,
    )