from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple


CURRENT_FILE = Path(__file__).resolve()
PROJECT_DIR = CURRENT_FILE.parents[2]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.retrieval.stratigraphy_retriever import (  # noqa: E402
    STRATIGRAPHY_ROUTE,
    _load_stratigraphy_vocab,
    is_stratigraphy_query,
    retrieve_stratigraphy_exact,
)
from src.utils.env import get_driver  # noqa: E402
from src.utils.text import make_json_safe, safe_text, unique_keep_order  # noqa: E402
from configs.settings import NEO4J_DATABASE  # noqa: E402


DEFAULT_INPUT = PROJECT_DIR / "evals" / "datasets" / "rag_mvp_eval_50.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "evals" / "datasets" / "rag_mvp_eval_50_stratigraphy_v4.jsonl"
DEFAULT_CASES_OUTPUT = PROJECT_DIR / "evals" / "results" / "v4_stratigraphy" / "v4_stratigraphy_cases.json"

REPLACEMENT_PLAN = [
    ("strat_member", "列出{term}对应的钻孔及分层信息。"),
    ("strat_member", "{term}有哪些钻孔？请给出对应层位和分层信息。"),
    ("strat_member", "给出{term}在各钻孔中的分布情况和层位信息。"),
    ("strat_member", "列出{term}相关的所有钻孔、层位和分层信息。"),
    ("strat_group", "给出{term}在各钻孔中的分布情况和层位信息。"),
    ("strat_group", "列出{term}对应的钻孔及分层信息。"),
    ("geologic_epoch", "{term}有哪些钻孔层位记录？请列出对应分层信息。"),
    ("geologic_epoch", "给出{term}对应的所有钻孔、层位和分层信息。"),
    ("geologic_period", "列出{term}对应的钻孔及分层信息。"),
    ("geologic_period", "给出{term}对应的所有钻孔、层位和分层信息。"),
]

DIMENSION_LABELS = {
    "geologic_period": "地质纪",
    "geologic_epoch": "地质世",
    "strat_group": "地层组",
    "strat_member": "地层段",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_no} 行不是 JSON 对象。")
            rows.append(row)
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_safe(row), ensure_ascii=False) + "\n")


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(value), f, ensure_ascii=False, indent=2)


def row_to_ground_truth(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "borehole_id": row.get("borehole_id"),
        "x": row.get("x"),
        "y": row.get("y"),
        "ground_elevation": row.get("ground_elevation"),
        "layer_id": row.get("layer_id"),
        "top_depth": row.get("top_depth"),
        "bottom_depth": row.get("bottom_depth"),
        "thickness": row.get("thickness"),
        "geologic_period": row.get("geologic_period"),
        "geologic_epoch": row.get("geologic_epoch"),
        "strat_group": row.get("strat_group"),
        "strat_member": row.get("strat_member"),
        "lithology_major_v2": row.get("lithology_major_v2"),
        "lithology_minor_v3": row.get("lithology_minor_v3"),
        "chunk_id": row.get("chunk_id"),
        "source_id": row.get("source_id"),
    }


def query_ground_truth(field: str, term: str) -> List[Dict[str, Any]]:
    params = {
        "geologic_period": [term] if field == "geologic_period" else [],
        "geologic_epoch": [term] if field == "geologic_epoch" else [],
        "strat_group": [term] if field == "strat_group" else [],
        "strat_member": [term] if field == "strat_member" else [],
    }
    cypher = """
    MATCH (b:Borehole)-[:HAS_LAYER]->(l:LithologyLayer)
    WHERE
        (size($geologic_period) = 0 OR toString(l.geologic_period) IN $geologic_period)
        AND (size($geologic_epoch) = 0 OR toString(l.geologic_epoch) IN $geologic_epoch)
        AND (size($strat_group) = 0 OR toString(l.strat_group) IN $strat_group)
        AND (size($strat_member) = 0 OR toString(l.strat_member) IN $strat_member)
    OPTIONAL MATCH (l)-[:HAS_EVIDENCE_CHUNK]->(chunk:EvidenceChunk)
    WITH b, l, head(collect(chunk)) AS chunk
    RETURN
        toString(b.borehole_id) AS borehole_id,
        b.x AS x,
        b.y AS y,
        b.ground_elevation AS ground_elevation,
        toString(l.layer_id) AS layer_id,
        l.top_depth AS top_depth,
        l.bottom_depth AS bottom_depth,
        l.thickness AS thickness,
        l.lithology_major_v2 AS lithology_major_v2,
        l.lithology_minor_v3 AS lithology_minor_v3,
        l.geologic_period AS geologic_period,
        l.geologic_epoch AS geologic_epoch,
        l.strat_group AS strat_group,
        l.strat_member AS strat_member,
        toString(coalesce(chunk.chunk_id, '')) AS chunk_id,
        'Layer:' + toString(l.layer_id) AS source_id
    ORDER BY borehole_id, toFloat(l.top_depth), layer_id
    """
    driver = get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        return [row_to_ground_truth(record) for record in session.run(cypher, params).data()]


def build_question(field: str, term: str, template: str) -> str:
    question = template.format(term=term)
    if not is_stratigraphy_query(question):
        raise ValueError(f"生成的问题未命中 Stratigraphy 路由：{question}")
    return question


def select_terms() -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    vocab = _load_stratigraphy_vocab()
    used_by_field: Dict[str, int] = {}
    selected: List[Tuple[str, str, List[Dict[str, Any]]]] = []

    for field, template in REPLACEMENT_PLAN:
        candidates: List[Tuple[str, List[Dict[str, Any]]]] = []
        for term in vocab.get(field, []):
            term = safe_text(term)
            if not term:
                continue

            question = build_question(field, term, template)
            chunks = retrieve_stratigraphy_exact(question, top_k=500)
            if chunks:
                candidates.append((term, chunks))

        if not candidates:
            raise RuntimeError(
                f"无法从 Neo4j 词表中找到可返回结果的 {field} 地层名称。"
            )

        index = used_by_field.get(field, 0) % len(candidates)
        term, chunks = candidates[index]
        used_by_field[field] = used_by_field.get(field, 0) + 1
        selected.append((field, term, chunks))

    return selected


def compact_expected_keywords(term: str, ground_truth: List[Dict[str, Any]]) -> List[str]:
    first = ground_truth[0]
    values = [
        term,
        first.get("borehole_id"),
        first.get("layer_id"),
        first.get("strat_group"),
        first.get("strat_member"),
        first.get("geologic_epoch"),
        first.get("geologic_period"),
    ]
    return unique_keep_order([safe_text(v) for v in values if safe_text(v)])


def build_replacement_row(
    original: Dict[str, Any],
    field: str,
    term: str,
    template: str,
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, Any]:
    row = deepcopy(original)
    original_question = safe_text(original.get("question"))
    question = build_question(field, term, template)
    layer_ids = unique_keep_order([item.get("layer_id") for item in ground_truth])
    borehole_ids = unique_keep_order([item.get("borehole_id") for item in ground_truth])
    chunk_ids = unique_keep_order([item.get("chunk_id") for item in ground_truth])

    row["category"] = "fact_query"
    row["question"] = question
    row["answerable"] = True
    row["expected_answer"] = (
        f"应返回 {term} 对应的 {len(layer_ids)} 条钻孔分层记录，"
        f"覆盖 {len(borehole_ids)} 个钻孔；完整结构化 ground truth 见 "
        "metadata.stratigraphy_ground_truth。"
    )
    row["expected_keywords"] = compact_expected_keywords(term, ground_truth)
    row["expected_entities"] = unique_keep_order([term] + borehole_ids[:10] + layer_ids[:10])
    row["expected_sources"] = chunk_ids[:10]
    row["expected_answer_points"] = [
        "成功进入 Stratigraphy 确定性地层事实查询路由",
        "返回 ground truth 中全部钻孔编号和层位 ID",
        "返回顶板深度、底板深度、厚度、纪、世、组、段字段",
        "返回岩性大类和岩性小类",
        "不得返回不属于该地层条件的额外层位",
    ]
    row["difficulty"] = "medium"

    metadata = deepcopy(row.get("metadata") or {})
    metadata.update(
        {
            "route": "stratigraphy",
            "expected_route": STRATIGRAPHY_ROUTE,
            "subtype": "stratigraphy_fact_query",
            "stratigraphy_dimension": field,
            "stratigraphy_dimension_label": DIMENSION_LABELS[field],
            "stratigraphy_term": term,
            "ground_truth_count": len(layer_ids),
            "ground_truth_borehole_count": len(borehole_ids),
            "ground_truth_layer_ids": layer_ids,
            "ground_truth_borehole_ids": borehole_ids,
            "stratigraphy_ground_truth": ground_truth,
            "replaced_from_question": original_question,
        }
    )
    row["metadata"] = metadata
    return row


def build_dataset(input_path: Path, output_path: Path, cases_output_path: Path) -> Dict[str, Any]:
    rows = load_jsonl(input_path)
    if len(rows) != 50:
        raise ValueError(f"原始测评集应为 50 条，实际为 {len(rows)} 条：{input_path}")

    fact_rows = rows[:15]
    non_fact_categories = [safe_text(row.get("category")) for row in fact_rows]
    if any(category != "fact_query" for category in non_fact_categories):
        raise ValueError("原始测评集前 15 条并非全部为 fact_query，停止替换。")

    selected = select_terms()
    output_rows = deepcopy(rows)
    case_rows: List[Dict[str, Any]] = []

    for index, ((field, template), (_, term, chunks)) in enumerate(
        zip(REPLACEMENT_PLAN, selected),
        start=0,
    ):
        original = rows[index]
        ground_truth = query_ground_truth(field=field, term=term)
        replacement = build_replacement_row(
            original=original,
            field=field,
            term=term,
            template=template,
            ground_truth=ground_truth,
        )
        output_rows[index] = replacement
        case_rows.append(
            {
                "case_id": replacement.get("id"),
                "original_question": original.get("question"),
                "new_question": replacement.get("question"),
                "stratigraphy_dimension": field,
                "stratigraphy_term": term,
                "expected_route": STRATIGRAPHY_ROUTE,
                "ground_truth_count": len(
                    replacement["metadata"]["ground_truth_layer_ids"]
                ),
                "ground_truth_borehole_count": len(
                    replacement["metadata"]["ground_truth_borehole_ids"]
                ),
            }
        )

    save_jsonl(output_rows, output_path)
    save_json(
        {
            "dataset_path": str(output_path),
            "replaced_count": len(case_rows),
            "kept_fact_control_ids": [row.get("id") for row in output_rows[10:15]],
            "total_count": len(output_rows),
            "cases": case_rows,
        },
        cases_output_path,
    )

    return {
        "dataset_path": str(output_path),
        "cases_output_path": str(cases_output_path),
        "total_count": len(output_rows),
        "replaced_count": len(case_rows),
        "cases": case_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v4 Stratigraphy eval dataset from the existing 50-case JSONL.",
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="原始 JSONL 测评集。")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="v4 JSONL 输出路径。")
    parser.add_argument(
        "--cases-output",
        default=str(DEFAULT_CASES_OUTPUT),
        help="新增 Stratigraphy case 清单输出路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        input_path=Path(args.input).resolve(),
        output_path=Path(args.output).resolve(),
        cases_output_path=Path(args.cases_output).resolve(),
    )
    print(json.dumps(make_json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
