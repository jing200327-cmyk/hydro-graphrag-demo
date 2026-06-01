from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evals" / "datasets" / "rag_mvp_eval_50.jsonl"
DEFAULT_V2_SUMMARY = ROOT / "evals" / "reports_v2" / "retrieval_v2_rankgpt_off_summary.json"
DEFAULT_V2_DETAILS = ROOT / "evals" / "reports_v2" / "retrieval_v2_rankgpt_off_details.jsonl"
DEFAULT_V2_BAD_CASES = ROOT / "evals" / "reports_v2" / "retrieval_v2_rankgpt_off_bad_cases.jsonl"
DEFAULT_V3_SUMMARY = ROOT / "evals" / "reports_v3" / "retrieval_v3_rankgpt_off_summary.json"
DEFAULT_V3_DETAILS = ROOT / "evals" / "reports_v3" / "retrieval_v3_rankgpt_off_details.jsonl"
DEFAULT_V3_BAD_CASES = ROOT / "evals" / "reports_v3" / "retrieval_v3_rankgpt_off_bad_cases.jsonl"
DEFAULT_OUT_DIR = ROOT / "docs" / "eval_results"


CATEGORY_LABELS = {
    "fact_query": "事实查询",
    "permeability_level": "透水等级",
    "causal_explanation": "因果解释",
    "multi_condition": "多条件查询",
    "fallback_unanswerable": "不可回答",
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def delta_pct(v2: float, v3: float) -> str:
    return f"{(v3 - v2) * 100:+.1f} pp"


def delta_num(v2: float, v3: float) -> str:
    return f"{v3 - v2:+.3f}".rstrip("0").rstrip(".")


def group_details(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("question_type", ""))].append(row)
    return grouped


def mean_bool(rows: List[Dict[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    if not values:
        return None
    return sum(1 for value in values if bool(value)) / len(values)


def category_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped = group_details(rows)
    stats: Dict[str, Dict[str, Any]] = {}
    for category, items in grouped.items():
        stats[category] = {
            "count": len(items),
            "route_accuracy": mean_bool(items, "route_correct"),
            "keyword_hit_top10": mean_bool(items, "keyword_hit_top10"),
            "fallback_triggered": mean_bool(items, "fallback_triggered"),
            "retrieved_non_empty": sum(1 for row in items if int(row.get("retrieved_count") or 0) > 0) / len(items),
            "effective_non_empty": sum(1 for row in items if int(row.get("effective_rerank_count") or 0) > 0) / len(items),
        }
    return stats


def reason_counts(rows: Iterable[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        reasons = row.get("bad_case_reasons") or []
        if not reasons:
            if row.get("error"):
                reasons = ["error"]
            else:
                reasons = ["unspecified"]
        counts.update(str(reason) for reason in reasons)
    return counts


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_bar_chart(
    path: Path,
    title: str,
    metrics: List[Tuple[str, float, float]],
    unit: str = "%",
) -> None:
    width = 1400
    height = 220 + len(metrics) * 100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = find_font(34, bold=True)
    label_font = find_font(22)
    value_font = find_font(20, bold=True)
    small_font = find_font(18)

    draw.text((50, 35), title, fill="#111827", font=title_font)
    draw.rectangle((50, 90, 80, 112), fill="#9CA3AF")
    draw.text((88, 87), "v2", fill="#374151", font=small_font)
    draw.rectangle((150, 90, 180, 112), fill="#2563EB")
    draw.text((188, 87), "v3", fill="#374151", font=small_font)

    left = 360
    right = 1280
    bar_h = 24
    max_value = max([1.0] + [max(v2, v3) for _, v2, v3 in metrics])
    if unit == "%":
        max_value = max(max_value, 1.0)

    y = 150
    for label, v2, v3 in metrics:
        draw.text((50, y - 8), label, fill="#111827", font=label_font)
        for idx, (value, color) in enumerate([(v2, "#9CA3AF"), (v3, "#2563EB")]):
            by = y + idx * 34
            draw.rectangle((left, by, right, by + bar_h), fill="#F3F4F6")
            bar_w = int((right - left) * (value / max_value)) if max_value else 0
            draw.rectangle((left, by, left + bar_w, by + bar_h), fill=color)
            shown = f"{value * 100:.1f}%" if unit == "%" else f"{value:.3f}".rstrip("0").rstrip(".")
            draw.text((left + bar_w + 10, by - 2), shown, fill="#111827", font=value_font)
        y += 100

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(
    dataset: List[Dict[str, Any]],
    v2_summary: Dict[str, Any],
    v3_summary: Dict[str, Any],
    v2_details: List[Dict[str, Any]],
    v3_details: List[Dict[str, Any]],
    v2_bad_cases: List[Dict[str, Any]],
    v3_bad_cases: List[Dict[str, Any]],
) -> str:
    dataset_counts = Counter(str(row.get("category", "")) for row in dataset)
    v2_cat = category_stats(v2_details)
    v3_cat = category_stats(v3_details)

    overall_rows = [
        ["测评样本数", num(v2_summary["total"], 0), num(v3_summary["total"], 0), "持平"],
        ["成功执行数", num(v2_summary["success_count"], 0), num(v3_summary["success_count"], 0), "持平"],
        [
            "整体路由准确率",
            pct(v2_summary["overall"]["route_accuracy"]),
            pct(v3_summary["overall"]["route_accuracy"]),
            delta_pct(v2_summary["overall"]["route_accuracy"], v3_summary["overall"]["route_accuracy"]),
        ],
        [
            "事实查询 Top1 命中率",
            pct(v2_summary["fact_query"]["retrieval_hit_layer_top1"]),
            pct(v3_summary["fact_query"]["retrieval_hit_layer_top1"]),
            delta_pct(v2_summary["fact_query"]["retrieval_hit_layer_top1"], v3_summary["fact_query"]["retrieval_hit_layer_top1"]),
        ],
        [
            "非事实查询 keyword_hit_top10",
            pct(v2_summary["non_fact_query"]["keyword_hit_top10"]),
            pct(v3_summary["non_fact_query"]["keyword_hit_top10"]),
            delta_pct(v2_summary["non_fact_query"]["keyword_hit_top10"], v3_summary["non_fact_query"]["keyword_hit_top10"]),
        ],
        [
            "不可回答兜底触发率",
            pct(v2_summary["fallback"]["fallback_triggered_rate"]),
            pct(v3_summary["fallback"]["fallback_triggered_rate"]),
            delta_pct(v2_summary["fallback"]["fallback_triggered_rate"], v3_summary["fallback"]["fallback_triggered_rate"]),
        ],
        ["Badcase 数量", str(len(v2_bad_cases)), str(len(v3_bad_cases)), str(len(v3_bad_cases) - len(v2_bad_cases))],
        [
            "平均置信度",
            num(v2_summary["overall"]["avg_confidence"], 1),
            num(v3_summary["overall"]["avg_confidence"], 1),
            delta_num(v2_summary["overall"]["avg_confidence"], v3_summary["overall"]["avg_confidence"]),
        ],
        [
            "平均耗时",
            f"{v2_summary['overall']['avg_elapsed_sec']:.3f}s",
            f"{v3_summary['overall']['avg_elapsed_sec']:.3f}s",
            f"{v3_summary['overall']['avg_elapsed_sec'] - v2_summary['overall']['avg_elapsed_sec']:+.3f}s",
        ],
    ]

    category_order = [
        "fact_query",
        "permeability_level",
        "causal_explanation",
        "multi_condition",
        "fallback_unanswerable",
    ]
    category_rows: List[List[str]] = []
    for category in category_order:
        c2 = v2_cat.get(category, {})
        c3 = v3_cat.get(category, {})
        category_rows.append(
            [
                CATEGORY_LABELS.get(category, category),
                str(dataset_counts.get(category, 0)),
                pct(c2.get("keyword_hit_top10")),
                pct(c3.get("keyword_hit_top10")),
                pct(c2.get("fallback_triggered")),
                pct(c3.get("fallback_triggered")),
                pct(c3.get("route_accuracy")),
            ]
        )

    v2_reasons = reason_counts(v2_bad_cases)
    v3_reasons = reason_counts(v3_bad_cases)
    badcase_rows = [
        [reason, str(v2_reasons.get(reason, 0)), str(v3_reasons.get(reason, 0))]
        for reason, _ in v2_reasons.most_common()
    ]
    if not badcase_rows:
        badcase_rows = [["无", "0", "0"]]

    dataset_rows = [
        [CATEGORY_LABELS.get(category, category), str(dataset_counts.get(category, 0))]
        for category in category_order
    ]

    return "\n\n".join(
        [
            "## 可视化测评结果",
            "![Eval Cases](https://img.shields.io/badge/MVP%20Eval-50%20cases-blue) "
            "![Route Accuracy](https://img.shields.io/badge/Route%20Accuracy-100%25-brightgreen) "
            "![Fact Top1](https://img.shields.io/badge/Fact%20Top1-100%25-brightgreen) "
            "![Non Fact Keyword Hit](https://img.shields.io/badge/Non--fact%20Keyword%20Hit@10-93.3%25-brightgreen) "
            "![Fallback](https://img.shields.io/badge/Unanswerable%20Fallback-100%25-brightgreen)",
            "测评集共 50 条，覆盖事实查询、透水等级、因果解释、多条件查询和不可回答问题。v2 版本事实查询链路已经稳定，事实类 Top1 命中率保持 100%；主要短板集中在非事实查询关键词命中和不可回答问题兜底。v3 通过结构化路由、岩性类别精确检索、水文特征精确检索和兜底策略优化，将 non_fact keyword_hit_top10 从 50.0% 提升至 93.3%，不可回答问题兜底触发率从 20.0% 提升至 100.0%，Badcase 从 20 条降至 0 条。",
            "### 测评集构成",
            markdown_table(["类型", "数量"], dataset_rows),
            "### 整体指标",
            markdown_table(["指标", "v2", "v3", "变化"], overall_rows),
            "### 分类测评结果",
            markdown_table(["类型", "数量", "v2 Keyword Hit@10", "v3 Keyword Hit@10", "v2 兜底触发", "v3 兜底触发", "v3 路由准确率"], category_rows),
            "### 图表",
            "![v2/v3 overall metrics](docs/eval_results/eval_overview_v2_v3.png)\n\n"
            "![v2/v3 category keyword hit](docs/eval_results/eval_by_type_keyword_hit.png)",
            "### Badcase 摘要",
            markdown_table(["Badcase 原因", "v2", "v3"], badcase_rows),
            "v2 的 Badcase 主要来自非事实查询的关键词命中不足和不可回答问题未触发兜底；v3 中不可回答问题不再进入无依据检索链路，非事实查询通过结构化证据路由补强，当前测评集中 Badcase 为 0。",
            "统计脚本：`python scripts/summarize_eval_reports.py`",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--v2-summary", type=Path, default=DEFAULT_V2_SUMMARY)
    parser.add_argument("--v2-details", type=Path, default=DEFAULT_V2_DETAILS)
    parser.add_argument("--v2-bad-cases", type=Path, default=DEFAULT_V2_BAD_CASES)
    parser.add_argument("--v3-summary", type=Path, default=DEFAULT_V3_SUMMARY)
    parser.add_argument("--v3-details", type=Path, default=DEFAULT_V3_DETAILS)
    parser.add_argument("--v3-bad-cases", type=Path, default=DEFAULT_V3_BAD_CASES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    dataset = read_jsonl(args.dataset)
    v2_summary = read_json(args.v2_summary)
    v3_summary = read_json(args.v3_summary)
    v2_details = read_jsonl(args.v2_details)
    v3_details = read_jsonl(args.v3_details)
    v2_bad_cases = read_jsonl(args.v2_bad_cases)
    v3_bad_cases = read_jsonl(args.v3_bad_cases)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    draw_bar_chart(
        args.out_dir / "eval_overview_v2_v3.png",
        "v2 vs v3 Overall Metrics",
        [
            ("Route accuracy", v2_summary["overall"]["route_accuracy"], v3_summary["overall"]["route_accuracy"]),
            ("Fact Top1 hit", v2_summary["fact_query"]["retrieval_hit_layer_top1"], v3_summary["fact_query"]["retrieval_hit_layer_top1"]),
            ("Non-fact Keyword Hit@10", v2_summary["non_fact_query"]["keyword_hit_top10"], v3_summary["non_fact_query"]["keyword_hit_top10"]),
            ("Unanswerable fallback", v2_summary["fallback"]["fallback_triggered_rate"], v3_summary["fallback"]["fallback_triggered_rate"]),
        ],
    )

    v2_cat = category_stats(v2_details)
    v3_cat = category_stats(v3_details)
    draw_bar_chart(
        args.out_dir / "eval_by_type_keyword_hit.png",
        "Keyword Hit@10 by Question Type",
        [
            (CATEGORY_LABELS.get(category, category), v2_cat.get(category, {}).get("keyword_hit_top10") or 0.0, v3_cat.get(category, {}).get("keyword_hit_top10") or 0.0)
            for category in [
                "fact_query",
                "permeability_level",
                "causal_explanation",
                "multi_condition",
                "fallback_unanswerable",
            ]
        ],
    )

    markdown = build_markdown(
        dataset,
        v2_summary,
        v3_summary,
        v2_details,
        v3_details,
        v2_bad_cases,
        v3_bad_cases,
    )
    (args.out_dir / "eval_summary.md").write_text(markdown + "\n", encoding="utf-8")

    print(f"Wrote {args.out_dir / 'eval_summary.md'}")
    print(f"Wrote {args.out_dir / 'eval_overview_v2_v3.png'}")
    print(f"Wrote {args.out_dir / 'eval_by_type_keyword_hit.png'}")


if __name__ == "__main__":
    main()
