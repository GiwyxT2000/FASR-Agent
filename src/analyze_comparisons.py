from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import NOTES_DIR, PROCESSED_DIR, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two retrieval prediction files and export case studies.")
    parser.add_argument("--baseline", required=True, help="Baseline prediction JSONL.")
    parser.add_argument("--method", required=True, help="Method prediction JSONL.")
    parser.add_argument("--name", default="bm25_vs_method")
    parser.add_argument("--limit-per-category", type=int, default=30)
    return parser.parse_args()


def load_prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["query_id"]): row for row in read_jsonl(path)}


def top_doc(row: dict[str, Any]) -> str:
    ranked = row.get("ranked", [])
    return str(ranked[0]["doc_id"]) if ranked else ""


def rank_of(row: dict[str, Any], doc_id: str, cutoff: int = 50) -> int | None:
    for idx, item in enumerate(row.get("ranked", [])[:cutoff], start=1):
        if str(item["doc_id"]) == str(doc_id):
            return idx
    return None


def preview_report(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report.get("metadata", {})
    if isinstance(metadata, dict):
        metadata_preview = {k: metadata[k] for k in list(metadata)[:12]}
    else:
        metadata_preview = {}
    return {
        "doc_id": report.get("doc_id", ""),
        "synopsis": report.get("synopsis", "")[:700],
        "metadata": metadata_preview,
        "text_preview": report.get("text_raw", "")[:1000],
    }


def build_case(
    query: dict[str, Any],
    baseline: dict[str, Any],
    method: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    gold = str(query["gold_doc_id"])
    b_top = top_doc(baseline)
    m_top = top_doc(method)
    return {
        "category": category,
        "query_id": query["query_id"],
        "query": query["query"],
        "gold_doc_id": gold,
        "baseline_top1": b_top,
        "method_top1": m_top,
        "baseline_gold_rank": rank_of(baseline, gold),
        "method_gold_rank": rank_of(method, gold),
        "gold_report": preview_report(reports.get(gold, {})),
        "baseline_top1_report": preview_report(reports.get(b_top, {})),
        "method_top1_report": preview_report(reports.get(m_top, {})),
        "method_route_features": method.get("route_features", {}),
        "method_top_ranked": method.get("ranked", [])[:5],
        "baseline_top_ranked": baseline.get("ranked", [])[:5],
    }


def case_to_markdown(case: dict[str, Any]) -> str:
    lines = [
        f"### {case['category']} | {case['query_id']}",
        "",
        f"- Query: {case['query']}",
        f"- Gold: `{case['gold_doc_id']}`",
        f"- Baseline top-1: `{case['baseline_top1']}`; gold rank: {case['baseline_gold_rank']}",
        f"- Method top-1: `{case['method_top1']}`; gold rank: {case['method_gold_rank']}",
        "",
        "**Gold synopsis**",
        "",
        case["gold_report"].get("synopsis", "") or "(empty)",
        "",
        "**Baseline top-1 synopsis**",
        "",
        case["baseline_top1_report"].get("synopsis", "") or "(empty)",
        "",
        "**Method top-1 synopsis**",
        "",
        case["method_top1_report"].get("synopsis", "") or "(empty)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline = load_prediction_map(Path(args.baseline))
    method = load_prediction_map(Path(args.method))
    queries = {str(row["query_id"]): row for row in read_jsonl(PROCESSED_DIR / "queries_test.jsonl")}
    reports = {str(row["doc_id"]): row for row in read_jsonl(PROCESSED_DIR / "reports.jsonl")}

    categories: dict[str, list[dict[str, Any]]] = {
        "baseline_wrong_method_correct": [],
        "baseline_correct_method_wrong": [],
        "both_correct": [],
        "both_wrong_method_improves_rank": [],
        "both_wrong": [],
    }

    for query_id in sorted(set(baseline) & set(method) & set(queries)):
        query = queries[query_id]
        gold = str(query["gold_doc_id"])
        b = baseline[query_id]
        m = method[query_id]
        b_top = top_doc(b)
        m_top = top_doc(m)
        b_rank = rank_of(b, gold)
        m_rank = rank_of(m, gold)
        if b_top != gold and m_top == gold:
            category = "baseline_wrong_method_correct"
        elif b_top == gold and m_top != gold:
            category = "baseline_correct_method_wrong"
        elif b_top == gold and m_top == gold:
            category = "both_correct"
        elif b_top != gold and m_top != gold and m_rank is not None and (b_rank is None or m_rank < b_rank):
            category = "both_wrong_method_improves_rank"
        else:
            category = "both_wrong"
        if len(categories[category]) < args.limit_per_category:
            categories[category].append(build_case(query, b, m, reports, category))

    out_dir = NOTES_DIR / "case_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.name}.json"
    md_path = out_dir / f"{args.name}.md"
    write_json(json_path, categories)

    lines = [f"# Case Studies: {args.name}", ""]
    for category, cases in categories.items():
        lines.extend([f"## {category}", ""])
        for case in cases:
            lines.append(case_to_markdown(case))
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {category: len(cases) for category, cases in categories.items()}
    print(json.dumps({"summary": summary, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

