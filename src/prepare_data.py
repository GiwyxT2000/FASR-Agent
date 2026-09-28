from __future__ import annotations

import ast
import csv
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .common import (
    NOTES_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    compact_text,
    ensure_dirs,
    normalize_id,
    write_json,
    write_jsonl,
)


REPORT_ID_FIELDS = [
    "acn_num_ACN",
    "ACN",
    "id",
    "Report Number",
    "Person 1.10_ASRS Report Number.Accession Number",
]

SYNOPSIS_FIELDS = [
    "Report 1.2_Synopsis",
    "Synopsis",
    "synopsis",
]

NARRATIVE_FIELDS = [
    "Report 1_Narrative",
    "Report 2_Narrative",
    "narrative",
    "Narrative",
]

METADATA_HINTS = [
    "Time_Date",
    "Place_Locale Reference",
    "Place.1_State Reference",
    "Environment_Flight Conditions",
    "Environment.1_Weather Elements / Visibility",
    "Environment.3_Light",
    "Aircraft 1.2_Make Model Name",
    "Aircraft 1.9_Flight Phase",
    "Aircraft 2.2_Make Model Name",
    "Aircraft 2.9_Flight Phase",
    "Person 1.7_Human Factors",
    "Person 1.8_Communication Breakdown",
    "Events_Anomaly",
    "Events.5_Result",
    "Assessments_Contributing Factors / Situations",
    "Assessments.1_Primary Problem",
]


def find_raw_files(group: str, suffix: str) -> list[Path]:
    root = RAW_DIR / group
    if not root.exists():
        return []
    return sorted(path for path in root.rglob(f"*{suffix}") if path.is_file())


def first_text(row: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = compact_text(row.get(field))
        if value:
            return value
    return ""


def join_texts(row: dict[str, Any], fields: list[str]) -> str:
    parts = []
    for field in fields:
        value = compact_text(row.get(field))
        if value:
            parts.append(value)
    return " ".join(parts)


def load_report_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    jsonl_files = find_raw_files("reports", ".jsonl")
    corpus_files = [path for path in find_raw_files("queries", ".csv") if path.name == "corpus.csv"]
    files = [*jsonl_files, *corpus_files]
    reports: list[dict[str, Any]] = []
    skipped_no_id = 0
    duplicate_ids = 0
    seen: set[str] = set()
    field_counter: Counter[str] = Counter()

    for path in jsonl_files:
        split = split_from_filename(path.name)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                field_counter.update(raw.keys())
                doc_id = ""
                for field in REPORT_ID_FIELDS:
                    doc_id = normalize_id(raw.get(field))
                    if doc_id:
                        break
                if not doc_id:
                    skipped_no_id += 1
                    continue
                if doc_id in seen:
                    duplicate_ids += 1
                    continue
                seen.add(doc_id)
                synopsis = first_text(raw, SYNOPSIS_FIELDS)
                narrative = join_texts(raw, NARRATIVE_FIELDS)
                metadata = {
                    key: compact_text(raw.get(key))
                    for key in METADATA_HINTS
                    if compact_text(raw.get(key))
                }
                metadata_text = " ".join(f"{k}: {v}" for k, v in metadata.items())
                text_raw = compact_text(" ".join([synopsis, narrative]))
                text_full = compact_text(" ".join([metadata_text, synopsis, narrative]))
                reports.append(
                    {
                        "doc_id": doc_id,
                        "split": split,
                        "synopsis": synopsis,
                        "narrative": narrative,
                        "metadata": metadata,
                        "text_raw": text_raw,
                        "text_full": text_full,
                    }
                )

    for path in corpus_files:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                field_counter.update(reader.fieldnames)
            for raw in reader:
                doc_id = normalize_id(raw.get("doc_id"))
                if not doc_id:
                    skipped_no_id += 1
                    continue
                if doc_id in seen:
                    duplicate_ids += 1
                    continue
                seen.add(doc_id)
                synopsis = compact_text(raw.get("synopsis"))
                narrative = compact_text(raw.get("narrative"))
                metadata = parse_metadata_json(raw.get("metadata_json", ""))
                metadata_text = " ".join(f"{k}: {v}" for k, v in metadata.items())
                text_raw = compact_text(" ".join([synopsis, narrative]))
                text_full = compact_text(" ".join([metadata_text, synopsis, narrative]))
                reports.append(
                    {
                        "doc_id": doc_id,
                        "split": "corpus",
                        "synopsis": synopsis,
                        "narrative": narrative,
                        "metadata": metadata,
                        "text_raw": text_raw,
                        "text_full": text_full,
                    }
                )

    stats = {
        "files": [str(path.relative_to(RAW_DIR)) for path in files],
        "jsonl_files": [str(path.relative_to(RAW_DIR)) for path in jsonl_files],
        "corpus_files": [str(path.relative_to(RAW_DIR)) for path in corpus_files],
        "rows": len(reports),
        "skipped_no_id": skipped_no_id,
        "duplicate_ids": duplicate_ids,
        "top_fields": field_counter.most_common(30),
    }
    return reports, stats


def split_from_filename(name: str) -> str:
    lowered = name.lower()
    if "validation" in lowered or "valid" in lowered or "dev" in lowered or "_val" in lowered:
        return "validation"
    if "test" in lowered:
        return "test"
    return "train"


def parse_pythonish(value: str) -> Any:
    value = compact_text(value)
    if not value:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def parse_metadata_json(value: str) -> dict[str, str]:
    value = compact_text(value)
    if not value:
        return {}
    try:
        data = json.loads(value)
    except Exception:
        try:
            data = ast.literal_eval(value)
        except Exception:
            return {"metadata_json": value}
    if not isinstance(data, dict):
        return {"metadata_json": compact_text(data)}
    return {str(k): compact_text(v) for k, v in data.items() if compact_text(v)}


def load_query_rows() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    files = [path for path in find_raw_files("queries", ".csv") if path.name != "corpus.csv"]
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    skipped_no_id = 0
    field_counter: Counter[str] = Counter()

    for path in files:
        split = split_from_filename(path.name)
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                field_counter.update(reader.fieldnames)
            for raw in reader:
                query_id = compact_text(raw.get("query_id")) or compact_text(raw.get("id"))
                query = compact_text(raw.get("query"))
                gold_doc_id = normalize_id(raw.get("seed_doc_id") or raw.get("gold_doc_id"))
                if not gold_doc_id or not query:
                    skipped_no_id += 1
                    continue
                row = {
                    "query_id": query_id or f"{gold_doc_id}_{len(by_split[split])}",
                    "query": query,
                    "gold_doc_id": gold_doc_id,
                    "split": split,
                    "style": compact_text(raw.get("style")),
                    "used_fields": parse_pythonish(raw.get("used_fields", "")),
                    "gold_facets_for_analysis_only": parse_pythonish(raw.get("facets", "")),
                }
                by_split.setdefault(split, []).append(row)

    stats = {
        "files": [str(path.relative_to(RAW_DIR)) for path in files],
        "rows_by_split": {split: len(rows) for split, rows in by_split.items()},
        "total_rows": sum(len(rows) for rows in by_split.values()),
        "skipped_no_id_or_query": skipped_no_id,
        "fields": dict(field_counter),
    }
    return by_split, stats


def avg_len(items: list[str]) -> float:
    if not items:
        return 0.0
    return round(mean(len(x.split()) for x in items if x), 2)


def build_report_md(
    report_stats: dict[str, Any],
    query_stats: dict[str, Any],
    alignment: dict[str, Any],
    samples: list[dict[str, Any]],
) -> str:
    lines = [
        "# ASRS Data Feasibility Report",
        "",
        "Generated by `src.prepare_data`.",
        "",
        "## Raw Files",
        "",
        "### Reports",
    ]
    for item in report_stats["files"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "### Queries"])
    for item in query_stats["files"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Reports: {report_stats['rows']}",
            f"- Report rows skipped without ID: {report_stats['skipped_no_id']}",
            f"- Duplicate report IDs skipped: {report_stats['duplicate_ids']}",
            f"- Queries total: {query_stats['total_rows']}",
        ]
    )
    for split, count in query_stats["rows_by_split"].items():
        lines.append(f"- Queries {split}: {count}")
    lines.extend(
        [
            "",
            "## Alignment",
            "",
            f"- Matched queries: {alignment['matched_queries']}",
            f"- Unmatched queries: {alignment['unmatched_queries']}",
            f"- Match rate: {alignment['match_rate']:.4f}",
            f"- Unique gold doc IDs: {alignment['unique_gold_doc_ids']}",
            f"- Matched unique gold doc IDs: {alignment['matched_unique_gold_doc_ids']}",
            "",
            "## Average Lengths",
            "",
            f"- Avg query length: {alignment['avg_query_words']}",
            f"- Avg report raw text length: {alignment['avg_report_raw_words']}",
            f"- Avg report full text length: {alignment['avg_report_full_words']}",
            "",
            "## Sample Matches",
            "",
        ]
    )
    for sample in samples[:10]:
        lines.extend(
            [
                f"### {sample['query_id']} -> {sample['gold_doc_id']}",
                "",
                f"- Query: {sample['query']}",
                f"- Synopsis: {sample['synopsis'][:500]}",
                f"- Metadata keys: {', '.join(sample['metadata_keys'])}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dirs()
    reports, report_stats = load_report_rows()
    queries_by_split, query_stats = load_query_rows()

    if not reports:
        raise SystemExit("No reports loaded. Run src.download_data first.")
    if not any(queries_by_split.values()):
        raise SystemExit("No queries loaded. Run src.download_data first.")

    report_by_id = {row["doc_id"]: row for row in reports}
    all_queries = [row for rows in queries_by_split.values() for row in rows]
    matched = [row for row in all_queries if row["gold_doc_id"] in report_by_id]
    unmatched = [row for row in all_queries if row["gold_doc_id"] not in report_by_id]
    unique_gold = {row["gold_doc_id"] for row in all_queries}
    matched_unique_gold = {row["gold_doc_id"] for row in matched}

    write_jsonl(PROCESSED_DIR / "reports.jsonl", reports)
    for split, rows in queries_by_split.items():
        write_jsonl(PROCESSED_DIR / f"queries_{split}.jsonl", rows)

    rng = random.Random(20260512)
    sample_queries = matched[:]
    rng.shuffle(sample_queries)
    samples = []
    for query in sample_queries[:50]:
        report = report_by_id[query["gold_doc_id"]]
        samples.append(
            {
                "query_id": query["query_id"],
                "gold_doc_id": query["gold_doc_id"],
                "query": query["query"],
                "synopsis": report["synopsis"],
                "metadata_keys": sorted(report["metadata"].keys()),
                "report_text_preview": report["text_raw"][:1000],
            }
        )
    write_jsonl(NOTES_DIR / "alignment_samples.jsonl", samples)

    report_raw_texts = [row["text_raw"] for row in reports]
    report_full_texts = [row["text_full"] for row in reports]
    alignment = {
        "matched_queries": len(matched),
        "unmatched_queries": len(unmatched),
        "match_rate": len(matched) / len(all_queries) if all_queries else 0.0,
        "unique_gold_doc_ids": len(unique_gold),
        "matched_unique_gold_doc_ids": len(matched_unique_gold),
        "avg_query_words": avg_len([row["query"] for row in all_queries]),
        "avg_report_raw_words": avg_len(report_raw_texts),
        "avg_report_full_words": avg_len(report_full_texts),
        "unmatched_examples": unmatched[:20],
    }

    write_json(PROCESSED_DIR / "data_stats.json", {
        "reports": report_stats,
        "queries": query_stats,
        "alignment": alignment,
    })
    (NOTES_DIR / "data_report.md").write_text(
        build_report_md(report_stats, query_stats, alignment, samples),
        encoding="utf-8",
    )

    print(json.dumps({
        "reports": report_stats["rows"],
        "queries": query_stats["total_rows"],
        "matched_queries": alignment["matched_queries"],
        "match_rate": alignment["match_rate"],
        "data_report": str(NOTES_DIR / "data_report.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
