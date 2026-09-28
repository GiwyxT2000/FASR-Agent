from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .common import OUTPUTS_DIR, PROJECT_ROOT


TABLE_DIR = PROJECT_ROOT / "paper" / "tables"


def load_metric(path: str) -> dict[str, Any]:
    return json.loads((OUTPUTS_DIR / "metrics" / path).read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def metric_row(method: str, metric: dict[str, Any]) -> dict[str, str]:
    return {
        "Method": method,
        "Recall@1": fmt(metric["recall@1"]),
        "Recall@5": fmt(metric["recall@5"]),
        "Recall@10": fmt(metric["recall@10"]),
        "MRR@10": fmt(metric["mrr@10"]),
        "nDCG@10": fmt(metric["ndcg@10"]),
        "LLM Calls": str(metric.get("llm_calls", 0)),
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    main_rows = [
        metric_row("BM25-basic", load_metric("bm25_test_text_raw.json")),
        metric_row("BM25-full", load_metric("bm25_test_text_full.json")),
        metric_row("Dense MiniLM", load_metric("dense_test_sentence-transformers__all-MiniLM-L6-v2_text_full.json")),
        metric_row("Dense BGE", load_metric("dense_test_BAAI__bge-small-en-v1.5_text_full.json")),
        metric_row("Hybrid RRF", load_metric("hybrid_test_rrf_dense_test_BAAI__bge-small-en-v1.5_text_full.json")),
        metric_row("FASR-Agent V0", load_metric("facet_reranker_bge_test.json")),
        metric_row("FASR-Agent V1", load_metric("facet_reranker_bge_top50_test.json")),
    ]
    main_fields = ["Method", "Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10", "LLM Calls"]
    write_csv(TABLE_DIR / "main_results.csv", main_rows, main_fields)
    (TABLE_DIR / "main_results.md").write_text(markdown_table(main_rows, main_fields), encoding="utf-8")

    hard_rows = [
        {"Subset": "BM25 miss top-1", "Method": "BM25-full", **small_metrics(load_metric("eval_bm25_full_miss_top1.json"))},
        {"Subset": "BM25 miss top-1", "Method": "Dense BGE", **small_metrics(load_metric("eval_dense_bge_miss_top1.json"))},
        {"Subset": "BM25 miss top-1", "Method": "FASR-Agent V1", **small_metrics(load_metric("eval_facet_top50_miss_top1.json"))},
        {"Subset": "BM25 miss top-10", "Method": "BM25-full", **small_metrics(load_metric("eval_bm25_full_miss_top10.json"))},
        {"Subset": "BM25 miss top-10", "Method": "Dense BGE", **small_metrics(load_metric("eval_dense_bge_top50_miss_top10.json"))},
        {"Subset": "BM25 miss top-10", "Method": "FASR-Agent V1", **small_metrics(load_metric("eval_facet_top50_miss_top10.json"))},
        {"Subset": "Low lexical overlap", "Method": "BM25-full", **small_metrics(load_metric("eval_bm25_full_low_overlap.json"))},
        {"Subset": "Low lexical overlap", "Method": "Dense BGE", **small_metrics(load_metric("eval_dense_bge_low_overlap.json"))},
        {"Subset": "Low lexical overlap", "Method": "FASR-Agent V1", **small_metrics(load_metric("eval_facet_top50_low_overlap.json"))},
    ]
    hard_fields = ["Subset", "Method", "Recall@1", "Recall@5", "Recall@10", "MRR@10"]
    write_csv(TABLE_DIR / "hard_subset_results.csv", hard_rows, hard_fields)
    (TABLE_DIR / "hard_subset_results.md").write_text(markdown_table(hard_rows, hard_fields), encoding="utf-8")

    ablation = load_metric("facet_ablation_bge_top50_test.json")["results"]
    ablation_names = {
        "full": "Full",
        "wo_dense": "w/o dense",
        "wo_bm25": "w/o BM25",
        "wo_text_coverage": "w/o text coverage",
        "wo_metadata_coverage": "w/o metadata coverage",
        "wo_code_number_coverage": "w/o code/year coverage",
        "wo_all_coverage": "w/o all coverage",
    }
    ablation_rows = [
        {"Variant": ablation_names[name], **small_metrics(metric, include_ndcg=True)}
        for name, metric in ablation.items()
    ]
    ablation_fields = ["Variant", "Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"]
    write_csv(TABLE_DIR / "ablation_results.csv", ablation_rows, ablation_fields)
    (TABLE_DIR / "ablation_results.md").write_text(markdown_table(ablation_rows, ablation_fields), encoding="utf-8")

    print(json.dumps({
        "tables": [
            str(TABLE_DIR / "main_results.md"),
            str(TABLE_DIR / "hard_subset_results.md"),
            str(TABLE_DIR / "ablation_results.md"),
        ]
    }, ensure_ascii=False, indent=2))


def small_metrics(metric: dict[str, Any], include_ndcg: bool = False) -> dict[str, str]:
    row = {
        "Recall@1": fmt(metric["recall@1"]),
        "Recall@5": fmt(metric["recall@5"]),
        "Recall@10": fmt(metric["recall@10"]),
        "MRR@10": fmt(metric["mrr@10"]),
    }
    if include_ndcg:
        row["nDCG@10"] = fmt(metric["ndcg@10"])
    return row


if __name__ == "__main__":
    main()

