from __future__ import annotations

import argparse
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import OUTPUTS_DIR, PROCESSED_DIR, read_jsonl, write_json
from .evaluate import evaluate_ranked_lists
from .text_utils import tokenize


CODE_RE = re.compile(r"\b[A-Z0-9]{2,5}\b")
NUMBER_RE = re.compile(r"\b\d{2,4}\b")


@dataclass
class CandidateFeatures:
    doc_id: str
    bm25_rr: float = 0.0
    dense_rr: float = 0.0
    bm25_norm: float = 0.0
    dense_norm: float = 0.0
    text_coverage: float = 0.0
    metadata_coverage: float = 0.0
    code_coverage: float = 0.0
    number_coverage: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Facet-aware lightweight reranker over BM25 and dense candidates.")
    parser.add_argument("--split", required=True, choices=["validation", "test"])
    parser.add_argument("--bm25", required=True)
    parser.add_argument("--dense", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--weights", default="", help="Optional JSON weights file. If absent, tune on this split.")
    parser.add_argument("--save-weights", default="", help="Optional path to save tuned weights.")
    parser.add_argument("--name", default="facet_reranker")
    return parser.parse_args()


def load_prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["query_id"]): row for row in read_jsonl(path)}


def metadata_text(report: dict[str, Any]) -> str:
    metadata = report.get("metadata", {})
    if isinstance(metadata, dict):
        return " ".join(f"{k} {v}" for k, v in metadata.items())
    return str(metadata or "")


def coverage(query_items: set[str], candidate_items: set[str]) -> float:
    if not query_items:
        return 0.0
    return len(query_items & candidate_items) / len(query_items)


def extract_codes(text: str) -> set[str]:
    return {item for item in CODE_RE.findall(text or "") if not item.isdigit()}


def extract_numbers(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text or ""))


def build_feature_rows(
    bm25_rows: dict[str, dict[str, Any]],
    dense_rows: dict[str, dict[str, Any]],
    reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id in sorted(set(bm25_rows) & set(dense_rows)):
        bm25 = bm25_rows[query_id]
        dense = dense_rows[query_id]
        query = bm25["query"]
        q_tokens = set(tokenize(query))
        q_codes = extract_codes(query)
        q_numbers = extract_numbers(query)

        feature_by_doc: dict[str, CandidateFeatures] = {}

        bm25_scores = [float(item.get("score", 0.0)) for item in bm25.get("ranked", [])]
        dense_scores = [float(item.get("score", 0.0)) for item in dense.get("ranked", [])]
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        dense_max = max(dense_scores) if dense_scores else 1.0

        for rank, item in enumerate(bm25.get("ranked", []), start=1):
            doc_id = str(item["doc_id"])
            feat = feature_by_doc.setdefault(doc_id, CandidateFeatures(doc_id=doc_id))
            feat.bm25_rr = 1.0 / rank
            feat.bm25_norm = float(item.get("score", 0.0)) / bm25_max if bm25_max else 0.0

        for rank, item in enumerate(dense.get("ranked", []), start=1):
            doc_id = str(item["doc_id"])
            feat = feature_by_doc.setdefault(doc_id, CandidateFeatures(doc_id=doc_id))
            feat.dense_rr = 1.0 / rank
            feat.dense_norm = float(item.get("score", 0.0)) / dense_max if dense_max else 0.0

        for doc_id, feat in feature_by_doc.items():
            report = reports.get(doc_id, {})
            text_full = report.get("text_full", "")
            meta = metadata_text(report)
            text_tokens = set(tokenize(text_full))
            meta_tokens = set(tokenize(meta))
            candidate_codes = extract_codes(text_full)
            candidate_numbers = extract_numbers(text_full)
            feat.text_coverage = coverage(q_tokens, text_tokens)
            feat.metadata_coverage = coverage(q_tokens, meta_tokens)
            feat.code_coverage = coverage(q_codes, candidate_codes)
            feat.number_coverage = coverage(q_numbers, candidate_numbers)

        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "gold_doc_id": bm25["gold_doc_id"],
                "features": [feat.__dict__ for feat in feature_by_doc.values()],
            }
        )
    return rows


def score_candidate(features: dict[str, Any], weights: dict[str, float]) -> float:
    return sum(float(features.get(name, 0.0)) * value for name, value in weights.items())


def rank_with_weights(feature_rows: list[dict[str, Any]], weights: dict[str, float], top_k: int) -> list[dict[str, Any]]:
    predictions = []
    for row in feature_rows:
        ranked_features = sorted(
            row["features"],
            key=lambda item: score_candidate(item, weights),
            reverse=True,
        )[:top_k]
        predictions.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "gold_doc_id": row["gold_doc_id"],
                "ranked": [
                    {
                        "doc_id": item["doc_id"],
                        "score": score_candidate(item, weights),
                        "bm25_rr": item["bm25_rr"],
                        "dense_rr": item["dense_rr"],
                        "text_coverage": item["text_coverage"],
                        "metadata_coverage": item["metadata_coverage"],
                        "code_coverage": item["code_coverage"],
                        "number_coverage": item["number_coverage"],
                    }
                    for item in ranked_features
                ],
            }
        )
    return predictions


def tune_weights(feature_rows: list[dict[str, Any]], top_k: int) -> tuple[dict[str, float], dict[str, float]]:
    grid = {
        "bm25_rr": [1.0, 2.0, 3.0],
        "dense_rr": [0.0, 0.5, 1.0],
        "bm25_norm": [0.0, 0.5],
        "dense_norm": [0.0, 0.5],
        "text_coverage": [0.0, 0.5],
        "metadata_coverage": [0.0, 0.5, 1.0],
        "code_coverage": [0.0, 0.5],
        "number_coverage": [0.0, 0.5],
    }
    names = list(grid)
    best_weights: dict[str, float] = {}
    best_metrics: dict[str, float] = {}
    best_key = (-1.0, -1.0, -1.0)
    total = 1
    for values in grid.values():
        total *= len(values)
    checked = 0
    for values in itertools.product(*(grid[name] for name in names)):
        weights = dict(zip(names, values))
        predictions = rank_with_weights(feature_rows, weights, top_k)
        metrics = evaluate_ranked_lists(predictions)
        key = (metrics["mrr@10"], metrics["recall@1"], metrics["recall@10"])
        if key > best_key:
            best_key = key
            best_weights = weights
            best_metrics = metrics
        checked += 1
        if checked % 2000 == 0:
            print(f"[tune] checked {checked}/{total}; best={best_key}")
    print(f"[tune] checked {checked}/{total}")
    return best_weights, best_metrics


def main() -> None:
    args = parse_args()
    reports = {row["doc_id"]: row for row in read_jsonl(PROCESSED_DIR / "reports.jsonl")}
    bm25_rows = load_prediction_map(Path(args.bm25))
    dense_rows = load_prediction_map(Path(args.dense))
    feature_rows = build_feature_rows(bm25_rows, dense_rows, reports)

    if args.weights:
        weights = json.loads(Path(args.weights).read_text(encoding="utf-8"))["weights"]
        dev_metrics = {}
    else:
        weights, dev_metrics = tune_weights(feature_rows, args.top_k)
        if args.save_weights:
            write_json(Path(args.save_weights), {"weights": weights, "validation_metrics": dev_metrics})

    predictions = rank_with_weights(feature_rows, weights, args.top_k)
    metrics = evaluate_ranked_lists(predictions)
    metrics.update(
        {
            "method": args.name,
            "split": args.split,
            "num_queries": len(predictions),
            "top_k": args.top_k,
            "bm25_prediction": args.bm25,
            "dense_prediction": args.dense,
            "weights": weights,
            "tuning_metrics": dev_metrics,
            "llm_calls": 0,
        }
    )

    pred_dir = OUTPUTS_DIR / "predictions"
    metric_dir = OUTPUTS_DIR / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{args.name}_{args.split}.jsonl"
    with pred_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metric_path = metric_dir / f"{args.name}_{args.split}.json"
    write_json(metric_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics: {metric_path}")


if __name__ == "__main__":
    main()
