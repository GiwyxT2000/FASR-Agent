from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import OUTPUTS_DIR, PROCESSED_DIR, read_jsonl, write_json
from .evaluate import evaluate_ranked_lists
from .run_facet_reranker import (
    build_feature_rows,
    load_prediction_map,
    rank_with_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablations for facet-aware reranker.")
    parser.add_argument("--split", required=True, choices=["validation", "test"])
    parser.add_argument("--bm25", required=True)
    parser.add_argument("--dense", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--name", default="facet_ablation")
    return parser.parse_args()


def zeroed(weights: dict[str, float], fields: list[str]) -> dict[str, float]:
    result = dict(weights)
    for field in fields:
        result[field] = 0.0
    return result


def main() -> None:
    args = parse_args()
    base_weights = json.loads(Path(args.weights).read_text(encoding="utf-8"))["weights"]
    reports = {row["doc_id"]: row for row in read_jsonl(PROCESSED_DIR / "reports.jsonl")}
    bm25_rows = load_prediction_map(Path(args.bm25))
    dense_rows = load_prediction_map(Path(args.dense))
    feature_rows = build_feature_rows(bm25_rows, dense_rows, reports)

    variants = {
        "full": base_weights,
        "wo_dense": zeroed(base_weights, ["dense_rr", "dense_norm"]),
        "wo_bm25": zeroed(base_weights, ["bm25_rr", "bm25_norm"]),
        "wo_text_coverage": zeroed(base_weights, ["text_coverage"]),
        "wo_metadata_coverage": zeroed(base_weights, ["metadata_coverage"]),
        "wo_code_number_coverage": zeroed(base_weights, ["code_coverage", "number_coverage"]),
        "wo_all_coverage": zeroed(
            base_weights,
            ["text_coverage", "metadata_coverage", "code_coverage", "number_coverage"],
        ),
    }

    results = {}
    for variant, weights in variants.items():
        predictions = rank_with_weights(feature_rows, weights, args.top_k)
        metrics = evaluate_ranked_lists(predictions)
        metrics["weights"] = weights
        results[variant] = metrics
        print(variant, json.dumps(metrics, ensure_ascii=False))

    metric_path = OUTPUTS_DIR / "metrics" / f"{args.name}_{args.split}.json"
    write_json(metric_path, {
        "split": args.split,
        "bm25_prediction": args.bm25,
        "dense_prediction": args.dense,
        "results": results,
    })
    print(f"Wrote metrics: {metric_path}")


if __name__ == "__main__":
    main()

