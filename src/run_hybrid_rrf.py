from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import OUTPUTS_DIR, read_jsonl, write_json
from .evaluate import evaluate_ranked_lists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse retrieval predictions with reciprocal rank fusion.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--bm25", required=True, help="BM25 prediction JSONL.")
    parser.add_argument("--dense", required=True, help="Dense prediction JSONL.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    return {str(row["query_id"]): row for row in rows}


def safe_stem(path: Path) -> str:
    text = path.stem
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:120]


def fuse_one(rows: list[dict[str, Any]], top_k: int, rrf_k: int) -> list[dict[str, float | str]]:
    scores: dict[str, float] = defaultdict(float)
    raw_scores: dict[str, float] = defaultdict(float)
    for source_idx, row in enumerate(rows):
        for rank, item in enumerate(row.get("ranked", []), start=1):
            doc_id = str(item["doc_id"])
            scores[doc_id] += 1.0 / (rrf_k + rank)
            raw_scores[doc_id] += float(item.get("score", 0.0))
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        {"doc_id": doc_id, "score": score, "raw_score_sum": raw_scores[doc_id]}
        for doc_id, score in ranked
    ]


def main() -> None:
    args = parse_args()
    bm25_path = Path(args.bm25)
    dense_path = Path(args.dense)
    bm25 = load_predictions(bm25_path)
    dense = load_predictions(dense_path)
    common_ids = sorted(set(bm25) & set(dense))
    if not common_ids:
        raise SystemExit("No overlapping query IDs between prediction files.")

    predictions = []
    for query_id in common_ids:
        first = bm25[query_id]
        fused = fuse_one([bm25[query_id], dense[query_id]], args.top_k, args.rrf_k)
        predictions.append(
            {
                "query_id": query_id,
                "query": first["query"],
                "gold_doc_id": first["gold_doc_id"],
                "ranked": fused,
            }
        )

    metrics = evaluate_ranked_lists(predictions)
    metrics.update(
        {
            "method": "hybrid_rrf",
            "split": args.split,
            "num_queries": len(predictions),
            "top_k": args.top_k,
            "rrf_k": args.rrf_k,
            "bm25_prediction": str(bm25_path),
            "dense_prediction": str(dense_path),
            "llm_calls": 0,
        }
    )

    pred_dir = OUTPUTS_DIR / "predictions"
    metric_dir = OUTPUTS_DIR / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}_rrf_{safe_stem(dense_path)}"
    pred_path = pred_dir / f"hybrid_{suffix}.jsonl"
    with pred_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metric_path = metric_dir / f"hybrid_{suffix}.json"
    write_json(metric_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics: {metric_path}")


if __name__ == "__main__":
    main()
