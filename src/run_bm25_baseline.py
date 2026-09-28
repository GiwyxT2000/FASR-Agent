from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .bm25_retriever import BM25Retriever
from .common import OUTPUTS_DIR, PROCESSED_DIR, read_jsonl, write_json
from .evaluate import evaluate_ranked_lists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 retrieval baseline.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--text-field", default="text_raw", choices=["text_raw", "text_full"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Optional query limit for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = list(read_jsonl(PROCESSED_DIR / "reports.jsonl"))
    queries = list(read_jsonl(PROCESSED_DIR / f"queries_{args.split}.jsonl"))
    if args.limit:
        queries = queries[: args.limit]

    started = time.perf_counter()
    retriever = BM25Retriever()
    retriever.build(reports, args.text_field)
    build_seconds = time.perf_counter() - started

    predictions = []
    search_started = time.perf_counter()
    for idx, query in enumerate(queries, start=1):
        results = retriever.search(query["query"], top_k=args.top_k)
        predictions.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "gold_doc_id": query["gold_doc_id"],
                "ranked": [
                    {"doc_id": result.doc_id, "score": result.score}
                    for result in results
                ],
            }
        )
        if idx % 1000 == 0:
            print(f"[progress] searched {idx}/{len(queries)} queries")
    search_seconds = time.perf_counter() - search_started

    metrics = evaluate_ranked_lists(predictions)
    metrics.update(
        {
            "method": f"bm25_{args.text_field}",
            "split": args.split,
            "num_reports": len(reports),
            "num_queries": len(queries),
            "top_k": args.top_k,
            "build_seconds": round(build_seconds, 3),
            "search_seconds": round(search_seconds, 3),
            "avg_search_ms": round(search_seconds * 1000 / len(queries), 3) if queries else 0.0,
            "llm_calls": 0,
        }
    )

    pred_dir = OUTPUTS_DIR / "predictions"
    metric_dir = OUTPUTS_DIR / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    top_suffix = "" if args.top_k == 10 else f"_top{args.top_k}"
    suffix = f"{args.split}_{args.text_field}{top_suffix}"
    pred_path = pred_dir / f"bm25_{suffix}.jsonl"
    with pred_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metric_path = metric_dir / f"bm25_{suffix}.json"
    write_json(metric_path, metrics)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics: {metric_path}")


if __name__ == "__main__":
    main()
