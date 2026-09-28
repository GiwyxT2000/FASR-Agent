from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import OUTPUTS_DIR, read_jsonl, write_json
from .evaluate import evaluate_ranked_lists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction JSONL, optionally on a subset.")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--subset", default="", help="Optional subset JSONL with query_id field.")
    parser.add_argument("--name", default="", help="Optional output metric name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.prediction)
    rows = list(read_jsonl(prediction_path))

    subset_name = "full"
    if args.subset:
        subset_path = Path(args.subset)
        subset_ids = {str(row["query_id"]) for row in read_jsonl(subset_path)}
        rows = [row for row in rows if str(row["query_id"]) in subset_ids]
        subset_name = subset_path.stem

    metrics = evaluate_ranked_lists(rows)
    metrics.update(
        {
            "prediction": str(prediction_path),
            "subset": subset_name,
            "num_queries": len(rows),
        }
    )
    metric_dir = OUTPUTS_DIR / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"eval_{prediction_path.stem}_{subset_name}"
    metric_path = metric_dir / f"{name}.json"
    write_json(metric_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote metrics: {metric_path}")


if __name__ == "__main__":
    main()

