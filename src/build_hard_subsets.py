from __future__ import annotations

import argparse
from pathlib import Path

from .common import OUTPUTS_DIR, PROCESSED_DIR, read_jsonl, write_json, write_jsonl
from .text_utils import tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build objective hard subsets from baseline predictions.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--prediction", default="", help="Prediction JSONL path.")
    parser.add_argument("--low-overlap-ratio", type=float, default=0.2)
    return parser.parse_args()


def lexical_overlap(query: str, doc_text: str) -> float:
    q_tokens = set(tokenize(query))
    d_tokens = set(tokenize(doc_text))
    if not q_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / len(q_tokens)


def main() -> None:
    args = parse_args()
    pred_path = Path(args.prediction) if args.prediction else (
        OUTPUTS_DIR / "predictions" / f"bm25_{args.split}_text_full.jsonl"
    )
    predictions = list(read_jsonl(pred_path))
    reports = {row["doc_id"]: row for row in read_jsonl(PROCESSED_DIR / "reports.jsonl")}

    miss_top1 = []
    miss_top10 = []
    scored = []
    for row in predictions:
        gold = str(row["gold_doc_id"])
        ranked = [str(item["doc_id"]) for item in row.get("ranked", [])]
        gold_report = reports.get(gold, {})
        overlap = lexical_overlap(row.get("query", ""), gold_report.get("text_raw", ""))
        enriched = {
            "query_id": row["query_id"],
            "query": row["query"],
            "gold_doc_id": gold,
            "gold_in_top1": bool(ranked[:1] and ranked[0] == gold),
            "gold_in_top10": gold in ranked[:10],
            "lexical_overlap": round(overlap, 6),
            "top1_doc_id": ranked[0] if ranked else "",
        }
        scored.append(enriched)
        if not enriched["gold_in_top1"]:
            miss_top1.append(enriched)
        if not enriched["gold_in_top10"]:
            miss_top10.append(enriched)

    scored_sorted = sorted(scored, key=lambda item: item["lexical_overlap"])
    low_count = max(1, int(len(scored_sorted) * args.low_overlap_ratio)) if scored_sorted else 0
    low_overlap = scored_sorted[:low_count]

    out_dir = PROCESSED_DIR / "hard_subsets"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / f"{args.split}_bm25_full_miss_top1.jsonl", miss_top1)
    write_jsonl(out_dir / f"{args.split}_bm25_full_miss_top10.jsonl", miss_top10)
    write_jsonl(out_dir / f"{args.split}_low_lexical_overlap.jsonl", low_overlap)

    stats = {
        "split": args.split,
        "prediction": str(pred_path),
        "total": len(scored),
        "bm25_full_miss_top1": len(miss_top1),
        "bm25_full_miss_top10": len(miss_top10),
        "low_lexical_overlap_count": len(low_overlap),
        "low_lexical_overlap_ratio": args.low_overlap_ratio,
        "low_lexical_overlap_threshold": low_overlap[-1]["lexical_overlap"] if low_overlap else 0.0,
    }
    write_json(out_dir / f"{args.split}_hard_subset_stats.json", stats)
    print(stats)


if __name__ == "__main__":
    main()

