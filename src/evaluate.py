from __future__ import annotations

from typing import Any


def evaluate_ranked_lists(rows: list[dict[str, Any]], cutoffs: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    if not rows:
        return {f"recall@{k}": 0.0 for k in cutoffs} | {"mrr@10": 0.0, "ndcg@10": 0.0}

    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        hits = 0
        for row in rows:
            gold = str(row["gold_doc_id"])
            ranked = [str(item["doc_id"]) for item in row.get("ranked", [])[:cutoff]]
            if gold in ranked:
                hits += 1
        metrics[f"recall@{cutoff}"] = hits / len(rows)

    rr_total = 0.0
    ndcg_total = 0.0
    for row in rows:
        gold = str(row["gold_doc_id"])
        ranked = [str(item["doc_id"]) for item in row.get("ranked", [])[:10]]
        if gold in ranked:
            rank = ranked.index(gold) + 1
            rr_total += 1.0 / rank
            ndcg_total += 1.0 / log2(rank + 1)
    metrics["mrr@10"] = rr_total / len(rows)
    metrics["ndcg@10"] = ndcg_total / len(rows)
    return metrics


def log2(value: int) -> float:
    import math

    return math.log(value, 2)

