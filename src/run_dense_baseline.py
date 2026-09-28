from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import OUTPUTS_DIR, PROCESSED_DIR, read_jsonl, write_json
from .evaluate import evaluate_ranked_lists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense retrieval baseline with sentence-transformers.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--text-field", default="text_full", choices=["text_raw", "text_full"])
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="", help="Optional sentence-transformers device, e.g. cuda or cpu.")
    parser.add_argument("--query-prefix", default="", help="Optional prefix added before each query before encoding.")
    parser.add_argument("--doc-prefix", default="", help="Optional prefix added before each document before encoding.")
    return parser.parse_args()


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace("\\", "__").replace(":", "_")


def prefix_slug(prefix: str) -> str:
    if not prefix:
        return "noprefix"
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in prefix[:40]).strip("_")
    return safe or "prefix"


def encode_or_load(
    model,
    texts: list[str],
    cache_path: Path,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if cache_path.exists():
        print(f"[cache] loading {cache_path}")
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(texts):
            return embeddings
        print(
            f"[cache-mismatch] {cache_path} has {embeddings.shape[0]} rows, "
            f"expected {len(texts)}; re-encoding"
        )
    print(f"[encode] {len(texts)} texts -> {cache_path}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device or None,
    )
    embeddings = embeddings.astype("float32")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def search_topk(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_ids: list[str],
    top_k: int,
    query_batch_size: int,
) -> list[list[dict[str, float | str]]]:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    doc_tensor = torch.from_numpy(doc_embeddings).to(device)
    all_ranked: list[list[dict[str, float | str]]] = []
    for start in range(0, len(query_embeddings), query_batch_size):
        end = min(start + query_batch_size, len(query_embeddings))
        q_tensor = torch.from_numpy(query_embeddings[start:end]).to(device)
        scores = q_tensor @ doc_tensor.T
        values, indices = torch.topk(scores, k=top_k, dim=1)
        values_np = values.detach().cpu().numpy()
        indices_np = indices.detach().cpu().numpy()
        for row_values, row_indices in zip(values_np, indices_np):
            all_ranked.append(
                [
                    {"doc_id": doc_ids[int(idx)], "score": float(score)}
                    for score, idx in zip(row_values, row_indices)
                ]
            )
        print(f"[progress] dense searched {end}/{len(query_embeddings)} queries")
    return all_ranked


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    from sentence_transformers import SentenceTransformer

    reports = list(read_jsonl(PROCESSED_DIR / "reports.jsonl"))
    queries = list(read_jsonl(PROCESSED_DIR / f"queries_{args.split}.jsonl"))
    if args.limit:
        queries = queries[: args.limit]

    print(f"[model] {args.model}")
    model = SentenceTransformer(args.model, device=args.device or None)
    slug = model_slug(args.model)
    cache_dir = OUTPUTS_DIR / "cache" / "embeddings"
    doc_prefix_slug = prefix_slug(args.doc_prefix)
    query_prefix_slug = prefix_slug(args.query_prefix)
    doc_cache = cache_dir / f"{slug}_{args.text_field}_{doc_prefix_slug}_docs.npy"
    query_cache = cache_dir / f"{slug}_{args.split}_{query_prefix_slug}_queries.npy"
    if args.limit:
        query_cache = cache_dir / f"{slug}_{args.split}_limit{args.limit}_queries.npy"

    doc_texts = [args.doc_prefix + row.get(args.text_field, "") for row in reports]
    query_texts = [args.query_prefix + row.get("query", "") for row in queries]
    doc_ids = [str(row["doc_id"]) for row in reports]

    encode_started = time.perf_counter()
    doc_embeddings = encode_or_load(model, doc_texts, doc_cache, args.batch_size, args.device)
    query_embeddings = encode_or_load(model, query_texts, query_cache, args.batch_size, args.device)
    encode_seconds = time.perf_counter() - encode_started

    search_started = time.perf_counter()
    ranked_lists = search_topk(
        query_embeddings,
        doc_embeddings,
        doc_ids,
        args.top_k,
        args.query_batch_size,
    )
    search_seconds = time.perf_counter() - search_started

    predictions = []
    for query, ranked in zip(queries, ranked_lists):
        predictions.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "gold_doc_id": query["gold_doc_id"],
                "ranked": ranked,
            }
        )

    metrics = evaluate_ranked_lists(predictions)
    metrics.update(
        {
            "method": f"dense_{slug}_{args.text_field}",
            "model": args.model,
            "query_prefix": args.query_prefix,
            "doc_prefix": args.doc_prefix,
            "split": args.split,
            "num_reports": len(reports),
            "num_queries": len(queries),
            "top_k": args.top_k,
            "encode_seconds": round(encode_seconds, 3),
            "search_seconds": round(search_seconds, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
            "avg_search_ms": round(search_seconds * 1000 / len(queries), 3) if queries else 0.0,
            "llm_calls": 0,
        }
    )

    pred_dir = OUTPUTS_DIR / "predictions"
    metric_dir = OUTPUTS_DIR / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    top_suffix = "" if args.top_k == 10 else f"_top{args.top_k}"
    suffix = f"{args.split}_{slug}_{args.text_field}{top_suffix}"
    pred_path = pred_dir / f"dense_{suffix}.jsonl"
    with pred_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metric_path = metric_dir / f"dense_{suffix}.json"
    write_json(metric_path, metrics)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote metrics: {metric_path}")


if __name__ == "__main__":
    main()
