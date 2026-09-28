# FASR-Agent

Code for **FASR-Agent: Facet-Aware Evidence Reranking for Aviation Safety Training
Case Retrieval** (AIFE 2026).
[[Paper]](https://doi.org/10.1145/3848861.3849006)

Given a natural-language description of an aviation scenario, FASR-Agent returns
the ASRS report that the scenario was derived from, so that a training system can
ground a case discussion in the original operational record.

Retrieval runs in two stages. A multi-channel candidate pool is first assembled
from metadata-enriched BM25 and Dense BGE retrieval, because aviation queries
depend both on exact operational terms and on semantics that survive
paraphrasing. The pool is then reranked by explicit evidence coverage features —
text, metadata, uppercase aviation codes, and numbers/years — combined linearly
with the reciprocal-rank and normalised-score signals of both channels. The
reranking weights are tuned on the validation split and then fixed for test
evaluation. No LLM is called at retrieval time.

## Implementation

Python 3.10, PyTorch 2.7.1+cu128 (CUDA 12.8). The reported experiments ran on a
single NVIDIA GeForce RTX 4060 Laptop GPU.

```bash
pip install -r requirements.txt
```

The dense channel uses `BAAI/bge-small-en-v1.5` through `sentence-transformers`,
which downloads the encoder on first use. The lexical channel is a self-contained
BM25 implementation in `src/bm25_retriever.py` and needs no external index.

## Dataset Construction

The benchmark is derived from two public HuggingFace datasets and is not
redistributed here. This repository ships the pipeline that builds it plus a
small sample of the construction output in `notes/` — enough to read the record
format and to follow the case study — so start by rebuilding the data:

```bash
python -m src.download_data   # rnapberkeley/asrs, elihoole/asrs-aviation-reports
python -m src.prepare_data
```

`download_data.py` fetches the raw sources into `data/raw/`. `prepare_data.py`
then turns them into the benchmark under `data/processed/`:

1. report records are keyed by `acn_num_ACN` and merged into a single corpus,
   each carrying a raw text field and a metadata-enriched one;
2. queries are normalised and grouped by their released train / validation /
   test split;
3. every query is linked to its seed report through `seed_doc_id`, so each
   sample keeps its mapping back to the original record;
4. a construction report with counts, alignment rate and sample matches is
   written to `notes/data_report.md`.

The resulting benchmark holds 47,723 reports and 47,725 queries, split into
33,407 train / 7,158 validation / 7,160 test. Every query carries a `gold_doc_id`
that points at its target report.

The corpus is far past GitHub's per-file size limit, so a rebuilt corpus is
normally kept compressed. `read_jsonl` in `src/common.py` transparently falls
back to a `.xz` or `.gz` sibling when the plain file is absent: the scripts
always pass the logical name, and nothing needs to be unpacked by hand.

## Usage

Run everything from the repository root, after the data has been rebuilt. The
two retrieval channels are built first; every later stage consumes their
prediction files.

### Retrieval channels

```bash
python -m src.run_bm25_baseline  --split test --text-field text_full --top-k 50
python -m src.run_dense_baseline --split test --text-field text_full --top-k 50 \
    --model BAAI/bge-small-en-v1.5
```

`--text-field text_full` indexes metadata together with synopsis and narrative.
Using `text_raw` instead restricts BM25 to synopsis and narrative, which gives
the `BM25-basic` baseline. Predictions land in `outputs/predictions/`.

### Reciprocal rank fusion

```bash
python -m src.run_hybrid_rrf --split test \
    --bm25  outputs/predictions/bm25_test_text_full_top50.jsonl \
    --dense outputs/predictions/dense_test_BAAI__bge-small-en-v1.5_text_full_top50.jsonl
```

### FASR-Agent

Tune the weights once on validation, then reuse them unchanged on test:

```bash
python -m src.run_facet_reranker --split validation \
    --bm25  outputs/predictions/bm25_validation_text_full_top50.jsonl \
    --dense outputs/predictions/dense_validation_BAAI__bge-small-en-v1.5_text_full_top50.jsonl \
    --save-weights outputs/weights/bge_top50.json

python -m src.run_facet_reranker --split test \
    --bm25  outputs/predictions/bm25_test_text_full_top50.jsonl \
    --dense outputs/predictions/dense_test_BAAI__bge-small-en-v1.5_text_full_top50.jsonl \
    --weights outputs/weights/bge_top50.json --name facet_reranker_bge_top50
```

The candidate pool is exactly what the two input files contain, so building the
channels with `--top-k 10` reproduces the small-pool variant and `--top-k 50` the
large-pool variant.

### Ablation, hard subsets, case study, tables

```bash
python -m src.run_facet_ablation --split test \
    --bm25  outputs/predictions/bm25_test_text_full_top50.jsonl \
    --dense outputs/predictions/dense_test_BAAI__bge-small-en-v1.5_text_full_top50.jsonl \
    --weights outputs/weights/bge_top50.json

python -m src.build_hard_subsets --split test \
    --prediction outputs/predictions/bm25_test_text_full.jsonl

python -m src.evaluate_predictions \
    --prediction outputs/predictions/facet_reranker_bge_top50_test.jsonl \
    --subset data/processed/hard_subsets/test_low_lexical_overlap.jsonl

python -m src.analyze_comparisons \
    --baseline outputs/predictions/bm25_test_text_full.jsonl \
    --method   outputs/predictions/facet_reranker_bge_top50_test.jsonl

python -m src.export_paper_tables
```

`build_hard_subsets.py` writes the BM25-miss and low lexical-overlap subsets used
by the evaluation above. `export_paper_tables.py` reads `outputs/metrics/`, so the
evaluation steps have to run first, and writes the paper's result tables as CSV
and Markdown.

## Layout

```
src/                    retrieval pipeline
  common.py             paths and JSONL/JSON helpers
  text_utils.py         tokenisation used by BM25 and the coverage features
  bm25_retriever.py     BM25 index
  download_data.py      fetch the source datasets
  prepare_data.py       normalise into the benchmark
  run_bm25_baseline.py  lexical channel
  run_dense_baseline.py dense channel
  run_hybrid_rrf.py     rank fusion baseline
  run_facet_reranker.py FASR-Agent
  run_facet_ablation.py component ablations
  build_hard_subsets.py BM25-miss and low-overlap subsets
  evaluate.py           ranking metrics
  evaluate_predictions.py  metrics for one prediction file
  analyze_comparisons.py   case-study extraction
  export_paper_tables.py   result tables
notes/                  construction report, alignment samples, case study
configs/                example configuration
```

`data/raw/`, `data/processed/` and `outputs/` are generated by the scripts and
are not tracked.

## A note on the source data

ASRS reports are voluntary, de-identified safety narratives. NASA does not verify
or validate the reported details, so this benchmark evaluates the retrieval of
reports as training cases, not the verification of aviation events.

## Citation

```bibtex
@inproceedings{wang2026fasr,
  title     = {FASR-Agent: Facet-Aware Evidence Reranking for Aviation Safety Training Case Retrieval},
  author    = {Wang, YiXiang and Li, JianDun},
  booktitle = {2026 3rd International Conference on Artificial Intelligence and Future Education (AIFE 2026)},
  year      = {2026},
  doi       = {10.1145/3848861.3849006}
}
```

## License

MIT — see [LICENSE](LICENSE).
