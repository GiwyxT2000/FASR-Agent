from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .text_utils import token_counts, tokenize


@dataclass
class BM25Result:
    doc_id: str
    score: float


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_lens: list[int] = []
        self.avgdl = 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}

    def build(self, docs: list[dict[str, str]], text_field: str) -> None:
        self.doc_ids = []
        self.doc_lens = []
        self.postings = defaultdict(list)
        doc_freq: Counter[str] = Counter()

        for idx, doc in enumerate(docs):
            self.doc_ids.append(str(doc["doc_id"]))
            counts = token_counts(doc.get(text_field, ""))
            doc_len = sum(counts.values())
            self.doc_lens.append(doc_len)
            for term, tf in counts.items():
                self.postings[term].append((idx, tf))
                doc_freq[term] += 1

        n_docs = len(self.doc_ids)
        self.avgdl = sum(self.doc_lens) / n_docs if n_docs else 0.0
        self.idf = {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        query_counts = Counter(query_terms)
        for term, qtf in query_counts.items():
            term_idf = self.idf.get(term)
            if term_idf is None:
                continue
            for doc_idx, tf in self.postings.get(term, []):
                doc_len = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / (self.avgdl or 1.0))
                scores[doc_idx] += qtf * term_idf * (tf * (self.k1 + 1.0) / denom)

        if not scores:
            return []
        best = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
        return [BM25Result(doc_id=self.doc_ids[idx], score=score) for idx, score in best]

