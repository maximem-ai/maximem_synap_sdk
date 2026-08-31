"""BM25 scorer for the anticipation cache."""

import math
import re
from typing import List


from ..behavior import bm25_params, stop_words, suffixes

# Tuned values live in the shared behavior contract so the Python and JS SDKs
# cannot drift. See synap/sdk/CONTRACT/README.md. Do NOT re-inline them here.
_SUFFIXES = suffixes()
_MIN_STEM_LEN = int(bm25_params()["min_stem_len"])
_MIN_TOKEN_LEN = int(bm25_params()["min_token_len"])
_STOP_WORDS = stop_words()


def _stem(word: str) -> str:
    # Order is load-bearing: longest suffix first, return on first match.
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM_LEN:
            return word[: -len(suffix)]
    return word


# An alias label, as it appears in text: `[[PERSON_PHONE_h2n7v5cx8m0d]]`.
# Kept whole rather than split into `person`, `phone` and the suffix, because
# the first two are boilerplate shared by every label of that field type and
# would make two different people's memories score alike.
#
# Mirrors `synap/cloud/shared/tokenizer.py`. Keep the two in step.
_ALIAS_TOKEN = re.compile(bm25_params()["alias_token"]["pattern"])


def tokenize(text: str) -> List[str]:
    """Tokenize, lowercase, remove stop words, and stem.

    Alias labels survive as single tokens. See `_ALIAS_TOKEN`.
    """
    if not text:
        return []

    labels: List[str] = []

    def _take(match) -> str:
        labels.append(match.group(1).lower())
        return " "

    remainder = _ALIAS_TOKEN.sub(_take, text)
    tokens = re.findall(r"[a-z0-9]+", remainder.lower())
    return labels + [
        _stem(t) for t in tokens if len(t) >= _MIN_TOKEN_LEN and t not in _STOP_WORDS
    ]


class BM25:
    """Okapi BM25 scorer over a small in-memory corpus."""

    IDF_FLOOR = float(bm25_params()["idf_floor"])

    def __init__(
        self,
        corpus: List[List[str]],
        k1: float = float(bm25_params()["k1"]),
        b: float = float(bm25_params()["b"]),
    ):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.avgdl = (
            sum(len(doc) for doc in corpus) / self.doc_count
            if self.doc_count
            else 1.0
        )
        self.doc_lens = [len(doc) for doc in corpus]

        self.idf = {}
        all_terms = set(t for doc in corpus for t in doc)
        for term in all_terms:
            df = sum(1 for doc in corpus if term in doc)
            raw_idf = math.log(
                (self.doc_count - df + 0.5) / (df + 0.5) + 1.0
            )
            self.idf[term] = max(raw_idf, self.IDF_FLOOR)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc = self.corpus[doc_idx]
        dl = self.doc_lens[doc_idx]
        s = 0.0

        tf_map: dict = {}
        for t in doc:
            tf_map[t] = tf_map.get(t, 0) + 1

        for term in query_tokens:
            if term not in tf_map:
                continue
            tf = tf_map[term]
            idf = self.idf.get(term, self.IDF_FLOOR)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += idf * numerator / denominator

        return s

    def scores(self, query_tokens: List[str]) -> List[float]:
        return [self.score(query_tokens, i) for i in range(self.doc_count)]
