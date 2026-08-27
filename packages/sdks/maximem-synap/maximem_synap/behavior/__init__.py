"""Loader for the shared SDK behavior contract.

The values here are tuned against measured eval failures and are shared,
byte-identical, with the JS/TS SDK. They live in data rather than code
specifically so the two SDKs cannot drift: see
``synap/sdk/CONTRACT/README.md``.

The JSON is vendored into the package at build time by
``synap/sdk/scripts/sync-behavior.sh`` because a wheel cannot reach a file
outside its own package root. ``test_behavior_contract.py`` enforces that the
vendored copy matches the canonical one.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Pattern, Tuple

_CONTRACT_PATH = Path(__file__).with_name("anticipation.json")


@lru_cache(maxsize=1)
def contract() -> Dict[str, Any]:
    """The parsed behavior contract. Cached; the file never changes at runtime."""
    return json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))


_PY_FLAGS = {"i": re.I, "m": re.M, "s": re.S}


def _compile(pattern: str, flags: str = "") -> Pattern[str]:
    bits = 0
    for ch in flags:
        if ch not in _PY_FLAGS:
            raise ValueError(
                f"Unsupported regex flag {ch!r} in behavior contract. Flags must "
                f"be valid in both Python and ECMAScript."
            )
        bits |= _PY_FLAGS[ch]
    return re.compile(pattern, bits)


@lru_cache(maxsize=1)
def recall_query_patterns() -> Tuple[Pattern[str], ...]:
    """Compiled recall-bypass patterns, in contract order."""
    return tuple(
        _compile(p["pattern"], p.get("flags", ""))
        for p in contract()["recall_bypass"]["patterns"]
    )


@lru_cache(maxsize=1)
def bm25_params() -> Dict[str, Any]:
    return contract()["bm25"]


@lru_cache(maxsize=1)
def thresholds() -> Dict[str, float]:
    t = contract()["thresholds"]
    return {
        "bm25": float(t["bm25"]["value"]),
        "novel_term": float(t["novel_term"]["value"]),
        "min_corpus_for_novel_gate": int(t["min_corpus_for_novel_gate"]["value"]),
        "effective_floor": float(t["effective_floor"]["value"]),
        "query_token_scale": float(t["query_token_scale"]["value"]),
    }


@lru_cache(maxsize=1)
def telemetry_limits() -> Dict[str, int]:
    t = contract()["telemetry"]
    return {
        "rejected_items_cap": int(t["rejected_items_cap"]["value"]),
        "content_preview_chars": int(t["content_preview_chars"]),
    }


def stop_words() -> frozenset:
    return frozenset(bm25_params()["stop_words"])


def suffixes() -> Tuple[str, ...]:
    # Order is load-bearing: _stem returns on first match, so this must stay
    # longest-first exactly as the contract lists it.
    return tuple(bm25_params()["suffixes"])


__all__ = [
    "contract",
    "recall_query_patterns",
    "bm25_params",
    "thresholds",
    "telemetry_limits",
    "stop_words",
    "suffixes",
]
