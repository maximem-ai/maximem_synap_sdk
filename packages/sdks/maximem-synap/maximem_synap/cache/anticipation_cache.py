"""In-memory TTL cache for context bundles pushed over gRPC."""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .bm25 import BM25, tokenize

logger = logging.getLogger("synap.sdk.cache.anticipation")

_DEFAULT_BM25_THRESHOLD = 1.5
_DEFAULT_NOVEL_TERM_THRESHOLD = 0.45


def _honor_ttl_hint_enabled() -> bool:
    """WS3 of the echo-poisoning plan: when on, a bundle's server-sent
    ttl_hint_seconds bounds its lifetime (it can only SHORTEN the global
    TTL, never extend it) and cache hits stop renewing hinted bundles'
    leases. Default OFF — eviction behavior is byte-identical to today
    (global TTL only, LRU refresh on hit)."""
    return os.environ.get("SYNAP_SDK_CACHE_HONOR_TTL_HINT", "false").lower() in ("true", "1", "yes")

def recall_bypass_enabled() -> bool:
    """WS4 of the playground query-hygiene plan: when on, queries that are
    explicitly asking the agent to RECALL something from memory ("remind me…",
    "what do you have on file…", "where am I located again?") never serve from
    the anticipation cache — they always fall through to a fresh fetch.

    Rationale: for a recall-shaped question, a stale bundle false-hit is
    catastrophic (the agent answers "I don't see that" while the store holds
    the fact — the BZQR6J eval's cross-session failure mode), while the cost
    of bypassing is only cold-fetch latency on a turn the user expects the
    agent to go look something up. Stem coverage can't gate this (measured
    false hits at coverage=1.00). Default OFF — lookup behavior is
    byte-identical to today."""
    return os.environ.get("SYNAP_SDK_CACHE_RECALL_BYPASS", "false").lower() in ("true", "1", "yes")


# Conservative markers of a recall-shaped question. Matched case-insensitively
# against each raw query string. Deliberately phrase-level (not single words)
# so ordinary task queries ("book me a ride again") don't over-bypass; "again"
# alone is NOT a marker. Over-matching costs one cloud fetch of latency;
# under-matching risks a wrong "I don't have that" answer — so ties break
# toward including a pattern.
_RECALL_QUERY_PATTERNS = (
    re.compile(r"\bremind me\b", re.I),
    re.compile(r"\bon file\b", re.I),
    re.compile(r"\b(your|my|the) records\b", re.I),
    re.compile(r"\bdo you (have|know|remember|recall|see)\b", re.I),
    re.compile(r"\bwhat (do|did) (i|you) (say|tell|mention|have)\b", re.I),
    re.compile(r"\bwhat('s| is) my\b", re.I),
    re.compile(r"\bwhere (am i|do i live)\b", re.I),
    re.compile(r"\bhow long have i\b", re.I),
    re.compile(r"\bwhich of my\b", re.I),
    re.compile(r"\bwhat .{0,40}\b(again|earlier|last time|previously|yesterday)\b", re.I),
    re.compile(r"\bwill you use to (contact|reach|notify)\b", re.I),
    # 2026-07-16 eval-register additions — every phrasing below was measured
    # slipping past the gate and serving a stale bundle (issue #11):
    #   "can you confirm my email address?"          → confirm/verify my
    #   "what name is linked to my Uber account?"    → linked to my
    #   "best way to contact me with updates…"       → best way to reach me
    #   "which email are you going to use?"          → which … will you use
    # "confirm my booking" now also bypasses — intentional over-match, costs
    # one cloud fetch (the tie-break rule above).
    re.compile(r"\b(confirm|verify|double.?check)\b.{0,40}\bmy\b", re.I),
    re.compile(r"\b(linked to|associated with|registered (to|on|with)) my\b", re.I),
    re.compile(r"\b(best|preferred) way to (reach|contact|update|notify) me\b", re.I),
    re.compile(r"\bhow (do|will|can|should) you (reach|contact|notify) me\b", re.I),
    re.compile(r"\bwhich .{0,40}\b(are|will) you (going to )?(use|using)\b", re.I),
    re.compile(r"\bon (my|the) account\b", re.I),
)


def is_recall_query(search_query: Optional[List[str]]) -> bool:
    """True when any query string looks like an explicit memory-recall ask."""
    for q in search_query or []:
        if not q:
            continue
        for pat in _RECALL_QUERY_PATTERNS:
            if pat.search(q):
                return True
    return False


# The novel-term gate only fires once the corpus has enough vocabulary
# for the ratio to be statistically meaningful. Below this size, BM25's
# own threshold is the right filter — the novel-term ratio is dominated
# by ordinary English stop-words and routine word forms that haven't
# happened to appear in the few items stored so far.
#
# Empirically tuned: a fresh playground demo conversation accumulates
# ~60 stems after the first agent push and trips the 0.45 gate on
# essentially every second-turn query. Raising the floor to ~200 means
# the gate kicks in once a customer has ~3-4 typical bundles in cache.
_MIN_CORPUS_FOR_NOVEL_GATE = 200

# Hook callable signatures (optional; default no-op).
#
#   on_bundle_stored(bundle, *, entry, items_added, items_deduped) -> None
#   on_lookup(payload: dict) -> None
#       where payload includes: search_query, entity_id, customer_id,
#       client_id, conversation_id, cache_state, scope_filter_request,
#       scope_filter_accepted, novel_term_ratio, bm25_threshold,
#       bm25_query_tokens, items_picked, items_rejected, transport, hit,
#       latency_ms_local.
#
# Both hooks are exception-safe: the cache catches and logs any exception
# raised by the callback so a buggy hook can never break the SDK. Hooks
# are intended for in-process server-side debugging (e.g. the playground's
# Anticipation Monitoring telemetry). Customer SDK deployments leave the
# hooks unset → zero overhead.
BundleStoreHook = Callable[..., None]
LookupHook = Callable[..., None]


def invalidate_on_write_enabled() -> bool:
    """When true, SDK write paths drop the writing user's cached bundles.

    The cloud invalidates its own Redis caches on ingest but that stops at
    the process boundary — the in-SDK cache otherwise keeps serving the
    pre-write view until TTL (2026-07-16 eval register issue #8). Default
    OFF — write-path behavior is byte-identical to today."""
    return os.environ.get("SYNAP_SDK_CACHE_INVALIDATE_ON_WRITE", "false").lower() in ("true", "1", "yes")


def _max_entry_age_seconds(default: float) -> float:
    """Absolute lifetime cap for a cache entry, hit-renewals included.

    ``stored_at`` is a sliding lease (hits renew it), so before this cap a
    bundle under steady traffic never expired — a pre-knowledge-update
    snapshot could serve a retired value indefinitely (2026-07-16 eval
    register issue #6). ``SYNAP_SDK_CACHE_MAX_ENTRY_AGE`` overrides the
    default (2x the sliding TTL); ``0`` disables the cap."""
    raw = os.environ.get("SYNAP_SDK_CACHE_MAX_ENTRY_AGE", "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class _CacheEntry:
    bundle: Dict
    entity_id: str
    conversation_id: Optional[str]
    stored_at: float
    created_at: float = 0.0
    bundle_type: str = "anticipation"
    search_queries: List[str] = field(default_factory=list)
    # Section 16 — bundle composition extensions, captured at store time so
    # lookups can rank by confidence and honor a per-bundle TTL hint without
    # walking back into the raw bundle dict.
    confidence: float = 0.0
    origin_pattern_id: str = ""
    ttl_hint_seconds: int = 0


@dataclass
class _ItemRecord:
    content: str
    tokens: List[str]
    item_dict: Dict
    item_type: str
    bundle_id: str
    confidence: float = 1.0


class AnticipationCache:
    """In-memory TTL cache with item-level BM25 matching."""

    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_entries: int = 100,
        bm25_threshold: float = _DEFAULT_BM25_THRESHOLD,
        novel_term_threshold: float = _DEFAULT_NOVEL_TERM_THRESHOLD,
    ):
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._bm25_threshold = bm25_threshold
        self._novel_term_threshold = novel_term_threshold

        self._entries: Dict[str, _CacheEntry] = {}

        self._items: List[_ItemRecord] = []
        self._item_dedup: Set[str] = set()

        # Optional observability hooks. Default no-op. Server-side wrappers
        # (e.g. the playground) call register_*_hook to bridge lookup/store
        # events into their own telemetry pipeline.
        self._lookup_hook: Optional[LookupHook] = None
        self._store_hook: Optional[BundleStoreHook] = None

        # BM25 index state. Rebuilt lazily in match() when _bm25_dirty.
        # These were previously initialized at the tail of _fire_store_hook,
        # which both (a) left them undefined on a fresh cache and
        # (b) reset the index to empty after every successful store —
        # silently breaking item-level BM25 matching.
        self._corpus_vocab: Set[str] = set()
        self._bm25: Optional[BM25] = None
        self._bm25_dirty: bool = True

    def register_lookup_hook(self, hook: Optional[LookupHook]) -> None:
        """Install a callback fired AFTER every cache lookup.

        The callback receives a single keyword payload with everything a
        debugger needs: the request scope, the cache state at lookup time,
        the scope-filter result, the BM25 threshold + per-item scores, the
        items picked, the items rejected (with reasons).

        Pass ``None`` to clear. Exceptions raised by the hook are caught
        and logged so a buggy hook can't break the SDK.
        """
        self._lookup_hook = hook

    def register_store_hook(self, hook: Optional[BundleStoreHook]) -> None:
        """Install a callback fired AFTER a bundle lands in the cache.

        The callback receives the raw bundle dict + a summary of what was
        indexed. Used by the playground's telemetry pipeline to attribute
        bundle-store events to a turn record.

        Pass ``None`` to clear. Exception-safe.
        """
        self._store_hook = hook

    def _fire_lookup_hook(self, payload: Dict[str, Any]) -> None:
        if self._lookup_hook is None:
            return
        try:
            self._lookup_hook(payload)
        except Exception:
            logger.debug("anticipation_cache lookup hook raised", exc_info=True)

    def _fire_store_hook(
        self,
        bundle: Dict[str, Any],
        *,
        entry: _CacheEntry,
        items_added: int,
        items_deduped: int,
    ) -> None:
        if self._store_hook is None:
            return
        try:
            self._store_hook(
                bundle,
                entry=entry,
                items_added=items_added,
                items_deduped=items_deduped,
            )
        except Exception:
            logger.debug("anticipation_cache store hook raised", exc_info=True)

    def store(self, bundle: Dict) -> None:
        """Store a bundle and index its items."""
        items_by_type = bundle.get("items_by_type", {})
        total_lt_items = sum(
            len(v) for v in items_by_type.values() if isinstance(v, list)
        )
        conv_ctx = bundle.get("conversation_context") or {}
        has_conv_context = bool(
            conv_ctx.get("summary")
            or conv_ctx.get("recent_turns")
            or conv_ctx.get("key_extractions", {}).get("facts")
        )
        if total_lt_items == 0 and not has_conv_context:
            logger.debug(
                "Skipping empty bundle: bundle_id=%s",
                bundle.get("bundle_id", "?"),
            )
            return

        self._evict_expired()

        if len(self._entries) >= self._max_entries:
            oldest_key = min(
                self._entries,
                key=lambda k: self._entries[k].stored_at,
            )
            self._remove_bundle(oldest_key)

        # Funnel store — pick the narrowest non-empty scope marker the
        # producer set. Bundles pushed at customer-scope (B2B account-level)
        # land under the customer_id; client-scope bundles (company
        # knowledge, FAQs) land under the client_id; the "_any" sentinel
        # remains for bundles with no scope context.
        #
        # The matching funnel lookup widens at request time — see
        # ``_get_valid_bundle_ids`` below. Mirrors the narrow-to-broad
        # vector-filter pattern from
        # ``RetrievalManager._build_customer_scope_filter``.
        entity_id = (
            bundle.get("_anticipation_user_id")
            or bundle.get("_anticipation_customer_id")
            or bundle.get("_anticipation_client_id")
            or "_any"
        )
        conversation_id = bundle.get("_anticipation_conversation_id")
        bundle_type = bundle.get("_bundle_type", "anticipation")
        bundle_id = bundle.get("bundle_id", str(time.monotonic()))
        search_queries = bundle.get("search_queries", [])

        store_time = time.monotonic()
        self._entries[bundle_id] = _CacheEntry(
            bundle=bundle,
            entity_id=entity_id,
            conversation_id=conversation_id,
            stored_at=store_time,
            created_at=store_time,
            bundle_type=bundle_type,
            search_queries=search_queries,
            confidence=float(bundle.get("_bundle_confidence", 0.0) or 0.0),
            origin_pattern_id=bundle.get("_origin_pattern_id", "") or "",
            ttl_hint_seconds=int(bundle.get("_ttl_hint_seconds", 0) or 0),
        )

        items_by_type = bundle.get("items_by_type", {})
        items_added = 0
        items_deduped = 0
        for item_type, items_list in items_by_type.items():
            if not isinstance(items_list, list):
                continue
            for item_dict in items_list:
                content = item_dict.get("content", "")
                if not content:
                    continue
                dedup_key = self._dedup_key(entity_id, content)
                if dedup_key in self._item_dedup:
                    items_deduped += 1
                    continue
                self._item_dedup.add(dedup_key)
                tokens = tokenize(content)
                if not tokens:
                    continue
                self._items.append(_ItemRecord(
                    content=content,
                    tokens=tokens,
                    item_dict=item_dict,
                    item_type=item_type,
                    bundle_id=bundle_id,
                    confidence=item_dict.get("confidence", 1.0),
                ))
                self._corpus_vocab.update(tokens)
                items_added += 1

        conv_ctx = bundle.get("conversation_context") or {}
        key_ext = conv_ctx.get("key_extractions", {}) if conv_ctx else {}
        for ext_type in ("facts", "decisions", "preferences", "constraints"):
            ext_items = key_ext.get(ext_type, [])
            if not isinstance(ext_items, list):
                continue
            for ext_item in ext_items:
                content = ext_item.get("content", "")
                if not content:
                    continue
                dedup_key = self._dedup_key(entity_id, content)
                if dedup_key in self._item_dedup:
                    items_deduped += 1
                    continue
                self._item_dedup.add(dedup_key)
                tokens = tokenize(content)
                if not tokens:
                    continue
                self._items.append(_ItemRecord(
                    content=content,
                    tokens=tokens,
                    item_dict=ext_item,
                    item_type=ext_type,
                    bundle_id=bundle_id,
                    confidence=float(ext_item.get("confidence", 1.0)) if ext_item.get("confidence") not in ("explicit", "inferred", "assumed") else 1.0,
                ))
                self._corpus_vocab.update(tokens)
                items_added += 1

        self._bm25_dirty = True

        logger.info(
            "Bundle stored: bundle_id=%s type=%s queries=%s "
            "items_indexed=%d deduped=%d total_items=%d total_bundles=%d entity=%s conv=%s",
            bundle_id,
            bundle_type,
            search_queries,
            items_added,
            items_deduped,
            len(self._items),
            len(self._entries),
            entity_id,
            conversation_id,
        )

        # Observability hook — server-side telemetry pipelines bridge this
        # into a per-turn record. Exception-safe.
        self._fire_store_hook(
            bundle,
            entry=self._entries[bundle_id],
            items_added=items_added,
            items_deduped=items_deduped,
        )

    def lookup(
        self,
        search_query: Optional[List[str]] = None,
        entity_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        max_items: int = 10,
        *,
        customer_id: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """Find cached items matching the query.

        Funnel scope (mirrors RetrievalManager's narrow-to-broad pattern):

        - ``entity_id``   the user_id of the requesting visitor (narrowest)
        - ``customer_id`` the customer scope (B2B account, or = user in B2C)
        - ``client_id``   the SDK's bound client (broadest; agent-wide)

        A stored bundle matches if its ``entry.entity_id`` is in
        ``{entity_id, customer_id, client_id, "_any"}``. Pass them all when
        known — the lookup will accept the widest applicable tier.

        ``customer_id`` and ``client_id`` are keyword-only to keep the
        existing positional signature compatible with older callers; those
        callers still work but only match user-scope and ``"_any"`` bundles.
        """
        self._evict_expired()

        if not self._entries or not self._items:
            logger.info(
                "Cache lookup: EMPTY (entries=%d items=%d)",
                len(self._entries), len(self._items),
            )
            self._fire_lookup_hook({
                "search_query": list(search_query or []),
                "entity_id": entity_id,
                "customer_id": customer_id,
                "client_id": client_id,
                "conversation_id": conversation_id,
                "cache_state": self._snapshot_state(),
                "scope_filter_request": {
                    "entity_id": entity_id,
                    "customer_id": customer_id,
                    "client_id": client_id,
                    "conversation_id": conversation_id,
                },
                "scope_filter_accepted": sorted(
                    self._build_accepted_scope(entity_id, customer_id, client_id)
                ),
                "novel_term_ratio": None,
                "bm25_threshold": None,
                "bm25_query_tokens": [],
                "items_picked": [],
                "items_rejected": [],
                "hit": False,
                "exit_reason": "empty",
            })
            return None

        has_query = search_query and any(q.strip() for q in search_query if q)

        if not has_query:
            return self._freshness_lookup(entity_id)

        # WS4 recall bypass: an explicit memory-recall question must never be
        # answered from a pre-fetched bundle — a stale false-hit here makes the
        # agent deny a fact the store holds. Fires BEFORE BM25 scoring: the
        # measured failure mode hit at score 2.86 / coverage 1.00, so no
        # score-side gate can catch it.
        if recall_bypass_enabled() and is_recall_query(search_query):
            logger.info(
                "Cache BYPASS (recall-shaped query): query=%s",
                [q[:80] for q in search_query if q][:3],
            )
            self._fire_lookup_hook({
                "search_query": list(search_query or []),
                "entity_id": entity_id,
                "customer_id": customer_id,
                "client_id": client_id,
                "conversation_id": conversation_id,
                "cache_state": self._snapshot_state(),
                "scope_filter_request": {
                    "entity_id": entity_id,
                    "customer_id": customer_id,
                    "client_id": client_id,
                    "conversation_id": conversation_id,
                },
                "scope_filter_accepted": sorted(
                    self._build_accepted_scope(entity_id, customer_id, client_id)
                ),
                "novel_term_ratio": None,
                "bm25_threshold": None,
                "bm25_query_tokens": [],
                "items_picked": [],
                "items_rejected": [],
                "hit": False,
                "exit_reason": "recall_bypass",
            })
            return None

        return self._item_lookup(
            search_query, entity_id, conversation_id, max_items,
            customer_id=customer_id, client_id=client_id,
        )

    def _item_lookup(
        self,
        search_query: List[str],
        entity_id: Optional[str],
        conversation_id: Optional[str],
        max_items: int,
        *,
        customer_id: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> Optional[Dict]:
        # Pre-build the telemetry skeleton — the hook fires on every exit
        # path. Most fields are filled below; we never reach `return None`
        # without populating them.
        hook_payload: Dict[str, Any] = {
            "search_query": list(search_query or []),
            "entity_id": entity_id,
            "customer_id": customer_id,
            "client_id": client_id,
            "conversation_id": conversation_id,
            "cache_state": self._snapshot_state(),
            "scope_filter_request": {
                "entity_id": entity_id,
                "customer_id": customer_id,
                "client_id": client_id,
                "conversation_id": conversation_id,
            },
            "scope_filter_accepted": sorted(
                self._build_accepted_scope(entity_id, customer_id, client_id)
            ),
            "novel_term_ratio": None,
            "bm25_threshold": None,
            "bm25_query_tokens": [],
            "items_picked": [],
            "items_rejected": [],
            "hit": False,
            "exit_reason": "unknown",
        }

        query_text = " ".join(q for q in search_query if q)
        query_tokens = tokenize(query_text)
        if not query_tokens:
            hook_payload["exit_reason"] = "no_query_tokens"
            self._fire_lookup_hook(hook_payload)
            return self._freshness_lookup(entity_id)

        hook_payload["bm25_query_tokens"] = list(query_tokens)

        unique_stems = set(query_tokens)
        novel_stems = unique_stems - self._corpus_vocab
        novel_ratio = len(novel_stems) / len(unique_stems) if unique_stems else 0
        hook_payload["novel_term_ratio"] = round(float(novel_ratio), 4)

        # The novel-term gate is a cheap early-out for queries about topics
        # the cache has never indexed — it skips BM25 when too many query
        # stems aren't in the corpus. Useful for steady-state caches where
        # the corpus is broad (thousands of stems). Counter-productive at
        # cold start: a 60-stem corpus naturally trips the gate on nearly
        # any new English sentence even when relevant items are present.
        #
        # We only enforce the gate once the corpus is large enough that
        # the ratio is a meaningful signal. Below ``_MIN_CORPUS_FOR_GATE``
        # we let BM25 do the actual relevance test — it's the right
        # filter at small corpus sizes anyway.
        if (
            len(self._corpus_vocab) >= _MIN_CORPUS_FOR_NOVEL_GATE
            and novel_ratio >= self._novel_term_threshold
        ):
            logger.info(
                "Cache MISS (gate): ratio=%.0f%% threshold=%.0f%% corpus=%d query=%s",
                novel_ratio * 100,
                self._novel_term_threshold * 100,
                len(self._corpus_vocab),
                search_query[:80] if search_query else None,
            )
            hook_payload["exit_reason"] = "novel_term_gate"
            self._fire_lookup_hook(hook_payload)
            return None

        if self._bm25_dirty or self._bm25 is None:
            corpus = [item.tokens for item in self._items]
            if not corpus:
                hook_payload["exit_reason"] = "empty_corpus"
                self._fire_lookup_hook(hook_payload)
                return None
            self._bm25 = BM25(corpus)
            self._bm25_dirty = False

        scores = self._bm25.scores(query_tokens)

        valid_bundles = self._get_valid_bundle_ids(
            entity_id, conversation_id,
            customer_id=customer_id, client_id=client_id,
        )

        effective_threshold = max(
            0.6,
            min(self._bm25_threshold, 0.3 * len(query_tokens)),
        )
        hook_payload["bm25_threshold"] = round(float(effective_threshold), 4)

        # Score all items + categorize each as picked / rejected with a
        # specific reason. items_rejected is bounded to top-20 by absolute
        # score so a fat cache doesn't bloat the telemetry payload.
        scored_items: List[Tuple[float, _ItemRecord]] = []
        rejected_with_reason: List[Tuple[float, Dict[str, Any]]] = []
        for idx, score in enumerate(scores):
            item = self._items[idx]
            score_f = float(score)
            if score_f < effective_threshold:
                rejected_with_reason.append((score_f, {
                    "item_index": idx,
                    "bundle_id": item.bundle_id,
                    "item_type": item.item_type,
                    "content_first_120": item.content[:120],
                    "bm25_score": round(score_f, 4),
                    "reason": "below_threshold",
                }))
                continue
            if item.bundle_id not in valid_bundles:
                rejected_with_reason.append((score_f, {
                    "item_index": idx,
                    "bundle_id": item.bundle_id,
                    "item_type": item.item_type,
                    "content_first_120": item.content[:120],
                    "bm25_score": round(score_f, 4),
                    "reason": "scope_filter_excluded",
                }))
                continue
            scored_items.append((score, item))

        # Keep the top-20 rejected items by score (those closest to a hit).
        rejected_with_reason.sort(key=lambda x: -x[0])
        hook_payload["items_rejected"] = [r[1] for r in rejected_with_reason[:20]]

        if not scored_items:
            logger.info(
                "Cache MISS: best=%.2f threshold=%.2f query=%s",
                max(scores) if scores else 0,
                effective_threshold,
                search_query[:80] if search_query else None,
            )
            hook_payload["exit_reason"] = "no_items_above_threshold"
            self._fire_lookup_hook(hook_payload)
            return None

        scored_items.sort(key=lambda x: -x[0])
        top_items = scored_items[:max_items]

        # Coverage: fraction of the query's stems present in the picked
        # items. A bundle can clear the BM25 threshold on one shared token
        # ("account") while lacking the asked-for fact entirely — the false
        # HIT behind the eval's "forgot the name" failure. Always computed
        # and logged (telemetry-first); ENFORCED only when
        # SYNAP_SDK_CACHE_COVERAGE_MIN is explicitly set, so default
        # behavior is byte-identical.
        picked_stems: Set[str] = set()
        for _, item in top_items:
            picked_stems.update(item.tokens)
        coverage = (
            len(unique_stems & picked_stems) / len(unique_stems)
            if unique_stems else 1.0
        )
        hook_payload["coverage"] = round(float(coverage), 4)

        coverage_min_raw = os.environ.get("SYNAP_SDK_CACHE_COVERAGE_MIN", "").strip()
        if coverage_min_raw:
            try:
                coverage_min = float(coverage_min_raw)
            except ValueError:
                coverage_min = None  # malformed setting: observe-only
            if coverage_min is not None and coverage < coverage_min:
                # Gate BEFORE any side effects (no stored_at refresh) so a
                # rejected lookup leaves the cache state untouched.
                logger.info(
                    "Cache MISS (coverage): coverage=%.2f min=%.2f best=%.2f query=%s",
                    coverage, coverage_min,
                    float(top_items[0][0]) if top_items else 0.0,
                    search_query[:80] if search_query else None,
                )
                hook_payload["exit_reason"] = "coverage_gate"
                self._fire_lookup_hook(hook_payload)
                return None

        items_by_type: Dict[str, list] = {}
        for score, item in top_items:
            items_by_type.setdefault(item.item_type, []).append(item.item_dict)

        bundle_ids_used = {item.bundle_id for _, item in top_items}

        now = time.monotonic()
        refresh_on_hit = not _honor_ttl_hint_enabled()
        for bid in bundle_ids_used:
            if bid in self._entries:
                entry = self._entries[bid]
                # With TTL-hint honoring on, hinted bundles must age out on
                # their clock — a hit must not renew a stale snapshot's lease
                # (that lease-renewal is how a pre-knowledge-update bundle
                # kept serving a retired value indefinitely under steady
                # traffic). Un-hinted bundles keep today's LRU refresh.
                if refresh_on_hit or entry.ttl_hint_seconds <= 0:
                    entry.stored_at = now

        base_entry = max(
            (self._entries[bid] for bid in bundle_ids_used if bid in self._entries),
            key=lambda e: e.stored_at,
            default=None,
        )

        best_score = top_items[0][0]

        logger.info(
            "Cache HIT: score=%.2f threshold=%.2f items=%d coverage=%.2f query=%s",
            best_score,
            effective_threshold,
            len(top_items),
            coverage,
            search_query[:80] if search_query else None,
        )

        # Populate the picked items for the hook before returning.
        # Each picked item records the bm25_score, its origin bundle, and
        # a content preview for the dashboard's drill-down.
        for rank, (score, item) in enumerate(top_items):
            hook_payload["items_picked"].append({
                "item_index": self._items.index(item) if item in self._items else rank,
                "bundle_id": item.bundle_id,
                "item_type": item.item_type,
                "content_first_240": item.content[:240],
                "bm25_score": round(float(score), 4),
                "passed_threshold": True,
                "confidence": float(item.confidence),
            })
        hook_payload["hit"] = True
        hook_payload["exit_reason"] = "hit"
        self._fire_lookup_hook(hook_payload)

        return {
            "bundle_id": f"anticipation_merged_{int(time.monotonic())}",
            "items_by_type": items_by_type,
            "items": [item.item_dict for _, item in top_items],
            "cache_hit": True,
            "source": "anticipation_cache",
            "search_queries": base_entry.search_queries if base_entry else [],
            "search_keywords": [],
            "source_bundle_ids": sorted(bundle_ids_used),
            "_anticipation_user_id": base_entry.entity_id if base_entry else None,
            "_anticipation_conversation_id": base_entry.conversation_id if base_entry else None,
            "_bundle_type": "anticipation",
        }

    def _build_accepted_scope(
        self,
        entity_id: Optional[str],
        customer_id: Optional[str],
        client_id: Optional[str],
    ) -> Set[str]:
        """The widened scope-match set for a request. Mirrors
        RetrievalManager._build_customer_scope_filter — a narrower-scope
        request also matches broader-scope bundles for the same customer/
        client. ``"_any"`` is always accepted; falsy IDs are dropped so an
        empty customer_id doesn't accidentally match bundles keyed at "".
        """
        accepted: Set[str] = {"_any"}
        if entity_id:
            accepted.add(entity_id)
        if customer_id:
            accepted.add(customer_id)
        if client_id:
            accepted.add(client_id)
        return accepted

    def _get_valid_bundle_ids(
        self,
        entity_id: Optional[str],
        conversation_id: Optional[str],
        *,
        customer_id: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> Set[str]:
        """Return bundle ids that match the requested scope.

        Dual-scope conversation lookup. When a consumer asks with a
        specific ``conversation_id``, we accept two flavours of bundle:

        1. **Exact-match conversation-scope** — ``entry.conversation_id ==
           conversation_id``. This is the bundle the producer pushed
           specifically for this conversation.
        2. **User-scope fallback** — ``entry.conversation_id is None``.
           These bundles weren't tied to any conversation at push time
           (e.g. a profile bundle pushed at ``session_start`` for the
           visitor's user_id); they're naturally applicable to any
           conversation that user runs.

        Bundles tied to a *different* conversation are still rejected —
        the entity_id widening above plus this user-scope fallback gives
        the same coverage as two separate lookups (one user-scope, one
        conversation-scope) without the round-trip.

        ``entity_id`` continues to use the widened scope tiers (see
        ``_build_accepted_scope``) so customer- and client-shared
        bundles remain reachable from a user-scope request.
        """
        accepted_scope = self._build_accepted_scope(entity_id, customer_id, client_id)
        valid: Set[str] = set()
        for bid, entry in self._entries.items():
            if entity_id is not None and entry.entity_id not in accepted_scope:
                continue
            if conversation_id is not None:
                # Accept exact match OR a user-scope bundle (no conv_id
                # at push time). Rejecting both used to silently miss
                # the agent's session_start profile bundle whenever the
                # consumer queried with conv_id set.
                if entry.conversation_id is not None and entry.conversation_id != conversation_id:
                    continue
            valid.add(bid)
        return valid

    def _snapshot_state(self) -> Dict[str, Any]:
        """Compact summary of the cache's contents — bound the size for
        telemetry so a fat cache doesn't blow up a row. Used by the
        lookup-hook payload."""
        scope_breakdown: Dict[str, int] = {}
        for entry in self._entries.values():
            scope_breakdown[entry.entity_id] = scope_breakdown.get(entry.entity_id, 0) + 1
        return {
            "total_entries": len(self._entries),
            "total_items": len(self._items),
            "corpus_vocab_size": len(self._corpus_vocab),
            "scope_breakdown": scope_breakdown,
        }

    def _freshness_lookup(
        self,
        entity_id: Optional[str] = None,
    ) -> Optional[Dict]:
        summary_candidates = {
            bid: e for bid, e in self._entries.items()
            if e.bundle_type == "user_summary"
            and (entity_id is None or e.entity_id in (entity_id, "_any"))
        }
        if not summary_candidates:
            return None

        freshest_bid = max(summary_candidates, key=lambda k: summary_candidates[k].stored_at)
        # Lease renewal on read follows the same rule as item hits: with
        # TTL-hint honoring on, a read must not extend a snapshot's life.
        if not _honor_ttl_hint_enabled():
            summary_candidates[freshest_bid].stored_at = time.monotonic()
        return summary_candidates[freshest_bid].bundle

    def lookup_user_summary(
        self,
        entity_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """Return the freshest user_summary bundle for ``entity_id``.

        Section 15 — privacy hardening: callers MUST supply an ``entity_id``.
        Previously a missing entity_id returned the freshest summary across
        all users in the cache, which could splice User A's summary into
        User B's response on the conversation-scope path. We now refuse the
        lookup unless an entity_id is provided.
        """
        if not entity_id:
            logger.debug(
                "lookup_user_summary called without entity_id — refusing "
                "to avoid cross-user summary leakage"
            )
            return None

        self._evict_expired()
        candidates = {
            bid: entry for bid, entry in self._entries.items()
            if entry.bundle_type == "user_summary"
            and entry.entity_id in (entity_id, "_any")
        }
        if not candidates:
            return None
        freshest_bid = max(candidates, key=lambda k: candidates[k].stored_at)
        if not _honor_ttl_hint_enabled():
            candidates[freshest_bid].stored_at = time.monotonic()
        return candidates[freshest_bid].bundle

    def _remove_bundle(self, bundle_id: str) -> None:
        if bundle_id in self._entries:
            del self._entries[bundle_id]
        self._items = [i for i in self._items if i.bundle_id != bundle_id]
        self._bm25_dirty = True
        self._rebuild_vocab()

    def _dedup_key(self, entity_id: str, content: str) -> str:
        """Scope-qualified item-dedup key.

        Keying on content alone made the dedup set cache-GLOBAL: with the
        one-SDK-per-instance sharing model, visitor B's byte-identical seed
        facts were swallowed against visitor A's copy, never indexed under
        B's bundle — so B was permanently scope_filter_excluded for that
        content and always cold-fetched (2026-07-16 eval register issue #9).
        Scoping the key keeps within-visitor dedup intact."""
        return f"{entity_id}|{content.lower().strip()[:120]}"

    def _rebuild_vocab(self) -> None:
        self._corpus_vocab = set()
        self._item_dedup = set()
        for item in self._items:
            self._corpus_vocab.update(item.tokens)
            entry = self._entries.get(item.bundle_id)
            entity_id = entry.entity_id if entry else "_any"
            self._item_dedup.add(self._dedup_key(entity_id, item.content))

    def _effective_ttl(self, entry: "_CacheEntry", honor_hint: bool) -> float:
        """Global TTL, optionally bounded by the bundle's server-sent hint.
        The hint can only SHORTEN (a hint longer than the global TTL is
        capped); missing/zero hint means the global TTL applies."""
        if honor_hint and entry.ttl_hint_seconds > 0:
            return min(self._ttl, entry.ttl_hint_seconds)
        return self._ttl

    def _evict_expired(self) -> None:
        now = time.monotonic()
        honor_hint = _honor_ttl_hint_enabled()
        # Absolute cap on top of the sliding lease: hit-renewals move
        # stored_at but never created_at, so no bundle outlives the cap no
        # matter how much traffic it serves. Entries from before this field
        # existed fall back to stored_at.
        max_age = _max_entry_age_seconds(2.0 * self._ttl)
        expired = [
            bid
            for bid, entry in self._entries.items()
            if now - entry.stored_at > self._effective_ttl(entry, honor_hint)
            or (max_age > 0 and now - (entry.created_at or entry.stored_at) > max_age)
        ]
        if expired:
            for bid in expired:
                del self._entries[bid]
            self._items = [
                i for i in self._items
                if i.bundle_id not in set(expired)
            ]
            self._bm25_dirty = True
            self._rebuild_vocab()

    def invalidate_entity(self, entity_id: str) -> int:
        """Drop every bundle whose funnel scope is ``entity_id``.

        Write-path hook (2026-07-16 eval register issue #8): the cloud
        invalidates its Redis L1/L2/L3 on ingest, but that stops at the
        process boundary — nothing ever told this in-process cache a write
        happened, so a pre-write bundle kept serving the old view for its
        whole TTL. Callers with a write path (SDK ingest, the playground's
        turn ingestion) invalidate the writing user's scope; the next
        lookup cold-fetches and the follow-up push repopulates.

        Returns the number of bundles dropped. ``"_any"``-scoped bundles
        are left alone — they carry client-shared content, not the written
        user's state.
        """
        if not entity_id:
            return 0
        doomed = [
            bid for bid, entry in self._entries.items()
            if entry.entity_id == entity_id
        ]
        for bid in doomed:
            del self._entries[bid]
        if doomed:
            doomed_set = set(doomed)
            self._items = [i for i in self._items if i.bundle_id not in doomed_set]
            self._bm25_dirty = True
            self._rebuild_vocab()
            logger.info(
                "Cache invalidated on write: entity=%s bundles_dropped=%d",
                entity_id, len(doomed),
            )
        return len(doomed)

    def clear(self) -> None:
        self._entries.clear()
        self._items.clear()
        self._item_dedup.clear()
        self._corpus_vocab.clear()
        self._bm25 = None
        self._bm25_dirty = True
