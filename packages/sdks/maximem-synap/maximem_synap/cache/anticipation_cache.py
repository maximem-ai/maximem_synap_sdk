"""In-memory TTL cache for context bundles pushed over gRPC."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .bm25 import BM25, tokenize

logger = logging.getLogger("synap.sdk.cache.anticipation")

_DEFAULT_BM25_THRESHOLD = 1.5
_DEFAULT_NOVEL_TERM_THRESHOLD = 0.45

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


@dataclass
class _CacheEntry:
    bundle: Dict
    entity_id: str
    conversation_id: Optional[str]
    stored_at: float
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

        self._entries[bundle_id] = _CacheEntry(
            bundle=bundle,
            entity_id=entity_id,
            conversation_id=conversation_id,
            stored_at=time.monotonic(),
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
                dedup_key = content.lower().strip()[:120]
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
                dedup_key = content.lower().strip()[:120]
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

        items_by_type: Dict[str, list] = {}
        for score, item in top_items:
            items_by_type.setdefault(item.item_type, []).append(item.item_dict)

        bundle_ids_used = {item.bundle_id for _, item in top_items}

        now = time.monotonic()
        for bid in bundle_ids_used:
            if bid in self._entries:
                self._entries[bid].stored_at = now

        base_entry = max(
            (self._entries[bid] for bid in bundle_ids_used if bid in self._entries),
            key=lambda e: e.stored_at,
            default=None,
        )

        best_score = top_items[0][0]

        logger.info(
            "Cache HIT: score=%.2f threshold=%.2f items=%d query=%s",
            best_score,
            effective_threshold,
            len(top_items),
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
        candidates[freshest_bid].stored_at = time.monotonic()
        return candidates[freshest_bid].bundle

    def _remove_bundle(self, bundle_id: str) -> None:
        if bundle_id in self._entries:
            del self._entries[bundle_id]
        self._items = [i for i in self._items if i.bundle_id != bundle_id]
        self._bm25_dirty = True
        self._rebuild_vocab()

    def _rebuild_vocab(self) -> None:
        self._corpus_vocab = set()
        self._item_dedup = set()
        for item in self._items:
            self._corpus_vocab.update(item.tokens)
            self._item_dedup.add(item.content.lower().strip()[:120])

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [
            bid
            for bid, entry in self._entries.items()
            if now - entry.stored_at > self._ttl
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

    def clear(self) -> None:
        self._entries.clear()
        self._items.clear()
        self._item_dedup.clear()
        self._corpus_vocab.clear()
        self._bm25 = None
        self._bm25_dirty = True
