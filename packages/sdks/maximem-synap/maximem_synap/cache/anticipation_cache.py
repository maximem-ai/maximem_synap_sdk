"""In-memory TTL cache for context bundles pushed over gRPC."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .bm25 import BM25, tokenize

logger = logging.getLogger("synap.sdk.cache.anticipation")

_DEFAULT_BM25_THRESHOLD = 1.5
_DEFAULT_NOVEL_TERM_THRESHOLD = 0.45


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
        ttl_seconds: int = 300,
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
        self._corpus_vocab: Set[str] = set()
        self._bm25: Optional[BM25] = None
        self._bm25_dirty: bool = True

    def store(self, bundle: Dict) -> None:
        """Store a bundle and index its items."""
        items_by_type = bundle.get("items_by_type", {})
        total_lt_items = sum(
            len(v) for v in items_by_type.values() if isinstance(v, list)
        )
        conv_ctx = bundle.get("conversation_context", {})
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

        entity_id = (
            bundle.get("_anticipation_user_id")
            or bundle.get("_anticipation_customer_id")
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

        conv_ctx = bundle.get("conversation_context", {})
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

    def lookup(
        self,
        search_query: Optional[List[str]] = None,
        entity_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        max_items: int = 10,
    ) -> Optional[Dict]:
        """Find cached items matching the query."""
        self._evict_expired()

        if not self._entries or not self._items:
            logger.info(
                "Cache lookup: EMPTY (entries=%d items=%d)",
                len(self._entries), len(self._items),
            )
            return None

        has_query = search_query and any(q.strip() for q in search_query if q)

        if not has_query:
            return self._freshness_lookup(entity_id)

        return self._item_lookup(
            search_query, entity_id, conversation_id, max_items,
        )

    def _item_lookup(
        self,
        search_query: List[str],
        entity_id: Optional[str],
        conversation_id: Optional[str],
        max_items: int,
    ) -> Optional[Dict]:
        query_text = " ".join(q for q in search_query if q)
        query_tokens = tokenize(query_text)
        if not query_tokens:
            return self._freshness_lookup(entity_id)

        unique_stems = set(query_tokens)
        novel_stems = unique_stems - self._corpus_vocab
        novel_ratio = len(novel_stems) / len(unique_stems) if unique_stems else 0

        if novel_ratio >= self._novel_term_threshold:
            logger.info(
                "Cache MISS (gate): ratio=%.0f%% threshold=%.0f%% query=%s",
                novel_ratio * 100,
                self._novel_term_threshold * 100,
                search_query[:80] if search_query else None,
            )
            return None

        if self._bm25_dirty or self._bm25 is None:
            corpus = [item.tokens for item in self._items]
            if not corpus:
                return None
            self._bm25 = BM25(corpus)
            self._bm25_dirty = False

        scores = self._bm25.scores(query_tokens)

        valid_bundles = self._get_valid_bundle_ids(entity_id, conversation_id)

        effective_threshold = max(
            0.6,
            min(self._bm25_threshold, 0.3 * len(query_tokens)),
        )

        scored_items: List[Tuple[float, _ItemRecord]] = []
        for idx, score in enumerate(scores):
            if score < effective_threshold:
                continue
            item = self._items[idx]
            if item.bundle_id not in valid_bundles:
                continue
            scored_items.append((score, item))

        if not scored_items:
            logger.info(
                "Cache MISS: best=%.2f threshold=%.2f query=%s",
                max(scores) if scores else 0,
                effective_threshold,
                search_query[:80] if search_query else None,
            )
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

    def _get_valid_bundle_ids(
        self,
        entity_id: Optional[str],
        conversation_id: Optional[str],
    ) -> Set[str]:
        """Return bundle ids that match the requested scope.

        Section 15 — privacy hardening:
        - When ``conversation_id`` is requested, the entry's stored
          ``conversation_id`` must match exactly. Bundles with a falsy
          (None / "") conversation_id are NOT eligible — that previously
          allowed cross-conversation leakage of cross-emitted bundles.
        - When ``entity_id`` is requested, the entry must match (or be the
          explicit ``"_any"`` sentinel that the SDK uses for client-scope
          fetches). Bundles whose stored entity_id doesn't match a specific
          entity_id request never participate.
        """
        valid = set()
        for bid, entry in self._entries.items():
            if entity_id is not None and entry.entity_id not in (entity_id, "_any"):
                continue
            if conversation_id is not None:
                # Strict match — no falsy fallback. A bundle that came in
                # without a conversation_id is not eligible for a
                # conversation-scoped lookup.
                if entry.conversation_id != conversation_id:
                    continue
            valid.add(bid)
        return valid

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
