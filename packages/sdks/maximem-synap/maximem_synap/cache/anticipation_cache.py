"""In-memory TTL cache for context bundles pushed over gRPC."""

import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..behavior import recall_query_patterns, semantic_match_params, thresholds
from .bm25 import BM25, tokenize

logger = logging.getLogger("synap.sdk.cache.anticipation")

# Tuned values live in the shared behavior contract so the Python and JS SDKs
# cannot drift. See synap/sdk/CONTRACT/README.md. Do NOT re-inline them here.
_DEFAULT_BM25_THRESHOLD = thresholds()["bm25"]
_DEFAULT_NOVEL_TERM_THRESHOLD = thresholds()["novel_term"]
_SEMANTIC = semantic_match_params()
MATCHERS = ("bm25", "semantic")


def _matcher_setting() -> str:
    """``SYNAP_SDK_CACHE_MATCHER`` (bm25 or semantic) over the contract's mode."""
    raw = os.environ.get("SYNAP_SDK_CACHE_MATCHER", "").strip().lower()
    return raw if raw in MATCHERS else _SEMANTIC["mode"]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _asked_kind_patterns(queries: List[str]) -> List["re.Pattern[str]"]:
    """Value patterns for the kinds of value the queries ask for (a time of
    day, a duration, an amount), from semantic_match.asked_details."""
    out = []
    for name, spec in (_SEMANTIC.get("asked_details") or {}).items():
        if name.startswith("$") or not isinstance(spec, dict):
            continue
        if any(re.search(spec["query"], q, re.IGNORECASE) for q in queries):
            out.append(re.compile(spec["value"], re.IGNORECASE))
    return out


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
# under-matching risks a wrong "I don't have that" answer -- so ties break
# toward including a pattern.
#
# The patterns themselves, and the dated eval-failure note behind each one,
# live in the shared behavior contract and are loaded at import time. Adding
# a pattern here instead of there changes Python's behavior without changing
# JS's, which is precisely the drift this contract exists to prevent.
_RECALL_QUERY_PATTERNS = recall_query_patterns()


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
_MIN_CORPUS_FOR_NOVEL_GATE = thresholds()["min_corpus_for_novel_gate"]
_EFFECTIVE_FLOOR = thresholds()["effective_floor"]
_QUERY_TOKEN_SCALE = thresholds()["query_token_scale"]

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


# ── the rung a request or a bundle stands at ────────────────────────────────
#
# A pushed bundle is stamped with the scope rung the server retrieved it at,
# and a fetch names one through `scope_path`. Both are written the same way so
# they can be compared as strings, because nothing in this process can resolve
# anything: the server speaks in registry node ids and this SDK has never seen
# one, so the comparison happens in the vocabulary the CALLER uses, which is
# level keys and the client's own external ids.
#
# The recipe and its golden vectors live in the shared behavior contract
# (CONTRACT/behavior/anticipation.json, `rung_key`), because it is written
# three times: here, in the JS SDK, and on the server, and the producer is not
# an SDK so the usual "port it once" rule cannot apply. Do NOT re-tune it here.
_RUNG_SEPARATOR = "|"
_RUNG_ASSIGN = "="


def rung_key(scope_path: Optional[Dict[str, str]]) -> Optional[str]:
    """The rung a `scope_path` names, canonically, or None for no path.

    Sorted by level key because a dict carries whatever order the caller wrote
    it in, and the ladder's own order is not knowable here. Only the value is
    escaped: level keys are `^[a-z][a-z0-9_]{0,63}$`, so they can hold neither
    separator. Backslash is escaped first, or `a|b` and `a\\pb` would encode to
    the same string, which is a false match between two different rungs.
    """
    if not scope_path:
        return None
    pairs = [
        (str(k), str(v))
        for k, v in scope_path.items()
        if k is not None and v is not None and str(k) != "" and str(v) != ""
    ]
    if not pairs:
        return None
    pairs.sort(key=lambda kv: kv[0])
    return _RUNG_SEPARATOR.join(
        f"{k}{_RUNG_ASSIGN}"
        + v.replace("\\", "\\\\")
           .replace(_RUNG_SEPARATOR, "\\p")
           .replace(_RUNG_ASSIGN, "\\e")
        for k, v in pairs
    )


def _rung_serves(bundle_rung: Optional[str], request_rung: Optional[str]) -> bool:
    """May a bundle stamped `bundle_rung` answer a request at `request_rung`?

    Equality, and equality on purpose. A subset rule ("the bundle names more
    rungs than you did, close enough") reads as the generous option and is the
    dangerous one: the rung nobody named is exactly the rung that tells two
    people who share an external id apart, which is the case the whole ladder
    exists for. Being wrong here costs a cold fetch; being wrong the other way
    hands one person's context to another.

    None on either side means "named no rung". Both None is every request in
    production today, and it serves, so nothing changes for anyone who has not
    turned nested scoping on. `""` is normalised to None because a proto string
    field has no null and an unstamped bundle arrives as the empty string.
    """
    return (bundle_rung or None) == (request_rung or None)


def _unowned_elsewhere(entry_entity: str, entry_conversation: Optional[str],
                       conversation_id: Optional[str]) -> bool:
    """A bundle the server pushed with no user or customer stamp, tied to a
    conversation other than the one this request names (or to one when the
    request names none).

    Nothing on such a bundle says whose it is: it is stamped "_any" because
    the request that triggered it named no user. Its conversation is the only
    owner it has, so outside that conversation it is served to nobody.
    Without this, a user-scope or client-scope fetch, which names no
    conversation, could be answered from another user's conversation.
    """
    return (entry_entity == "_any" and entry_conversation is not None
            and entry_conversation != conversation_id)


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
    # The scope rung the server retrieved this bundle at, as `rung_key` writes
    # one. None for a bundle whose request named no rung, which is every bundle
    # a client without nested scoping receives.
    scope_rung: Optional[str] = None


@dataclass
class _ItemRecord:
    content: str
    tokens: List[str]
    item_dict: Dict
    item_type: str
    bundle_id: str
    confidence: float = 1.0
    # The item's vector under `vec_model`, filled off the lookup path once a
    # model is loaded. None until then: the item is matched by BM25 alone.
    vec: Optional[List[float]] = None
    vec_model: str = ""
    # The questions the memory answers (its HQEs), sent with the bundle item,
    # and their vectors under `vec_model`.
    questions: List[str] = field(default_factory=list)
    qvecs: List[List[float]] = field(default_factory=list)


def canonical_query(query: str) -> str:
    """The asker's own pronouns in the third person memories are stored in."""
    out = query
    for lang, rules in (_SEMANTIC.get("self_pronouns") or {}).items():
        if lang.startswith("$"):
            continue
        for pattern, repl in rules:
            out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def _item_questions(item_dict: Dict) -> List[str]:
    qs = item_dict.get("questions") if isinstance(item_dict, dict) else None
    return [q.strip() for q in qs if isinstance(q, str) and q.strip()] if isinstance(qs, list) else []


class AnticipationCache:
    """In-memory TTL cache with item-level matching: BM25, and by meaning
    with a local embedding model when one is loaded (``matcher="semantic"``)."""

    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_entries: int = 100,
        bm25_threshold: float = _DEFAULT_BM25_THRESHOLD,
        novel_term_threshold: float = _DEFAULT_NOVEL_TERM_THRESHOLD,
        *,
        encoder: Any = None,
        matcher: Optional[str] = None,
        min_cosine: Optional[float] = None,
        serve_min_cosine: Optional[float] = None,
        asked_kind: Optional[bool] = None,
        item_questions: Optional[bool] = None,
        canonical: Optional[str] = None,
        bundle_serve_min_cosine: Optional[float] = None,
        sure_min_cosine: Optional[float] = None,
        embed_in_background: bool = True,
    ):
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._bm25_threshold = bm25_threshold
        self._novel_term_threshold = novel_term_threshold

        # Semantic matching. The encoder is the SDK's history-search model,
        # handed over once it has loaded (set_encoder); until then, and with
        # matcher "bm25", lookups are BM25 only, as before.
        self._encoder = encoder
        self._matcher = matcher if matcher in MATCHERS else _matcher_setting()
        self._min_cosine = float(_SEMANTIC["min_cosine"] if min_cosine is None else min_cosine)
        self._serve_min_cosine = float(
            _SEMANTIC["serve_min_cosine"] if serve_min_cosine is None else serve_min_cosine)
        self._asked_kind = bool(_SEMANTIC["asked_kind"] if asked_kind is None else asked_kind)
        self._item_questions = bool(_SEMANTIC["item_questions"] if item_questions is None else item_questions)
        self._canonical = str(_SEMANTIC["canonical_query"] if canonical is None else canonical)
        self._bundle_serve_min_cosine = float(
            _SEMANTIC["bundle_serve_min_cosine"] if bundle_serve_min_cosine is None else bundle_serve_min_cosine)
        self._sure_min_cosine = float(_SEMANTIC["sure_min_cosine"] if sure_min_cosine is None else sure_min_cosine)
        self._embed_in_background = embed_in_background
        self._embed_lock = threading.Lock()

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

    def set_encoder(self, encoder: Any) -> None:
        """Hand over the local embedding model once it has loaded. Items
        already cached are embedded off the lookup path."""
        self._encoder = encoder
        self._embed_items(list(self._items))

    @property
    def semantic(self) -> bool:
        return self._matcher == "semantic" and self._encoder is not None

    def _embed_items(self, items: List["_ItemRecord"]) -> None:
        """Give ``items`` their vectors: in a background thread, so neither a
        bundle arriving on the stream nor a lookup waits for the model. A
        lookup before they are ready matches those items by BM25 alone."""
        encoder = self._encoder
        if encoder is None or self._matcher != "semantic":
            return
        todo = [i for i in items if i.vec_model != getattr(encoder, "model", "")]
        if not todo:
            return

        def run() -> None:
            with self._embed_lock:
                texts: List[str] = []
                for i in todo:
                    texts.append(i.content)
                    if self._item_questions:
                        texts.extend(i.questions)
                try:
                    vecs = [list(v) for v in encoder.encode(texts)]
                except Exception:
                    logger.warning("anticipation_cache: embedding items failed", exc_info=True)
                    return
                model = getattr(encoder, "model", "")
                k = 0
                for item in todo:
                    n = len(item.questions) if self._item_questions else 0
                    item.vec, item.qvecs = vecs[k], vecs[k + 1:k + 1 + n]
                    item.vec_model = model
                    k += 1 + n

        if self._embed_in_background:
            threading.Thread(target=run, name="synap-anticipation-embed", daemon=True).start()
        else:
            run()

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
        # The rung. Stamped by the server on the bundle it pushed; absent (and
        # therefore None) from anything an older server sent, which the lookup
        # treats as "named no rung" rather than as "matches anything".
        scope_rung = bundle.get("_anticipation_scope_rung") or None
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
            scope_rung=scope_rung,
        )

        items_by_type = bundle.get("items_by_type", {})
        items_added = 0
        items_deduped = 0
        first_new = len(self._items)
        for item_type, items_list in items_by_type.items():
            if not isinstance(items_list, list):
                continue
            for item_dict in items_list:
                content = item_dict.get("content", "")
                if not content:
                    continue
                dedup_key = self._dedup_key(entity_id, content, scope_rung)
                if dedup_key in self._item_dedup:
                    items_deduped += 1
                    continue
                self._item_dedup.add(dedup_key)
                # Kept with no word tokens too (Hindi, say, which the BM25
                # tokenizer drops): BM25 cannot match it, the embedding can.
                # The JS SDK always kept these.
                tokens = tokenize(content)
                self._items.append(_ItemRecord(
                    content=content,
                    tokens=tokens,
                    item_dict=item_dict,
                    item_type=item_type,
                    bundle_id=bundle_id,
                    confidence=item_dict.get("confidence", 1.0),
                    questions=_item_questions(item_dict),
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
                dedup_key = self._dedup_key(entity_id, content, scope_rung)
                if dedup_key in self._item_dedup:
                    items_deduped += 1
                    continue
                self._item_dedup.add(dedup_key)
                # Kept with no word tokens too (Hindi, say, which the BM25
                # tokenizer drops): BM25 cannot match it, the embedding can.
                # The JS SDK always kept these.
                tokens = tokenize(content)
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
        self._embed_items(self._items[first_new:])

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
        scope_rung: Optional[str] = None,
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

        ``scope_rung`` is a SECOND, independent dimension and NOT another tier
        of that funnel. The funnel widens; the rung does not. A bundle is
        servable only when the rung it was prefetched at equals the rung this
        request names, because a rung is a boundary and a boundary that widens
        is not one. Build it with ``rung_key(scope_path)``; ``None`` means the
        caller named no rung, which is every caller today.
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
                    "scope_rung": scope_rung,
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
            return self._freshness_lookup(entity_id, conversation_id, scope_rung=scope_rung)

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
                    "scope_rung": scope_rung,
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
            scope_rung=scope_rung,
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
        scope_rung: Optional[str] = None,
        lexical_only: bool = False,
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
                "scope_rung": scope_rung,
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
        semantic = self.semantic and not lexical_only
        hook_payload["matcher"] = "semantic" if semantic else "bm25"
        if semantic:
            return self._semantic_lookup(
                search_query, query_tokens, hook_payload, entity_id, conversation_id,
                max_items, customer_id=customer_id, client_id=client_id, scope_rung=scope_rung)
        if not query_tokens:
            hook_payload["exit_reason"] = "no_query_tokens"
            self._fire_lookup_hook(hook_payload)
            return self._freshness_lookup(entity_id, conversation_id, scope_rung=scope_rung)

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
            scope_rung=scope_rung,
        )

        # Derived per query. Short queries cannot reach the nominal score, so
        # holding them to it would miss every time. Both constants live in the
        # shared behavior contract.
        effective_threshold = max(
            _EFFECTIVE_FLOOR,
            min(self._bm25_threshold, _QUERY_TOKEN_SCALE * len(query_tokens)),
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

        return self._serve(top_items, hook_payload, search_query, effective_threshold, coverage)

    def _serve(
        self,
        top_items: List[Tuple[float, "_ItemRecord"]],
        hook_payload: Dict[str, Any],
        search_query: List[str],
        threshold: float,
        coverage: Optional[float],
        score_key: str = "bm25_score",
    ) -> Dict:
        """A hit: renew the leases of the bundles used, report, and merge the
        picked items into one response."""
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
            "Cache HIT (%s): score=%.2f threshold=%.2f items=%d coverage=%s query=%s",
            score_key,
            best_score,
            threshold,
            len(top_items),
            "-" if coverage is None else f"{coverage:.2f}",
            search_query[:80] if search_query else None,
        )

        # Populate the picked items for the hook before returning.
        # Each picked item records its score (bm25_score, or cosine for a
        # semantic match), its origin bundle, and a content preview for the
        # dashboard's drill-down.
        for rank, (score, item) in enumerate(top_items):
            hook_payload["items_picked"].append({
                "item_index": self._items.index(item) if item in self._items else rank,
                "bundle_id": item.bundle_id,
                "item_type": item.item_type,
                "content_first_240": item.content[:240],
                score_key: round(float(score), 4),
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


    def _semantic_lookup(
        self,
        search_query: List[str],
        query_tokens: List[str],
        hook_payload: Dict[str, Any],
        entity_id: Optional[str],
        conversation_id: Optional[str],
        max_items: int,
        *,
        customer_id: Optional[str] = None,
        client_id: Optional[str] = None,
        scope_rung: Optional[str] = None,
    ) -> Optional[Dict]:
        """Match by meaning: a lookup is a hit when an item's cosine to a query
        reaches ``min_cosine``, and the hit also serves the items at
        ``serve_min_cosine`` or above. An item with no vector yet is matched by
        BM25 with its novel-term gate, as before.

        BM25 does not also get a say over embedded items: on the bench, letting
        either signal serve brought back the near misses (an item on the asked
        topic without the asked detail shares its words) and served wrongly as
        often as BM25 alone. The coverage gate is a word-overlap test, exactly
        what a paraphrase or another language fails, so it does not apply.
        """
        hook_payload["bm25_query_tokens"] = list(query_tokens)
        queries = [q.strip() for q in search_query if q and q.strip()][: int(_SEMANTIC["max_queries"])]
        encoder = self._encoder
        model = getattr(encoder, "model", "")
        # The query as asked, and/or with the asker's pronouns in the third
        # person the memories are written in (canonical_query).
        asked = list(queries)  # the detail check reads the queries as asked
        if self._canonical in ("canon", "both"):
            canon = [canonical_query(q) for q in queries]
            changed = [c for c, q in zip(canon, queries) if c != q]
            queries = (queries + changed) if self._canonical == "both" else canon
        try:
            qvecs = [list(v) for v in (getattr(encoder, "encode_query", None) or encoder.encode)(queries)]
        except Exception:
            logger.warning("anticipation_cache: query embedding failed; BM25 only", exc_info=True)
            return self._item_lookup(
                search_query, entity_id, conversation_id, max_items,
                customer_id=customer_id, client_id=client_id, scope_rung=scope_rung, lexical_only=True)

        if query_tokens and (self._bm25_dirty or self._bm25 is None) and self._items:
            self._bm25 = BM25([item.tokens for item in self._items])
            self._bm25_dirty = False
        bm25_scores = self._bm25.scores(query_tokens) if query_tokens and self._bm25 else [0.0] * len(self._items)
        effective_threshold = max(
            _EFFECTIVE_FLOOR,
            min(self._bm25_threshold, _QUERY_TOKEN_SCALE * max(len(query_tokens), 1)),
        )
        hook_payload["bm25_threshold"] = round(float(effective_threshold), 4)
        hook_payload["min_cosine"] = self._min_cosine
        # Items not embedded yet are matched as before, BM25 with its
        # novel-term gate: without the gate BM25 served wrongly to 27% of the
        # bench's queries, against 12% with it.
        unique_stems = set(query_tokens)
        novel_ratio = (len(unique_stems - self._corpus_vocab) / len(unique_stems)) if unique_stems else 1.0
        hook_payload["novel_term_ratio"] = round(float(novel_ratio), 4)
        word_route = bool(query_tokens) and not (
            len(self._corpus_vocab) >= _MIN_CORPUS_FOR_NOVEL_GATE
            and novel_ratio >= self._novel_term_threshold)

        valid_bundles = self._get_valid_bundle_ids(
            entity_id, conversation_id,
            customer_id=customer_id, client_id=client_id, scope_rung=scope_rung,
        )
        picked: List[Tuple[float, _ItemRecord]] = []
        # Items that do not decide a hit but are served with one: a query that
        # needs two items rarely has both at the hit bar.
        companions: List[Tuple[float, _ItemRecord]] = []
        rejected: List[Tuple[float, Dict[str, Any]]] = []
        for idx, item in enumerate(self._items):
            b = float(bm25_scores[idx]) if idx < len(bm25_scores) else 0.0
            if item.bundle_id not in valid_bundles:
                # Scope first: another user's items are never scored.
                rejected.append((b, {
                    "item_index": idx, "bundle_id": item.bundle_id, "item_type": item.item_type,
                    "content_first_120": item.content[:120], "bm25_score": round(b, 4),
                    "cosine": None, "reason": "scope_filter_excluded",
                }))
                continue
            if item.vec is not None and item.vec_model == model and qvecs:
                # The memory's own text, and the questions it answers.
                cos = max(_cosine(q, v) for q in qvecs for v in [item.vec, *item.qvecs])
                passes = cos >= self._min_cosine
                rank = cos
            else:
                cos = None
                passes = word_route and b >= effective_threshold
                rank = self._min_cosine  # not embedded yet: ranked at the threshold
            if not passes and cos is not None and cos >= min(self._serve_min_cosine, self._bundle_serve_min_cosine):
                companions.append((cos, item))
                continue
            if not passes:
                rejected.append((rank, {
                    "item_index": idx, "bundle_id": item.bundle_id, "item_type": item.item_type,
                    "content_first_120": item.content[:120], "bm25_score": round(b, 4),
                    "cosine": None if cos is None else round(cos, 4), "reason": "below_threshold",
                }))
                continue
            picked.append((rank, item))

        rejected.sort(key=lambda x: -x[0])
        hook_payload["items_rejected"] = [r[1] for r in rejected[:20]]
        if not picked:
            hook_payload["exit_reason"] = "no_items_above_threshold"
            self._fire_lookup_hook(hook_payload)
            return None
        picked.sort(key=lambda x: -x[0])
        # Served with a hit: items above the serve bar, and items from a bundle
        # that holds a hit above the lower same-bundle bar (the server fetched
        # them together, for one anticipated topic).
        hit_bundles = {item.bundle_id for _, item in picked}
        companions = [(c, it) for c, it in companions
                      if c >= self._serve_min_cosine
                      or (it.bundle_id in hit_bundles and c >= self._bundle_serve_min_cosine)]
        companions.sort(key=lambda x: -x[0])
        top_items = (picked + companions)[:max_items]

        # A question for a kind of value (a number to call, an amount, a date)
        # is not answered by items on its topic that hold no such value.
        if self._asked_kind:
            patterns = _asked_kind_patterns(asked)
            if patterns and not any(p.search(item.content) for p in patterns for _, item in top_items):
                hook_payload["exit_reason"] = "asked_kind_missing"
                self._fire_lookup_hook(hook_payload)
                return None

        served = self._serve(top_items, hook_payload, search_query, self._min_cosine, None, score_key="cosine")
        # Below the sure bar a hit is borderline: the caller may still ask the
        # server, with a short deadline, and serve this only if it is late.
        served["_match_band"] = "sure" if picked[0][0] >= self._sure_min_cosine else "borderline"
        return served

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
        scope_rung: Optional[str] = None,
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

        The rung is checked FIRST and it never widens. Everything else here
        widens deliberately: a user-scope request reaches customer-shared
        bundles, and a conversation-scope request reaches the user-scope ones.
        A rung is the opposite kind of thing. It is the statement of what a
        caller may read, so "close enough" is the one answer it cannot give,
        and it is checked before the tiers so no amount of widening can put a
        cross-rung bundle back in.
        """
        accepted_scope = self._build_accepted_scope(entity_id, customer_id, client_id)
        valid: Set[str] = set()
        for bid, entry in self._entries.items():
            if not _rung_serves(entry.scope_rung, scope_rung):
                continue
            # Always, including when the request names no user: skipping this
            # check for a missing user_id served every user's bundles to a
            # conversation fetch that left user_id out. With no user named,
            # only customer-, client- and "_any"-stamped bundles remain.
            if entry.entity_id not in accepted_scope:
                continue
            if _unowned_elsewhere(entry.entity_id, entry.conversation_id, conversation_id):
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
        rung_breakdown: Dict[str, int] = {}
        for entry in self._entries.values():
            scope_breakdown[entry.entity_id] = scope_breakdown.get(entry.entity_id, 0) + 1
            rung = entry.scope_rung or "_none"
            rung_breakdown[rung] = rung_breakdown.get(rung, 0) + 1
        return {
            "total_entries": len(self._entries),
            "total_items": len(self._items),
            "corpus_vocab_size": len(self._corpus_vocab),
            "scope_breakdown": scope_breakdown,
            # Without this, a lookup refused for standing at the wrong rung and
            # a lookup that found an empty cache produce the same telemetry,
            # and "nothing leaked" and "nothing came back" are the two things
            # this feature must never confuse.
            "rung_breakdown": rung_breakdown,
        }

    def _freshness_lookup(
        self,
        entity_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        *,
        scope_rung: Optional[str] = None,
    ) -> Optional[Dict]:
        # This path serves a WHOLE bundle on no query at all, so it is the
        # least filtered way out of this cache and needs the rung as much as
        # the scored path does. With no user named it serves only "_any"
        # summaries: it used to serve the freshest summary of ANY user, so a
        # conversation fetch with no user_id and no query got a stranger's.
        owners = (entity_id, "_any") if entity_id else ("_any",)
        summary_candidates = {
            bid: e for bid, e in self._entries.items()
            if e.bundle_type == "user_summary"
            and _rung_serves(e.scope_rung, scope_rung)
            and e.entity_id in owners
            and not _unowned_elsewhere(e.entity_id, e.conversation_id, conversation_id)
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
        *,
        scope_rung: Optional[str] = None,
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
            and _rung_serves(entry.scope_rung, scope_rung)
            and entry.entity_id in (entity_id, "_any")
            and not _unowned_elsewhere(entry.entity_id, entry.conversation_id, None)
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

    def _dedup_key(
        self, entity_id: str, content: str, scope_rung: Optional[str] = None
    ) -> str:
        """Scope-qualified item-dedup key.

        Keying on content alone made the dedup set cache-GLOBAL: with the
        one-SDK-per-instance sharing model, visitor B's byte-identical seed
        facts were swallowed against visitor A's copy, never indexed under
        B's bundle — so B was permanently scope_filter_excluded for that
        content and always cold-fetched (2026-07-16 eval register issue #9).
        Scoping the key keeps within-visitor dedup intact.

        The rung is part of that scope for exactly the same reason one level
        down. Two rungs sharing an external id are two people, and their seed
        facts read identically; without the rung here the second rung's items
        are swallowed against the first's, never indexed under the second
        rung's bundle, and that rung then misses forever while the first is
        the only one that can be served. `None` keeps the key byte-identical
        for every bundle that names no rung."""
        rung_part = f"{scope_rung}|" if scope_rung else ""
        return f"{rung_part}{entity_id}|{content.lower().strip()[:120]}"

    def _rebuild_vocab(self) -> None:
        self._corpus_vocab = set()
        self._item_dedup = set()
        for item in self._items:
            self._corpus_vocab.update(item.tokens)
            entry = self._entries.get(item.bundle_id)
            entity_id = entry.entity_id if entry else "_any"
            # With the rung, as store() writes it: without it every rebuild
            # turned a rung's keys into the no-rung keys, and a later no-rung
            # bundle's identical items were swallowed against them.
            self._item_dedup.add(self._dedup_key(
                entity_id, item.content, entry.scope_rung if entry else None))

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
