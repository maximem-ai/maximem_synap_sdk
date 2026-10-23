"""Local search over a conversation's earlier turns (COMPACTION_HYBRID_PLAN.md P5).

A compaction keeps where things stand; the exact wording, figures and
earlier discussion stay in the raw turns, which TurnStore keeps on this
machine. Before the agent's model call, the SDK searches them for the
latest user message and adds the best matches, verbatim, after the
compaction block. Nothing leaves the machine: the search is local, and so
is the embedding model when one is installed.

Cascade, cheapest first:

  1. literal values in the query (numbers, codes, quoted text) found as
     written in a turn
  2. BM25 over the turns, with the anticipation cache's tokenizer and
     parameters (behavior contract ``bm25``)
  3. an embedding model, when one is installed (encoders.py)

Ranks are fused by reciprocal rank, with a small prior for later turns: when
two turns match equally, the later one is more likely to hold the current
state. Excerpts are shown in conversation order, so the latest reads last.
Tuned values live in the behavior contract (``history_search``), shared with
the JS SDK.
"""
from __future__ import annotations

import math
import re
import unicodedata
from array import array
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Protocol, Sequence, Set

from ..behavior import history_search_params
from .bm25 import BM25, tokenize as _latin_tokenize

def _words(text: str) -> List[str]:
    r"""Runs of letters, digits and combining marks, in any script (JS:
    /[\p{L}\p{N}\p{M}]+/u). Python's \w drops the marks that Devanagari
    vowels are written with, splitting words apart."""
    out, word = [], []
    for ch in text:
        if unicodedata.category(ch)[0] in "LNM":
            word.append(ch)
        elif word:
            out.append("".join(word))
            word = []
    if word:
        out.append("".join(word))
    return out


def tokenize(text: str) -> List[str]:
    """The anticipation tokenizer keeps [a-z0-9] only, which made Hindi turns
    invisible to keyword search (12/50 found, measured 19 Oct). Latin words
    still go through it, for its stop words and stemming."""
    forms = history_search_params().get("irregular_forms") or {}
    out: List[str] = []
    for word in _words((text or "").lower()):
        if word.isascii():
            # "sold" and "sell" are one word to a search; the stemmer only
            # strips regular endings.
            out += _latin_tokenize(forms.get(word, word))
        else:
            out.append(word)
    return out
from .turn_store import StoredTurn, TurnStore


class Encoder(Protocol):
    """A local embedding model (P6). ``model`` names it, so stored vectors
    from another model are never compared."""

    model: str

    def encode(self, texts: List[str]) -> List[List[float]]: ...

    # Optional: a model trained with a query prefix (e5) encodes queries
    # with ``encode_query`` and turns with ``encode``.


@dataclass
class Excerpt:
    position: Optional[int]
    role: str
    text: str
    score: float
    timestamp: str = ""
    via: str = ""  # "literal", "bm25", "embedding": the strongest signal that found it


# Values typed as written: numbers with separators, codes, quoted text.
_LITERAL = re.compile(r'"([^"]{3,})"|\b([A-Z0-9][A-Z0-9-]{3,})\b|(\d[\d,.:/-]*\d|\d)')
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _literals(query: str) -> List[str]:
    out = []
    for quoted, code, number in _LITERAL.findall(query or ""):
        value = quoted or code or number
        if value and (quoted or any(c.isdigit() for c in value)):
            out.append(value.lower())
    return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _excerpt(content: str, terms: Set[str], literals: List[str], max_words: int) -> str:
    """The sentence that best matches, with its neighbours while they fit."""
    sentences = [s for s in _SENTENCE.split(content.strip()) if s.strip()] or [content.strip()]

    def score(sentence: str) -> float:
        low = sentence.lower()
        return len(terms & set(tokenize(sentence))) + 2 * sum(lit in low for lit in literals)

    best = max(range(len(sentences)), key=lambda i: (score(sentences[i]), -i))
    lo = hi = best
    words = len(sentences[best].split())
    while True:
        grew = False
        for j in (hi + 1, lo - 1):
            if 0 <= j < len(sentences) and words + len(sentences[j].split()) <= max_words:
                words += len(sentences[j].split())
                lo, hi = min(lo, j), max(hi, j)
                grew = True
        if not grew:
            break
    text = " ".join(sentences[lo:hi + 1])
    if len(text.split()) > max_words:
        text = " ".join(text.split()[:max_words]) + " ..."
    return ("... " if lo > 0 else "") + text + (" ..." if hi < len(sentences) - 1 and not text.endswith("...") else "")


_FENCE = re.compile(r"^\s*(?:```|~~~)")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}")


def split_passages(content: str, max_words: int) -> List[str]:
    """A long turn as passages of up to about ``max_words`` words, so a
    value deep in a pasted table, log or document can be found and shown
    (20 Oct: in 3 of 5 large-turn misses the right turn was found but its
    best sentence window cut the value out). A turn that fits is one
    passage. Table rows each carry the table's header row; fenced blocks
    and other line-shaped text go a few lines at a time; prose goes a few
    sentences at a time."""
    text = (content or "").strip()
    if len(text.split()) <= max_words:
        return [text] if text else []
    out: List[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line):
            j = i + 1
            while j < len(lines) and not _FENCE.match(lines[j]):
                j += 1
            out += _windows(lines[i + 1:j], max_words)
            i = j + 1
            continue
        if line.strip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            rows = [r.strip() for r in lines[i:j] if not _TABLE_RULE.match(r)]
            header, body = (rows[0], rows[1:]) if len(rows) > 1 else ("", rows)
            for row in body:
                both = f"{header} {row}".strip()
                out.append(both if len(both.split()) <= max_words * 3 // 2 else row)
            i = j
            continue
        j = i
        while j < len(lines) and lines[j].strip() and not _FENCE.match(lines[j]) \
                and not lines[j].strip().startswith("|"):
            j += 1
        if j == i:
            i += 1
            continue
        block = lines[i:j]
        if len(block) > 1 and all(len(b.split()) <= 25 for b in block):
            out += _windows(block, max_words)  # list items, log lines: kept as lines
        else:
            sentences = [x for x in _SENTENCE.split(" ".join(b.strip() for b in block)) if x.strip()]
            out += _windows(sentences, max_words)
        i = j
    return [p for p in out if p.strip()] or [text]


def _windows(pieces: Sequence[str], max_words: int) -> List[str]:
    """Consecutive pieces joined while they fit in ``max_words`` words; a
    piece longer than that is cut."""
    out, cur, used = [], [], 0
    for piece in (p.strip() for p in pieces):
        if not piece:
            continue
        n = len(piece.split())
        if cur and used + n > max_words:
            out.append(" ".join(cur))
            cur, used = [], 0
        if n > max_words:
            words = piece.split()
            out += [" ".join(words[k:k + max_words]) for k in range(0, len(words), max_words)]
            continue
        cur.append(piece)
        used += n
    if cur:
        out.append(" ".join(cur))
    return out


@dataclass
class _Unit:
    """What search scores: a whole turn, or one passage of a long turn."""
    seq: int
    content: str
    turn: int  # index into the pool
    first: bool
    last: bool
    part: int = -1  # the passage's index, -1 for a whole turn (its vector is the turn's)


def _asked_kinds(query: str, params: dict) -> List[re.Pattern]:
    """The value patterns of the kinds of value ``query`` asks for."""
    out = []
    for name, spec in (params.get("asked_kinds") or {}).items():
        if name.startswith("$") or not isinstance(spec, dict):
            continue
        if re.search(spec["query"], query or "", re.IGNORECASE):
            out.append(re.compile(spec["value"]))
    return out


def _asks_how_it_started(query: str, params: dict) -> bool:
    spec = params.get("early_query") or {}
    flags = re.IGNORECASE if "i" in str(spec.get("flags", "")) else 0
    return bool(spec.get("pattern")) and re.search(spec["pattern"], query or "", flags) is not None


def search(turns: Sequence[StoredTurn], query: str, *, exclude: Iterable[str] = (),
           encoder: Optional[Encoder] = None, store: Optional[TurnStore] = None) -> List[Excerpt]:
    """The best-matching earlier turns for ``query``. ``exclude``: message
    ids or exact texts already in the prompt (the recent turns)."""
    params = history_search_params()
    k = int(params["max_excerpts"])
    skip = set(exclude)
    pool = [t for t in turns if t.content and t.content not in skip and (t.message_id or "") not in skip]
    terms = set(tokenize(query or ""))
    literals = _literals(query)
    if not pool or (not terms and not literals and encoder is None):
        return []

    words = int(params["max_words_per_excerpt"])
    if params.get("passages"):
        units = []
        for ti, t in enumerate(pool):
            parts = split_passages(t.content, words)
            for pi, part in enumerate(parts):
                # A turn that is one passage keeps the turn's vector (the same
                # text); passages of a longer one are stored per passage.
                units.append(_Unit(t.seq, part, ti, pi == 0, pi == len(parts) - 1, pi if len(parts) > 1 else -1))
    else:
        units = [_Unit(t.seq, t.content, ti, True, True) for ti, t in enumerate(pool)]

    docs = [tokenize(u.content) for u in units]
    bm = BM25(docs)
    rows = []
    for i, (turn, doc) in enumerate(zip(units, docs)):
        held = set(doc)
        matched = len(terms & held)
        low = turn.content.lower()
        literal_hits = sum(lit in low for lit in literals)
        rows.append({"i": i, "bm25": bm.score(list(terms), i) if matched else 0.0, "matched": matched,
                     "literal": literal_hits})
    need = min(int(params["min_matched_terms"]), max(1, len(terms)))
    # One query word is enough when few turns use it ("levothyroxine"), not
    # when many do ("car").
    rare_cap = max(1, int(float(params.get("rare_term_max_share", 0.0)) * len(units)))
    df = {t: sum(1 for doc in docs if t in doc) for t in terms}
    rare = {t for t, n in df.items() if 0 < n <= rare_cap}
    if params.get("passages"):
        # A passage of a turn that matches enough query words as a whole: the
        # words of a question are often spread over a long turn.
        turn_terms: dict = {}
        for r in rows:
            turn_terms.setdefault(units[r["i"]].turn, set()).update(terms & set(docs[r["i"]]))
        eligible = [r for r in rows if r["literal"] or (r["matched"] and (
            len(turn_terms[units[r["i"]].turn]) >= need or rare & set(docs[r["i"]])))]
    else:
        eligible = [r for r in rows if r["literal"] or r["matched"] >= need
                    or (r["matched"] and rare & set(docs[r["i"]]))]

    rrf = float(params["rrf_k"])
    fused = {r["i"]: 0.0 for r in eligible}
    via = {}
    for rank, r in enumerate(sorted((r for r in eligible if r["literal"]), key=lambda r: -r["literal"])):
        fused[r["i"]] += 1.0 / (rrf + rank + 1)
        via.setdefault(r["i"], "literal")
    for rank, r in enumerate(sorted((r for r in eligible if r["bm25"] > 0), key=lambda r: -r["bm25"])):
        fused[r["i"]] += 1.0 / (rrf + rank + 1)
        via.setdefault(r["i"], "bm25")

    # A question asking for a kind of value also ranks the passages holding
    # one, among those that share a query word.
    kinds = _asked_kinds(query, params)
    if kinds:
        holders = [r for r in rows if r["matched"] and any(k.search(units[r["i"]].content) for k in kinds)]
        for rank, r in enumerate(sorted(holders, key=lambda r: -r["bm25"])):
            fused[r["i"]] = fused.get(r["i"], 0.0) + 1.0 / (rrf + rank + 1)
            via.setdefault(r["i"], "kind")

    # Fused on every query when a model is installed: measured 19 Oct, gating
    # it on keywords coming up short found fewer turns, and a query costs
    # milliseconds (turn vectors are kept in the store).
    if encoder is not None:
        for rank, (i, _) in enumerate(_embedding_ranks(units, query, encoder, store, float(params.get("min_cosine", 0.4)))):
            fused[i] = fused.get(i, 0.0) + 1.0 / (rrf + rank + 1)
            via.setdefault(i, "embedding")

    if not fused:
        return []
    # No recency prior for a question about how things started.
    weight = 0.0 if _asks_how_it_started(query, params) else float(params["recency_weight"])
    n = max(len(pool) - 1, 1)
    # A turn ranks by its best unit; its best units are what is shown of it.
    top: dict = {}
    ranked: dict = {}
    for u, score in fused.items():
        ti = units[u].turn
        score += weight * (ti / n) / (rrf + 1)
        ranked.setdefault(ti, []).append((score, u))
        if ti not in top or score > top[ti][0]:
            top[ti] = (score, u)
    best = sorted(top, key=lambda ti: -top[ti][0])[:k]
    per_turn = int(params.get("passages_per_turn", 1))

    def shown(ti: int) -> str:
        unit = units[top[ti][1]]
        if not params.get("passages") or (unit.first and unit.last):
            return _excerpt(pool[ti].content, terms, literals, words)
        picked = sorted(u for _, u in sorted(ranked[ti], reverse=True)[:per_turn])  # in the turn's order
        parts = []
        for u in picked:
            parts.append(("" if units[u].first or parts else "... ") + units[u].content)
        return " ... ".join(parts) + ("" if units[picked[-1]].last else " ...")

    found = {
        ti: Excerpt(position=pool[ti].position, role=pool[ti].role, timestamp=pool[ti].timestamp,
                    score=top[ti][0], via=via.get(top[ti][1], ""), text=shown(ti))
        for ti in best
    }
    # The turn before a hit is often what it answers or confirms ("Has it
    # arrived?" / "Yes, this morning"). It comes along, cut to its end.
    before = int(params.get("previous_turns", 0))
    tail = int(params.get("previous_turn_words", 40))
    for i in best:
        for j in range(i - before, i):
            if j < 0 or j in found:
                continue
            if pool[i].position and pool[j].position and pool[i].position - pool[j].position > before:
                continue  # the turn just before is not in the pool (it is in the prompt)
            text = pool[j].content.split()
            found[j] = Excerpt(position=pool[j].position, role=pool[j].role, timestamp=pool[j].timestamp,
                               score=0.0, via="previous",
                               text=" ".join(text) if len(text) <= tail else "... " + " ".join(text[-tail:]))
    return [found[i] for i in sorted(found)]


def _embedding_ranks(pool: Sequence, query: str, encoder: Encoder, store: Optional[TurnStore],
                     min_cosine: float) -> List[tuple]:
    """(index, cosine) of turns or passages close to the query, best first.
    Vectors are kept in the store under the turn's seq and the model's name
    (a passage's: the name and "#p<index>"), so each is encoded once and goes
    when its turn goes."""
    def key(u) -> str:
        part = getattr(u, "part", -1)
        return encoder.model if part < 0 else f"{encoder.model}#p{part}"

    vectors = {}
    if store is not None:
        by_key: dict = {}
        for i, u in enumerate(pool):
            by_key.setdefault(key(u), []).append(i)
        for k, idx in by_key.items():
            held = store.vectors([pool[i].seq for i in idx], k)
            for i in idx:
                blob = held.get(pool[i].seq)
                if blob is not None:
                    vec = array("f")
                    vec.frombytes(blob)
                    vectors[i] = list(vec)
    missing = [i for i in range(len(pool)) if i not in vectors]
    if missing:
        fresh = encoder.encode([pool[i].content for i in missing])
        put: dict = {}
        for i, vec in zip(missing, fresh):
            vectors[i] = list(vec)
            put.setdefault(key(pool[i]), {})[pool[i].seq] = array("f", vectors[i]).tobytes()
        if store is not None:
            for k, items in put.items():
                store.put_vectors(k, items)
    q = getattr(encoder, "encode_query", encoder.encode)([query])[0]
    scored = [(i, _cosine(q, vectors[i])) for i in range(len(pool))]
    return sorted([s for s in scored if s[1] >= min_cosine], key=lambda s: -s[1])


def tool_result_text(result: Any, tool_name: Optional[str] = None) -> str:
    """How an archived tool result reads in the turn store and in excerpts
    (the same bytes in both SDKs: compact JSON, cut to the contract's size)."""
    import json

    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, separators=(",", ":"),
                                                             default=str)
    limit = int(history_search_params().get("tool_result_max_chars", 16_000))
    if len(text) > limit:
        text = text[:limit] + " ...[cut]"
    return f"[tool result: {tool_name or 'tool'}] {text}"


def render_excerpts(excerpts: Sequence[Excerpt], latest: str = "") -> str:
    """The block appended after the compaction and the recent turns. Each
    excerpt carries its turn's timestamp as stored (the host's own), and,
    when ``latest`` (the newest turn's timestamp) is given, how long before
    it the turn was said: the agent should not have to work that out."""
    if not excerpts:
        return ""
    lines = [history_search_params()["heading"]]
    for e in excerpts:
        where = f"turn {e.position}, {e.role}" if e.position else f"{e.role}, earlier"
        if e.timestamp:
            ago = _before(e.timestamp, latest)
            where += f", {e.timestamp}" + (f" ({ago})" if ago else "")
        lines.append(f'- {where}: "{e.text}"')
    return "\n".join(lines)


_ZONED = re.compile(r"(?:Z|[+-]\d{2}:\d{2})$")


def _instant(stamp: str) -> Optional[float]:
    """Seconds since the epoch, for an ISO-8601 timestamp with a zone only:
    without one the two SDKs would read it in different zones (JS: Date.parse)."""
    text = (stamp or "").strip()
    if not _ZONED.search(text):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _before(stamp: str, latest: str) -> str:
    """"3 days before the latest message", or "" when either is unreadable."""
    at, now = _instant(stamp), _instant(latest)
    if at is None or now is None or now < at:
        return ""
    seconds = int(now - at)
    for unit, size in (("day", 86_400), ("hour", 3_600), ("minute", 60)):
        if seconds >= size:
            n = seconds // size
            return f"{n} {unit}{'s' if n != 1 else ''} before the latest message"
    return "less than a minute before the latest message"