/**
 * Local search over a conversation's earlier turns.
 *
 * Port of `maximem_synap/cache/history_search.py`; see it for the design.
 * Cascade, cheapest first: literal values in the query, then BM25 (the
 * anticipation cache's tokenizer and parameters), then an embedding model
 * when one is supplied. Ranks are fused by reciprocal rank with a small recency prior, and
 * excerpts are shown in conversation order. The rendered block reaches the
 * LLM prompt and is byte-identical to Python's.
 */

import { HISTORY_SEARCH } from '../behavior/contract.js';
import { BM25, tokenize as latinTokenize } from './bm25.js';
import type { StoredTurn, TurnStore } from './turn-store.js';

/** A local embedding model. `model` names it, so vectors from another model are never compared. */
export interface Encoder {
  model: string;
  encode(texts: string[]): number[][] | Promise<number[][]>;
  /** Optional: a model trained with a query prefix (e5) encodes queries here and turns with `encode`. */
  encodeQuery?(texts: string[]): number[][] | Promise<number[][]>;
}

export interface Excerpt {
  position: number | null;
  role: string;
  text: string;
  score: number;
  timestamp: string;
  /** "literal", "bm25" or "embedding": the strongest signal that found it. */
  via: string;
}

// Any script's letters and digits; Latin words still go through the
// anticipation tokenizer for its stop words and stemming. Python: `tokenize`.
const WORD_RE = /[\p{L}\p{N}\p{M}]+/gu;

const FORMS: Readonly<Record<string, string>> = HISTORY_SEARCH.irregular_forms ?? {};

function tokenize(text: string): string[] {
  const out: string[] = [];
  for (const word of (text || '').toLowerCase().match(WORD_RE) ?? []) {
    // "sold" and "sell" are one word to a search; the stemmer only strips regular endings.
    if (/^[\x00-\x7f]+$/.test(word)) out.push(...latinTokenize(FORMS[word] ?? word));
    else out.push(word);
  }
  return out;
}

const LITERAL_SOURCE = String.raw`"([^"]{3,})"|\b([A-Z0-9][A-Z0-9-]{3,})\b|(\d[\d,.:/-]*\d|\d)`;
const SENTENCE_RE = /(?<=[.!?])\s+/;

function literalsOf(query: string): string[] {
  const out: string[] = [];
  for (const m of (query || '').matchAll(new RegExp(LITERAL_SOURCE, 'g'))) {
    const quoted = m[1];
    const value = quoted || m[2] || m[3];
    if (value && (quoted || /\d/.test(value))) out.push(value.toLowerCase());
  }
  return out;
}

function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

function words(text: string): string[] {
  return text.split(/\s+/).filter(Boolean);
}

function excerptOf(content: string, terms: Set<string>, literals: string[], maxWords: number): string {
  let sentences = content.trim().split(SENTENCE_RE).filter((s) => s.trim());
  if (sentences.length === 0) sentences = [content.trim()];
  const score = (sentence: string): number => {
    const low = sentence.toLowerCase();
    const held = new Set(tokenize(sentence));
    let n = 0;
    for (const t of terms) if (held.has(t)) n += 1;
    for (const lit of literals) if (low.includes(lit)) n += 2;
    return n;
  };
  let best = 0;
  for (let i = 1; i < sentences.length; i++) if (score(sentences[i]!) > score(sentences[best]!)) best = i;
  let lo = best;
  let hi = best;
  let count = words(sentences[best]!).length;
  for (;;) {
    let grew = false;
    for (const j of [hi + 1, lo - 1]) {
      if (j >= 0 && j < sentences.length && count + words(sentences[j]!).length <= maxWords) {
        count += words(sentences[j]!).length;
        lo = Math.min(lo, j);
        hi = Math.max(hi, j);
        grew = true;
      }
    }
    if (!grew) break;
  }
  let text = sentences.slice(lo, hi + 1).join(' ');
  if (words(text).length > maxWords) text = words(text).slice(0, maxWords).join(' ') + ' ...';
  return (lo > 0 ? '... ' : '') + text + (hi < sentences.length - 1 && !text.endsWith('...') ? ' ...' : '');
}

/**
 * The best-matching earlier turns for `query`. `exclude`: message ids or
 * exact texts already in the prompt (the recent turns).
 */
const FENCE = /^\s*(?:```|~~~)/;
const TABLE_RULE = /^\s*\|?\s*:?-{2,}/;
const SENTENCE = /(?<=[.!?])\s+/;

/**
 * A long turn as passages of up to about `maxWords` words: table rows carry the
 * table's header row, fenced blocks and other line-shaped text go a few lines
 * at a time, prose a few sentences at a time. Python: `split_passages`.
 */
export function splitPassages(content: string, maxWords: number): string[] {
  const text = (content ?? '').trim();
  if (words(text).length <= maxWords) return text ? [text] : [];
  const out: string[] = [];
  const lines = text.split('\n');
  let i = 0;
  while (i < lines.length) {
    const line = lines[i]!;
    if (FENCE.test(line)) {
      let j = i + 1;
      while (j < lines.length && !FENCE.test(lines[j]!)) j += 1;
      out.push(...windowsOf(lines.slice(i + 1, j), maxWords));
      i = j + 1;
      continue;
    }
    if (line.trim().startsWith('|')) {
      let j = i;
      while (j < lines.length && lines[j]!.trim().startsWith('|')) j += 1;
      const rows = lines.slice(i, j).filter((r) => !TABLE_RULE.test(r)).map((r) => r.trim());
      const [header, body] = rows.length > 1 ? [rows[0]!, rows.slice(1)] : ['', rows];
      for (const row of body) {
        const both = `${header} ${row}`.trim();
        out.push(words(both).length <= Math.trunc((maxWords * 3) / 2) ? both : row);
      }
      i = j;
      continue;
    }
    let j = i;
    while (j < lines.length && lines[j]!.trim() && !FENCE.test(lines[j]!) && !lines[j]!.trim().startsWith('|')) j += 1;
    if (j === i) {
      i += 1;
      continue;
    }
    const block = lines.slice(i, j);
    if (block.length > 1 && block.every((b) => words(b).length <= 25)) {
      out.push(...windowsOf(block, maxWords));
    } else {
      const sentences = block.map((b) => b.trim()).join(' ').split(SENTENCE).filter((x) => x.trim());
      out.push(...windowsOf(sentences, maxWords));
    }
    i = j;
  }
  const kept = out.filter((x) => x.trim());
  return kept.length ? kept : [text];
}

function windowsOf(pieces: readonly string[], maxWords: number): string[] {
  const out: string[] = [];
  let cur: string[] = [];
  let used = 0;
  for (const raw of pieces) {
    const piece = raw.trim();
    if (!piece) continue;
    const n = words(piece).length;
    if (cur.length && used + n > maxWords) {
      out.push(cur.join(' '));
      cur = [];
      used = 0;
    }
    if (n > maxWords) {
      const w = words(piece);
      for (let k = 0; k < w.length; k += maxWords) out.push(w.slice(k, k + maxWords).join(' '));
      continue;
    }
    cur.push(piece);
    used += n;
  }
  if (cur.length) out.push(cur.join(' '));
  return out;
}

/** What search scores: a whole turn, or one passage of a long turn. */
interface Unit {
  seq: number;
  content: string;
  turn: number;
  first: boolean;
  last: boolean;
  /** The passage's index, -1 for a whole turn (its vector is the turn's). */
  part: number;
}

/** The value patterns of the kinds of value `query` asks for. Python: `_asked_kinds`. */
function askedKinds(query: string): RegExp[] {
  const out: RegExp[] = [];
  for (const [name, spec] of Object.entries(HISTORY_SEARCH.asked_kinds ?? {})) {
    if (name.startsWith('$') || typeof spec !== 'object') continue;
    if (new RegExp(spec.query, 'i').test(query ?? '')) out.push(new RegExp(spec.value));
  }
  return out;
}

/** A question about how things started gets no recency prior. Python: `_asks_how_it_started`. */
function asksHowItStarted(query: string): boolean {
  const spec = HISTORY_SEARCH.early_query;
  if (!spec?.pattern) return false;
  return new RegExp(spec.pattern, (spec.flags ?? '').includes('i') ? 'i' : '').test(query ?? '');
}

export async function searchTurns(
  turns: readonly StoredTurn[],
  query: string,
  options: { exclude?: Iterable<string>; encoder?: Encoder | null; store?: TurnStore | null } = {},
): Promise<Excerpt[]> {
  const params = HISTORY_SEARCH;
  const k = params.max_excerpts;
  const skip = new Set(options.exclude ?? []);
  const pool = turns.filter((t) => t.content && !skip.has(t.content) && !skip.has(t.message_id ?? ''));
  const terms = new Set(tokenize(query || ''));
  const literals = literalsOf(query);
  if (pool.length === 0 || (terms.size === 0 && literals.length === 0 && !options.encoder)) return [];

  const maxWords = params.max_words_per_excerpt;
  const units: Unit[] = [];
  pool.forEach((t, ti) => {
    if (!params.passages) {
      units.push({ seq: t.seq, content: t.content, turn: ti, first: true, last: true, part: -1 });
      return;
    }
    const parts = splitPassages(t.content, maxWords);
    parts.forEach((part, pi) => {
      // A one-passage turn keeps the turn's vector; passages of a longer one are stored per passage.
      units.push({
        seq: t.seq, content: part, turn: ti, first: pi === 0, last: pi === parts.length - 1,
        part: parts.length > 1 ? pi : -1,
      });
    });
  });

  const docs = units.map((u) => tokenize(u.content));
  const bm = new BM25(docs);
  const termList = [...terms];
  const rows = units.map((turn, i) => {
    const held = new Set(docs[i]);
    let matched = 0;
    for (const t of terms) if (held.has(t)) matched += 1;
    const low = turn.content.toLowerCase();
    let literal = 0;
    for (const lit of literals) if (low.includes(lit)) literal += 1;
    return { i, bm25: matched ? bm.score(termList, i) : 0, matched, literal };
  });
  const need = Math.min(params.min_matched_terms, Math.max(1, terms.size));
  // One query word is enough when few turns use it, not when many do.
  const rareCap = Math.max(1, Math.trunc((params.rare_term_max_share ?? 0) * units.length));
  const rare = new Set<string>();
  for (const t of terms) {
    let n = 0;
    for (const doc of docs) if (doc.includes(t)) n += 1;
    if (n > 0 && n <= rareCap) rare.add(t);
  }
  let eligible;
  if (params.passages) {
    // A passage of a turn that matches enough query words as a whole.
    const turnTerms = new Map<number, Set<string>>();
    for (const r of rows) {
      const held = turnTerms.get(units[r.i]!.turn) ?? new Set<string>();
      for (const t of docs[r.i]!) if (terms.has(t)) held.add(t);
      turnTerms.set(units[r.i]!.turn, held);
    }
    eligible = rows.filter(
      (r) =>
        r.literal > 0 ||
        (r.matched > 0 &&
          (turnTerms.get(units[r.i]!.turn)!.size >= need || docs[r.i]!.some((t) => rare.has(t)))),
    );
  } else {
    eligible = rows.filter(
      (r) => r.literal > 0 || r.matched >= need || (r.matched > 0 && docs[r.i]!.some((t) => rare.has(t))),
    );
  }

  const rrf = params.rrf_k;
  const fused = new Map<number, number>();
  const via = new Map<number, string>();
  for (const r of eligible) fused.set(r.i, 0);
  const add = (i: number, rank: number, how: string): void => {
    fused.set(i, (fused.get(i) ?? 0) + 1 / (rrf + rank + 1));
    if (!via.has(i)) via.set(i, how);
  };
  eligible
    .filter((r) => r.literal > 0)
    .sort((a, b) => b.literal - a.literal || a.i - b.i)
    .forEach((r, rank) => add(r.i, rank, 'literal'));
  eligible
    .filter((r) => r.bm25 > 0)
    .sort((a, b) => b.bm25 - a.bm25 || a.i - b.i)
    .forEach((r, rank) => add(r.i, rank, 'bm25'));

  // A question asking for a kind of value also ranks the passages holding one.
  const kinds = askedKinds(query);
  if (kinds.length) {
    rows
      .filter((r) => r.matched > 0 && kinds.some((k) => k.test(units[r.i]!.content)))
      .sort((a, b) => b.bm25 - a.bm25 || a.i - b.i)
      .forEach((r, rank) => add(r.i, rank, 'kind'));
  }

  // Fused on every query when a model is supplied (Python: measured 19 Oct).
  if (options.encoder) {
    const ranked = await embeddingRanks(units, query, options.encoder, options.store ?? null, params.min_cosine ?? 0.4);
    ranked.forEach(([i], rank) => add(i, rank, 'embedding'));
  }
  if (fused.size === 0) return [];

  const n = Math.max(pool.length - 1, 1);
  const weight = asksHowItStarted(query) ? 0 : params.recency_weight;
  // A turn ranks by its best unit; its best units are what is shown of it.
  const top = new Map<number, [number, number]>();
  const ranked = new Map<number, [number, number][]>();
  for (const [u, v] of fused) {
    const ti = units[u]!.turn;
    const score = v + (weight * (ti / n)) / (rrf + 1);
    ranked.set(ti, [...(ranked.get(ti) ?? []), [score, u]]);
    const held = top.get(ti);
    if (!held || score > held[0]) top.set(ti, [score, u]);
  }
  const best = [...top.keys()].sort((a, b) => top.get(b)![0] - top.get(a)![0] || a - b).slice(0, k);
  const perTurn = params.passages_per_turn ?? 1;
  const shown = (ti: number): string => {
    const unit = units[top.get(ti)![1]]!;
    if (!params.passages || (unit.first && unit.last)) return excerptOf(pool[ti]!.content, terms, literals, maxWords);
    const picked = [...ranked.get(ti)!]
      .sort((a, b) => b[0] - a[0] || b[1] - a[1])
      .slice(0, perTurn)
      .map(([, u]) => u)
      .sort((a, b) => a - b);
    const parts: string[] = [];
    for (const u of picked) parts.push((units[u]!.first || parts.length ? '' : '... ') + units[u]!.content);
    return parts.join(' ... ') + (units[picked[picked.length - 1]!]!.last ? '' : ' ...');
  };
  const found = new Map<number, Excerpt>();
  for (const ti of best) {
    found.set(ti, {
      position: pool[ti]!.position,
      role: pool[ti]!.role,
      timestamp: pool[ti]!.timestamp,
      score: top.get(ti)![0],
      via: via.get(top.get(ti)![1]) ?? '',
      text: shown(ti),
    });
  }
  // The turn before a hit is often what it answers or confirms; it comes along, cut to its end.
  const before = params.previous_turns ?? 0;
  const tail = params.previous_turn_words ?? 40;
  for (const i of best) {
    for (let j = i - before; j < i; j++) {
      if (j < 0 || found.has(j)) continue;
      const hit = pool[i]!.position;
      const prev = pool[j]!.position;
      if (hit && prev && hit - prev > before) continue; // the turn just before is in the prompt
      const w = words(pool[j]!.content);
      found.set(j, {
        position: prev,
        role: pool[j]!.role,
        timestamp: pool[j]!.timestamp,
        score: 0,
        via: 'previous',
        text: w.length <= tail ? w.join(' ') : '... ' + w.slice(-tail).join(' '),
      });
    }
  }
  return [...found.keys()].sort((a, b) => a - b).map((i) => found.get(i)!);
}

async function embeddingRanks(
  pool: readonly { seq: number; content: string; part?: number }[],
  query: string,
  encoder: Encoder,
  store: TurnStore | null,
  minCosine: number,
): Promise<[number, number][]> {
  // Stored under the turn's seq and the model's name (a passage's: name + "#p<index>"). Python: `_embedding_ranks`.
  const keyOf = (u: { part?: number }): string => ((u.part ?? -1) < 0 ? encoder.model : `${encoder.model}#p${u.part}`);
  const vectors = new Map<number, ArrayLike<number>>();
  if (store !== null) {
    const byKey = new Map<string, number[]>();
    pool.forEach((u, i) => byKey.set(keyOf(u), [...(byKey.get(keyOf(u)) ?? []), i]));
    for (const [key, idx] of byKey) {
      const held = store.vectors(idx.map((i) => pool[i]!.seq), key);
      for (const i of idx) {
        const v = held.get(pool[i]!.seq);
        if (v) vectors.set(i, v);
      }
    }
  }
  const missing = pool.map((_, i) => i).filter((i) => !vectors.has(i));
  if (missing.length > 0) {
    const fresh = await encoder.encode(missing.map((i) => pool[i]!.content));
    const put = new Map<string, Map<number, Float32Array>>();
    missing.forEach((i, j) => {
      const v = Float32Array.from(fresh[j] ?? []);
      vectors.set(i, v);
      const key = keyOf(pool[i]!);
      put.set(key, (put.get(key) ?? new Map()).set(pool[i]!.seq, v));
    });
    if (store) for (const [key, items] of put) store.putVectors(key, items);
  }
  const q = (await (encoder.encodeQuery ?? encoder.encode).call(encoder, [query]))[0] ?? [];
  return pool
    .map((_, i): [number, number] => [i, cosine(q, vectors.get(i) ?? [])])
    .filter(([, c]) => c >= minCosine)
    .sort((a, b) => b[1] - a[1]);
}

/**
 * How an archived tool result reads in the turn store and in excerpts: compact
 * JSON, cut to the contract's size. Python: `tool_result_text`.
 */
export function toolResultText(result: unknown, toolName?: string | null): string {
  let text = typeof result === 'string' ? result : JSON.stringify(result) ?? String(result);
  const limit = HISTORY_SEARCH.tool_result_max_chars ?? 16_000;
  if (text.length > limit) text = text.slice(0, limit) + ' ...[cut]';
  return `[tool result: ${toolName || 'tool'}] ${text}`;
}

/** The block appended after the compaction and the recent turns. */
export function renderExcerpts(excerpts: readonly Excerpt[], latest = ''): string {
  if (excerpts.length === 0) return '';
  const lines = [HISTORY_SEARCH.heading];
  for (const e of excerpts) {
    let where = e.position ? `turn ${e.position}, ${e.role}` : `${e.role}, earlier`;
    if (e.timestamp) {
      const ago = before(e.timestamp, latest);
      where += `, ${e.timestamp}` + (ago ? ` (${ago})` : '');
    }
    lines.push(`- ${where}: "${e.text}"`);
  }
  return lines.join('\n');
}

const ZONED = /(?:Z|[+-]\d{2}:\d{2})$/;

/** Seconds since the epoch, for an ISO-8601 timestamp with a zone only. Python: `_instant`. */
function instant(stamp: string): number | null {
  const text = (stamp ?? '').trim();
  if (!ZONED.test(text)) return null;
  const ms = Date.parse(text);
  return Number.isNaN(ms) ? null : ms / 1000;
}

/** "3 days before the latest message", or "" when either is unreadable. Python: `_before`. */
function before(stamp: string, latest: string): string {
  const at = instant(stamp);
  const now = instant(latest);
  if (at === null || now === null || now < at) return '';
  const seconds = Math.trunc(now - at);
  for (const [unit, size] of [['day', 86_400], ['hour', 3_600], ['minute', 60]] as const) {
    if (seconds >= size) {
      const n = Math.trunc(seconds / size);
      return `${n} ${unit}${n !== 1 ? 's' : ''} before the latest message`;
    }
  }
  return 'less than a minute before the latest message';
}
