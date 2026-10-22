/**
 * `response_shape: "ranked"`: one list in the order the server ranked it.
 *
 * Ports the ranked half of Python's `models/context.py` (`rank_items`,
 * `_wire_rank_score`, `_wire_category` and the ranked branch of
 * `ContextResponse.from_cloud_response`). The cases are pinned in
 * `CONTRACT/conformance/ranked_shape.json`, captured from Python.
 *
 * Opt-in. A by-type fetch, the default, is returned exactly as the server sent
 * it, so nothing here touches a caller who does not ask for the ranked shape.
 */

import type { Json, RawContext, RawContextItem } from './types.js';

/** The two shapes a context fetch can ask for. */
export const RESPONSE_SHAPES = ['by_type', 'ranked'] as const;
export type ResponseShape = (typeof RESPONSE_SHAPES)[number];

/** The memory types, in the order the by-type response lists them. */
export const MEMORY_TYPES = [
  'facts',
  'preferences',
  'episodes',
  'emotions',
  'temporal_events',
] as const;

/**
 * The type of a chunk in a ranked response: a quote from a source document,
 * not an extracted memory. A by-type response files it under `facts`.
 */
export const EXCERPT_CATEGORY = 'excerpts';

/**
 * An item's rank score: `metadata.rank_score` on the HTTP response, a
 * top-level `rank_score` on a prefetched (gRPC) item. `null` when neither is a
 * number, which is how a server that predates ranking reads.
 */
export function wireRankScore(item: RawContextItem): number | null {
  const md = item.metadata;
  let raw: unknown = md !== null && typeof md === 'object' ? (md as Json)['rank_score'] : undefined;
  if (raw === undefined || raw === null) raw = item['rank_score'];
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : null;
}

/**
 * An item's type. A chunk is an excerpt whatever list it came in: a server or
 * a prefetched bundle that predates ranking files it under facts.
 */
export function wireCategory(item: RawContextItem, fallback = ''): string {
  if (item.source === 'chunks') return EXCERPT_CATEGORY;
  const category = item['category'];
  if (typeof category === 'string' && category !== '') return category;
  const contextType = item['context_type'];
  if (typeof contextType === 'string' && contextType !== '') return contextType;
  return fallback;
}

/** Highest score first; items without one after, in the order they came. */
function byRankScore(a: RawContextItem, b: RawContextItem): number {
  const sa = wireRankScore(a);
  const sb = wireRankScore(b);
  if (sa === null && sb === null) return 0;
  if (sa === null) return 1;
  if (sb === null) return -1;
  return sb - sa;
}

/**
 * Merge per-type item lists into one list in rank order, numbered from 1.
 *
 * For a server that predates `response_shape` (it returned the five lists) and
 * for a prefetched cache hit. Each returned item is a copy carrying `rank`,
 * `category` and `id` (a prefetched item names it `item_id`), so the input is
 * not changed. `Array.prototype.sort` is stable, like Python's.
 */
export function rankItems(byType: RawContext): RawContextItem[] {
  const flat: RawContextItem[] = [];
  // The five lists only: a raw context also carries other arrays
  // (`conversations` in summary mode), and none of them are memories.
  for (const type of MEMORY_TYPES) {
    const list = (byType as Json)[type];
    if (!Array.isArray(list)) continue;
    for (const it of list as RawContextItem[]) {
      if (it === null || typeof it !== 'object') continue;
      flat.push({
        ...it,
        category: wireCategory(it, type),
        id: (it.id || it.item_id || '') as string,
      });
    }
  }
  flat.sort(byRankScore);
  return flat.map((it, n) => ({ ...it, rank: n + 1 }));
}

/**
 * The raw context in the shape the caller asked for.
 *
 * "ranked": `items` in rank order (the server's own list, or one rebuilt by
 * rank score when the server or the cache predates the parameter), the five
 * typed lists as views of those items, and `excerpts` for the chunks. A
 * response that already carries `items` is read the same way even when the
 * caller did not ask, as Python does. Anything else is returned untouched.
 */
export function applyResponseShape(raw: RawContext, shape: string = 'by_type'): RawContext {
  const wire = (raw as Json)['items'];
  if (shape !== 'ranked' && !Array.isArray(wire)) return raw;
  const source: RawContextItem[] = Array.isArray(wire)
    ? (wire as RawContextItem[])
    : rankItems(raw);
  const items = source.map((it) => ({
    ...it,
    id: (it.id || it.item_id || '') as string,
    category: wireCategory(it),
  }));
  const out: RawContext = { ...raw, items };
  for (const type of MEMORY_TYPES) {
    out[type] = items.filter((it) => it['category'] === type);
  }
  (out as Json)['excerpts'] = items.filter((it) => it['category'] === EXCERPT_CATEGORY);
  return out;
}

/** How each type is named on a line of the ranked prompt list. */
const RANKED_LABELS: Record<string, string> = {
  facts: 'fact',
  preferences: 'preference',
  episodes: 'episode',
  emotions: 'emotion',
  temporal_events: 'temporal event',
  [EXCERPT_CATEGORY]: 'excerpt',
};

/** The label printed in front of a ranked item: `fact`, `temporal event`, ... */
export function rankedLabel(category: string): string {
  return RANKED_LABELS[category] ?? (category !== '' ? category : 'item');
}

/** Shared by the ranked merge: the same ordering Python's `merge` sorts with. */
export function sortByRankScore(items: RawContextItem[]): RawContextItem[] {
  return [...items].sort(byRankScore);
}
