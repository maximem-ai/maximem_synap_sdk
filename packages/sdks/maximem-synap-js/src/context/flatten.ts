/**
 * Flatten typed context collections into one memory list.
 *
 * A faithful port of `flatten_context_items` from the 0.3.x bridge, which is
 * what `searchMemory` and `getMemories` returned. The per-type asymmetries are
 * deliberate and are reproduced exactly:
 *
 *  - `memory` comes from a different field per type: fact/preference/temporal
 *    event use `content`, episode uses `summary`, emotion uses `context`.
 *  - `score` likewise: fact -> confidence, preference -> strength,
 *    episode -> significance, emotion -> intensity, temporal event ->
 *    temporal_confidence.
 *  - **episode and emotion carry no `source` key at all.** The bridge simply
 *    did not set one. Adding it would be an improvement and a shape change, so
 *    it stays absent.
 *  - Order is facts, preferences, episodes, emotions, temporal events.
 */

import type { Json, RawContext, RawContextItem } from './types.js';

export interface FlatMemory {
  id: string;
  memory: string;
  score: number;
  /** Absent for episodes and emotions, matching the 0.3.x bridge. */
  source?: string;
  metadata: Json;
  context_type: 'fact' | 'preference' | 'episode' | 'emotion' | 'temporal_event';
  event_date: string | null;
  valid_until: string | null;
  temporal_category: string | null;
  temporal_confidence: number;
}

const s = (v: unknown): string => (typeof v === 'string' ? v : v == null ? '' : String(v));
const n = (v: unknown, d = 0): number => (typeof v === 'number' ? v : d);
const o = (v: unknown): Json => (v as Json) ?? {};
/** The bridge stringified datetimes and mapped falsy to null. */
const dt = (v: unknown): string | null => (v ? String(v) : null);

function temporal(item: RawContextItem) {
  return {
    event_date: dt(item['event_date']),
    valid_until: dt(item['valid_until']),
    temporal_category: (item['temporal_category'] as string | null) ?? null,
    temporal_confidence: n(item['temporal_confidence'], 0),
  };
}

export function flattenContextItems(context: RawContext): FlatMemory[] {
  const items: FlatMemory[] = [];

  for (const f of context.facts ?? []) {
    items.push({
      id: s(f['id']), memory: s(f['content']), score: n(f['confidence']),
      source: s(f['source']), metadata: o(f['metadata']),
      context_type: 'fact', ...temporal(f),
    });
  }
  for (const p of context.preferences ?? []) {
    items.push({
      id: s(p['id']), memory: s(p['content']), score: n(p['strength']),
      source: s(p['source']), metadata: o(p['metadata']),
      context_type: 'preference', ...temporal(p),
    });
  }
  for (const e of context.episodes ?? []) {
    // No `source`: the bridge did not set one for episodes.
    items.push({
      id: s(e['id']), memory: s(e['summary']), score: n(e['significance']),
      metadata: o(e['metadata']), context_type: 'episode', ...temporal(e),
    });
  }
  for (const m of context.emotions ?? []) {
    // No `source` here either, and `memory` comes from `context`, not `content`.
    items.push({
      id: s(m['id']), memory: s(m['context']), score: n(m['intensity']),
      metadata: o(m['metadata']), context_type: 'emotion', ...temporal(m),
    });
  }
  for (const t of context.temporal_events ?? []) {
    items.push({
      id: s(t['id']), memory: s(t['content']), score: n(t['temporal_confidence']),
      source: s(t['source']), metadata: o(t['metadata']),
      context_type: 'temporal_event', ...temporal(t),
    });
  }

  return items;
}
