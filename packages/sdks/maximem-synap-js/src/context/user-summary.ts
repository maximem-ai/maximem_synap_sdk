/**
 * Periodic user-summary injection.
 *
 * Ports Python's `_should_inject_user_summary` / `_merge_user_summary_into_response`.
 *
 * Every Nth turn, a cached `user_summary` bundle is folded into the fetch
 * response. It is what keeps a long conversation grounded in who the user is
 * without paying for a broader retrieval on every turn. Without it, JS returns
 * less context than Python from identical usage, and the gap widens the longer
 * the conversation runs.
 */

import type { Json, RawContext, RawContextItem } from './types.js';

/** Python's `_user_summary_interval`. */
export const USER_SUMMARY_INTERVAL = 5;
/** Python's `max_summary_items`. */
const MAX_SUMMARY_ITEMS = 3;

const COLLECTIONS = ['facts', 'preferences', 'episodes', 'emotions', 'temporal_events'] as const;

/** Per-conversation turn counters, so the cadence is per conversation. */
export class TurnCounter {
  readonly #counts = new Map<string, number>();

  /** Increment and return the new count. Python buckets a missing id under `_global`. */
  increment(conversationId: string | undefined): number {
    const key = conversationId !== undefined && conversationId !== '' ? conversationId : '_global';
    const next = (this.#counts.get(key) ?? 0) + 1;
    this.#counts.set(key, next);
    return next;
  }

  /**
   * True on every Nth turn, and never on the first.
   *
   * Python: `count > 0 and count % interval == 0`. Turn 5 injects, turn 1 does
   * not, which matters because the first turn is usually the one that just
   * triggered a fresh retrieval anyway.
   */
  shouldInject(conversationId: string | undefined): boolean {
    const key = conversationId !== undefined && conversationId !== '' ? conversationId : '_global';
    const count = this.#counts.get(key) ?? 0;
    return count > 0 && count % USER_SUMMARY_INTERVAL === 0;
  }

  clear(): void {
    this.#counts.clear();
  }
}

function itemId(item: Json): string {
  return String(item['item_id'] ?? item['id'] ?? '');
}

function pick(item: Json, ...names: string[]): unknown {
  for (const name of names) {
    const value = item[name];
    if (value !== undefined && value !== null) return value;
  }
  return undefined;
}

/**
 * Normalise a summary-bundle item to the response item shape.
 *
 * The bundle carries the proto's field names, which differ from the REST
 * response's. Python spells the whole mapping out rather than passing items
 * through, because a consumer reading `.content` on an episode would otherwise
 * get undefined.
 */
function normalise(item: Json): RawContextItem {
  const content = pick(item, 'content') ?? '';
  const confidence = pick(item, 'confidence') ?? 0;
  const extractedAt = pick(item, 'extracted_at', 'created_at') ?? null;
  return {
    id: itemId(item),
    content: String(content),
    confidence: Number(confidence),
    source: String(pick(item, 'source') ?? ''),
    category: String(pick(item, 'context_type', 'category') ?? ''),
    emotion_type: String(pick(item, 'context_type', 'emotion_type') ?? ''),
    strength: Number(pick(item, 'strength', 'confidence') ?? 0),
    summary: String(pick(item, 'summary', 'content') ?? ''),
    significance: Number(pick(item, 'significance', 'confidence') ?? 0),
    intensity: Number(pick(item, 'intensity', 'confidence') ?? 0),
    context: String(pick(item, 'context', 'content') ?? ''),
    participants: (pick(item, 'participants') ?? []) as unknown[],
    extracted_at: extractedAt as string | null,
    occurred_at: (pick(item, 'occurred_at', 'extracted_at', 'created_at') ?? null) as string | null,
    detected_at: (pick(item, 'detected_at', 'extracted_at', 'created_at') ?? null) as string | null,
    metadata: (pick(item, 'metadata') ?? {}) as Json,
    event_date: (pick(item, 'event_date') ?? null) as string | null,
    valid_until: (pick(item, 'valid_until') ?? null) as string | null,
    temporal_category: (pick(item, 'temporal_category') ?? null) as string | null,
    temporal_confidence: Number(pick(item, 'temporal_confidence') ?? 0),
  };
}

/**
 * Fold a cached summary bundle into a response, in place.
 *
 * Appends at most `MAX_SUMMARY_ITEMS` per collection, skipping anything whose
 * id the response already carries: the summary supplements a retrieval, it does
 * not replace or duplicate it.
 */
export function mergeUserSummary(response: RawContext, bundle: Json): RawContext {
  const itemsByType = (bundle['items_by_type'] ?? {}) as Record<string, unknown>;
  for (const collection of COLLECTIONS) {
    const items = itemsByType[collection];
    if (!Array.isArray(items)) continue;

    const existing = (response[collection] ?? []) as RawContextItem[];
    const existingIds = new Set(existing.map((i) => String(i.id ?? i.item_id ?? '')));
    const fresh = (items as Json[])
      .filter((i) => typeof i === 'object' && i !== null && !existingIds.has(itemId(i)))
      .slice(0, MAX_SUMMARY_ITEMS)
      .map(normalise);
    if (fresh.length === 0) continue;

    response[collection] = [...existing, ...fresh];
  }
  return response;
}
