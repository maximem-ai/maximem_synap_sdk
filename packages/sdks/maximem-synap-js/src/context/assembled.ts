/**
 * `context_assembled`: what the SDK actually composed for this turn.
 *
 * Ports Python's `_emit_context_assembled_event`. Fires after every fetch
 * resolves, whether it was served from the cache, the local short-term store or
 * the network, and records what the caller actually received. That can differ
 * from what the server returned, because of SDK-side trimming, the overlay, or
 * a cache hit that skipped the server entirely.
 *
 * It drives the Requests-page audit. The server backfills a synthetic row if no
 * event arrives within its window, so a missing event is not silently lost, it
 * just produces a less accurate one.
 *
 * Privacy: ids, counts and assembly metadata only. Never prompt or item content.
 */

import type { Json, RawContext } from './types.js';

const COLLECTIONS = ['facts', 'preferences', 'episodes', 'emotions', 'temporal_events'] as const;

export interface AssembledEvent extends Json {
  correlation_id: string;
  conversation_id: string;
  user_id: string;
  customer_id: string;
  final_item_ids: string[];
  final_total_tokens: number;
  compaction_id: string;
  recent_turn_count: number;
  compaction_end_timestamp: string;
  assembly_source: string;
  assembly_duration_ms: number;
  cache_hit: boolean;
  sdk_version: string;
}

/** Every item id the caller ended up with, across all five collections. */
export function finalItemIds(response: RawContext): string[] {
  const ids: string[] = [];
  for (const collection of COLLECTIONS) {
    const items = response[collection];
    if (!Array.isArray(items)) continue;
    for (const item of items) {
      const id = (item as Json)['id'] ?? (item as Json)['item_id'];
      if (typeof id === 'string' && id !== '') ids.push(id);
    }
  }
  return ids;
}

/**
 * Normalise the assembly source and derive `cache_hit` from it.
 *
 * `ResponseMetadata` uses `cache` as shorthand for the local HTTP cache; the
 * proto wants the explicit name. Three sources imply a hit regardless of what
 * the metadata's own flag says, because in each the server was not consulted.
 */
export function resolveSource(
  response: RawContext,
  override?: string,
): { source: string; cacheHit: boolean } {
  const metadata = (response.metadata ?? {}) as Json;
  let source = override ?? String(metadata['source'] ?? '') ?? '';
  if (source === '') source = 'cloud';
  if (source === 'cache') source = 'http_cache';
  const impliesHit = ['anticipation', 'anticipation_cache', 'http_cache', 'sdk_authoritative'];
  const cacheHit = impliesHit.includes(source) ? true : Boolean(metadata['cache_hit']);
  return { source, cacheHit };
}

export function buildAssembledEvent(params: {
  correlationId: string;
  response: RawContext;
  conversationId?: string | undefined;
  userId?: string | undefined;
  customerId?: string | undefined;
  startedAt: number;
  sdkVersion: string;
  sourceOverride?: string | undefined;
}): AssembledEvent {
  const { response } = params;
  const { source, cacheHit } = resolveSource(response, params.sourceOverride);
  const conversationContext = (response.conversation_context ?? {}) as Json;
  const recentTurns = conversationContext['recent_turns'];
  const metadata = (response.metadata ?? {}) as Json;

  return {
    correlation_id: params.correlationId,
    conversation_id: params.conversationId ?? '',
    user_id: params.userId ?? '',
    customer_id: params.customerId ?? '',
    final_item_ids: finalItemIds(response),
    final_total_tokens: Number(metadata['total_tokens'] ?? 0),
    compaction_id: String(conversationContext['compaction_id'] ?? ''),
    recent_turn_count: Array.isArray(recentTurns) ? recentTurns.length : 0,
    compaction_end_timestamp: String(
      conversationContext['end_timestamp'] ?? conversationContext['compacted_at'] ?? '',
    ),
    assembly_source: source,
    assembly_duration_ms: Math.max(0, Date.now() - params.startedAt),
    cache_hit: cacheHit,
    sdk_version: params.sdkVersion,
  };
}
