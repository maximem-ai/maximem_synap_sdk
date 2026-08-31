/**
 * Serving a fetch from the anticipation cache.
 *
 * Ports Python's `_build_anticipation_response` and the cache-first branch each
 * scope fetch runs before going to the network.
 *
 * This is the point of the whole anticipation feature. The gRPC stream pushes
 * bundles ahead of time; without a lookup on the read path the cache is
 * write-only, every fetch is a billed round trip, and the stream does nothing
 * but consume bandwidth. That was the state of this SDK before: `store()` was
 * wired, `lookup()` was never called.
 */

import type { Json, RawContext } from './types.js';
import type { AnticipationCache, LookupResult } from './anticipation-cache.js';

export type Scope = 'conversation' | 'user' | 'customer' | 'client';

export interface AnticipationLookupParams {
  // `| undefined` explicitly, because exactOptionalPropertyTypes distinguishes
  // an absent key from one set to undefined, and callers build these by
  // spreading optionals.
  searchQuery?: readonly string[] | null | undefined;
  /** The user. Absent for customer- and client-scope fetches. */
  entityId?: string | null | undefined;
  customerId?: string | null | undefined;
  clientId?: string | null | undefined;
  conversationId?: string | null | undefined;
  /**
   * The rung the request names, from its `scope_path`. Passed through every
   * scope unchanged: unlike the ids below it, a rung never widens, so there is
   * no scope for which dropping it would be the safe thing to do.
   */
  scopeRung?: string | null | undefined;
}

/**
 * Which identifiers a scope may widen its lookup across.
 *
 * A customer-scope request widens to client-shared bundles but explicitly NOT
 * to user-scoped ones: those would leak one visitor's data into another
 * visitor's customer fetch. Client scope narrows furthest for the same reason.
 * This mirrors the scoping notes on Python's four fetch methods.
 */
export function scopeLookupParams(
  scope: Scope,
  params: AnticipationLookupParams,
): AnticipationLookupParams {
  const { searchQuery, entityId, customerId, clientId, conversationId, scopeRung } = params;
  switch (scope) {
    case 'conversation':
      return { searchQuery, entityId, customerId, clientId, conversationId, scopeRung };
    case 'user':
      // No conversationId: a user-scope fetch is not bound to one thread.
      return { searchQuery, entityId, customerId, clientId, scopeRung };
    case 'customer':
      return { searchQuery, entityId: null, customerId, clientId, scopeRung };
    case 'client':
      return { searchQuery, entityId: null, customerId: null, clientId, scopeRung };
  }
}

/** Turn a cache hit into the same raw shape the network path returns. */
export function buildAnticipationResponse(hit: LookupResult): RawContext {
  const response: RawContext = {};
  for (const [type, items] of Object.entries(hit.itemsByType)) {
    (response as Json)[type] = items;
  }
  // `source: "anticipation"` is how a caller (and the metering that reads it)
  // can tell a locally served fetch from a billed one.
  response.metadata = {
    source: 'anticipation',
    ttl_seconds: 0,
    cache_hit: true,
    retrieved_at: new Date().toISOString(),
  };
  return response;
}

/** Every item id in a hit, for the `context_used` event's `served_item_ids`. */
export function servedItemIds(hit: LookupResult): string[] {
  const ids: string[] = [];
  for (const items of Object.values(hit.itemsByType)) {
    for (const item of items) {
      const id = item.item_id ?? (item as Json)['id'];
      if (typeof id === 'string' && id !== '') ids.push(id);
    }
  }
  return ids;
}

export interface AnticipationAttempt {
  /** The response to return, or null to fall through to the network. */
  response: RawContext | null;
  hit: LookupResult | null;
}

/**
 * Try to serve this fetch locally.
 *
 * Deliberately swallows its own errors: a fault in the cache must degrade to a
 * network fetch, never fail a retrieval that the server could have answered.
 */
export function tryServeFromCache(
  cache: AnticipationCache,
  scope: Scope,
  params: AnticipationLookupParams,
  maxItems: number,
): AnticipationAttempt {
  try {
    const scoped = scopeLookupParams(scope, params);
    const hit = cache.lookup({
      searchQuery: scoped.searchQuery ?? null,
      entityId: scoped.entityId ?? null,
      customerId: scoped.customerId ?? null,
      clientId: scoped.clientId ?? null,
      conversationId: scoped.conversationId ?? null,
      scopeRung: scoped.scopeRung ?? null,
      maxItems,
    });
    if (hit === null) return { response: null, hit: null };
    return { response: buildAnticipationResponse(hit), hit };
  } catch {
    return { response: null, hit: null };
  }
}
