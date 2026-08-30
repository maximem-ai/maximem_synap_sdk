/**
 * Cache namespace. Mirrors `CacheInterface` in the Python SDK.
 *
 * This is a deliberately narrow facade over the AnticipationCache rather than
 * the cache object itself. Exposing the instance directly leaked internals
 * (`client.cache.now` was reachable) and gave JS a different surface from
 * Python for no benefit. The cache is still reachable as
 * `client.anticipationCache` for advanced use.
 */

import type { AnticipationCache } from '../context/anticipation-cache.js';
import { InvalidInputError } from '../errors.js';

export interface CacheStats {
  bundles: number;
  items: number;
  [key: string]: unknown;
}

export interface CacheNamespace {
  clear(): void;
  clear_user(userId: string): void;
  clear_customer(customerId: string): void;
  stats(): CacheStats;
}

export function createCacheNamespace(cache: AnticipationCache): CacheNamespace {
  return {
    clear: () => cache.clear(),

    clear_user: (userId) => {
      if (!userId) throw new InvalidInputError('user_id is required');
      cache.dropEntity(userId);
    },

    clear_customer: (customerId) => {
      if (!customerId) throw new InvalidInputError('customer_id is required');
      cache.dropEntity(customerId);
    },

    stats: () => ({ bundles: cache.size, items: cache.itemCount }),
  };
}
