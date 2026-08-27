/** User namespace. Mirrors `UserInterface` in the Python SDK. */

import { InvalidInputError } from '../errors.js';
import type { HttpTransport } from '../transport/http.js';
import { fetchContext, type FetchHooks } from '../context/fetch.js';
import type { FetchOptions, Json, RawContext } from '../context/types.js';

export interface UserNamespace {
  get_profile(options: { user_id?: string; userId?: string; customer_id?: string; customerId?: string } | string): Promise<Json>;
  context: { fetch(options?: FetchOptions): Promise<RawContext> };
}

export function createUserNamespace(
  transport: HttpTransport,
  hooks: FetchHooks = {},
): UserNamespace {
  return {
    async get_profile(options) {
      const userId = typeof options === 'string' ? options : (options.user_id ?? options.userId);
      if (!userId) throw new InvalidInputError('user_id is required');
      const customerId =
        typeof options === 'string' ? undefined : (options.customer_id ?? options.customerId);
      return transport.request<Json>('user_profile', {
        pathParams: { user_id: userId },
        // Sent only when provided, matching Python. Note the server returns
        // 500 rather than a clean error if it needs one and does not get it.
        ...(customerId !== undefined ? { query: { customer_id: customerId } } : {}),
      });
    },
    context: { async fetch(options = {}) { return fetchContext(transport, 'user', options, hooks); } },
  };
}
