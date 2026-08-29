/**
 * Scope-aware context fetching.
 *
 * Scopes are client > customer > user. `conversation_id` is NOT a tier: it
 * groups turns within a scope. Passing it narrows an existing scope rather
 * than selecting a different one.
 */

import type { EndpointName } from '../transport/endpoints.js';
import type { HttpTransport } from '../transport/http.js';
import { InvalidInputError } from '../errors.js';
import type { FetchOptions, Json, RawContext } from './types.js';

export type Scope = 'user' | 'customer' | 'client' | 'conversation';

const ENDPOINT_BY_SCOPE: Record<Scope, EndpointName> = {
  user: 'context_fetch_user',
  customer: 'context_fetch_customer',
  client: 'context_fetch_client',
  conversation: 'context_fetch_conversation',
};

/** Reads either spelling of an option. Callers depend on both. */
function opt<T>(options: FetchOptions, snake: keyof FetchOptions, camel: keyof FetchOptions): T | undefined {
  const a = options[snake];
  if (a !== undefined) return a as T;
  const b = options[camel];
  return b === undefined ? undefined : (b as T);
}

export interface BuiltRequest {
  endpoint: EndpointName;
  body: Json;
}

export function buildFetchRequest(scope: Scope, options: FetchOptions): BuiltRequest {
  const userId = opt<string>(options, 'user_id', 'userId');
  const customerId = opt<string>(options, 'customer_id', 'customerId');
  const conversationId = opt<string>(options, 'conversation_id', 'conversationId');
  const searchQuery = opt<string[]>(options, 'search_query', 'searchQuery');
  const maxResults = opt<number>(options, 'max_results', 'maxResults') ?? 10;
  const precisionLevel = opt<string>(options, 'precision_level', 'precisionLevel') ?? 'high';
  const mode = options.mode ?? 'fast';

  if (searchQuery !== undefined && !Array.isArray(searchQuery)) {
    throw new InvalidInputError('search_query must be an array when provided');
  }
  if (options.types !== undefined && !Array.isArray(options.types)) {
    throw new InvalidInputError('types must be an array when provided');
  }

  // Field-for-field identical to what the Python controllers post, because the
  // server is tuned against that payload and Python is the reference
  // implementation. In particular:
  //   - conversation_id is ALWAYS present, null when unset, not omitted
  //   - mode always defaults to "fast"
  //   - precision_level is sent ONLY when it differs from "high"
  //   - include_conversation_context is sent ONLY when false
  // Sending an extra key, or omitting one Python always sends, is exactly the
  // kind of difference that shows up as "works in Python, empty in JS".
  const body: Json = {
    conversation_id: conversationId ?? null,
    search_query: searchQuery ?? [],
    max_results: maxResults,
    types: options.types ?? ['all'],
    mode,
  };
  if (precisionLevel !== 'high') body['precision_level'] = precisionLevel;
  if (options.include_conversation_context === false) {
    body['include_conversation_context'] = false;
  }

  switch (scope) {
    case 'user': {
      if (!userId) throw new InvalidInputError('user_id is required');
      body['user_id'] = userId;

      const scopePath = opt<Record<string, string>>(options, 'scope_path', 'scopePath');
      // Only sent when supplied, so a caller who does not use custom levels
      // sends exactly the body they sent before.
      if (scopePath !== undefined) body['scope'] = scopePath;

      const contextMode = opt<string>(options, 'context_mode', 'contextMode') ?? 'in-conversation';
      if (!VALID_CONTEXT_MODES.includes(contextMode)) {
        throw new InvalidInputError(
          `Invalid context_mode '${contextMode}'. Must be one of: ${VALID_CONTEXT_MODES.join(', ')}`,
        );
      }
      const lastN = opt<number>(options, 'last_n_conversations', 'lastNConversations') ?? 1;
      if (!Number.isInteger(lastN)) {
        throw new InvalidInputError('last_n_conversations must be an integer');
      }
      if (lastN < 0 || lastN > 20) {
        throw new InvalidInputError('last_n_conversations must be between 0 and 20');
      }

      // Conditional, exactly as Python does it: the summary-mode keys are sent
      // ONLY in summary mode, so an in-conversation fetch is byte-identical to
      // one that never knew about them.
      if (contextMode === 'conversation-summary') {
        body['context_mode'] = contextMode;
        body['include_profile'] = opt<boolean>(options, 'include_profile', 'includeProfile') ?? true;
        body['last_n_conversations'] = lastN;
      }
      // Python sends this only when not None. It is passed through the same
      // way rather than defaulted to '', so the server sees the same request
      // from either SDK.
      if (customerId !== undefined) body['customer_id'] = customerId;
      break;
    }
      break;
    case 'customer':
      if (!customerId) throw new InvalidInputError('customer_id is required');
      body['customer_id'] = customerId;
      break;
    case 'conversation':
      if (!conversationId) throw new InvalidInputError('conversation_id is required');
      if (userId !== undefined) body['user_id'] = userId;
      if (customerId !== undefined) body['customer_id'] = customerId;
      break;
    case 'client':
      break;
  }

  return { endpoint: ENDPOINT_BY_SCOPE[scope], body };
}

/**
 * Hooks that let the client serve a fetch locally and report the outcome.
 *
 * Passed in rather than imported so this module stays free of the cache and
 * the stream: it is the one place every scope's fetch goes through, and it
 * should not grow a dependency on either.
 */
export interface FetchHooks {
  /** Returns a response to short-circuit the network, or null to continue. */
  beforeFetch?: (
    scope: Scope,
    options: FetchOptions,
  ) => { response: RawContext; servedItemIds: string[]; bundleId: string } | null;
  /** Called after a cache hit is served, for the learning-loop event. */
  onServedFromCache?: (
    scope: Scope,
    options: FetchOptions,
    served: { servedItemIds: string[]; bundleId: string },
  ) => void;
  /** Called after EVERY fetch resolves, cache or network, for the audit event. */
  onAssembled?: (
    scope: Scope,
    options: FetchOptions,
    response: RawContext,
    startedAt: number,
  ) => void;
}

const VALID_CONTEXT_MODES = ['in-conversation', 'conversation-summary'];

/** Fetches and returns the RAW snake_case context, as the namespaced surface does. */
export async function fetchContext(
  transport: HttpTransport,
  scope: Scope,
  options: FetchOptions = {},
  hooks: FetchHooks = {},
): Promise<RawContext> {
  // Anticipation cache FIRST, as Python does on all four scope fetches. Without
  // this the stream's bundles are never read and every fetch is a billed round
  // trip.
  const startedAt = Date.now();

  // Before the anticipation cache, not after: a cached hit would otherwise let
  // an illegal shape through silently, which is the exact failure mode this
  // contract exists to remove.
  transport.checkCustomerId(
    opt<string>(options, 'customer_id', 'customerId'),
    `${scope}.context.fetch`,
  );

  const anticipated = hooks.beforeFetch?.(scope, options) ?? null;
  if (anticipated !== null) {
    hooks.onServedFromCache?.(scope, options, anticipated);
    hooks.onAssembled?.(scope, options, anticipated.response, startedAt);
    return anticipated.response;
  }

  const { endpoint, body } = buildFetchRequest(scope, options);
  const result = await transport.request<Json>(endpoint, { body });
  // The wrapper returned `result.context || {}`; preserve that exactly.
  const response = (result?.['context'] as RawContext) ?? {};
  hooks.onAssembled?.(scope, options, response, startedAt);
  return response;
}
