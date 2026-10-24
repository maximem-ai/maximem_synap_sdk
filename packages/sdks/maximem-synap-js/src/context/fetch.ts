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
import { RESPONSE_SHAPES, applyResponseShape } from './ranked.js';
import { hasQuery } from './no-query.js';
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
  const queryDate = opt<string>(options, 'query_date', 'queryDate');
  const perQuery = opt<number>(options, 'max_results_per_query', 'maxResultsPerQuery');
  const responseShape = responseShapeOf(options);
  const { mode, precisionLevel } = retrievalOptionsOf(options);
  const { contextMode, lastN } = scopeOptionsOf(scope, options);

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
  //   - query_date is sent ONLY when the caller supplied one
  //   - max_results_per_query is sent ONLY when the caller supplied one
  //   - response_shape is sent ONLY when "ranked"
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
  // Omitted at its default, for the same reason precision_level is.
  if (queryDate) body['query_date'] = queryDate;
  // Omitted when unset, and `0` counts as unset, as in Python.
  if (perQuery) body['max_results_per_query'] = perQuery;
  if (responseShape !== 'by_type') body['response_shape'] = responseShape;
  if (options.include_conversation_context === false) {
    body['include_conversation_context'] = false;
  }
  // On all four scopes, as Python sends it. Until 0.5.10 only the user scope
  // did, so a scope_path passed to the client, customer or conversation fetch
  // was dropped and the read was served at the default levels instead.
  // Omitted when absent or empty, like Python's `if scope_path:`, so a caller
  // who does not use custom levels sends exactly the body they sent before.
  const scopePath = opt<Record<string, string>>(options, 'scope_path', 'scopePath');
  if (scopePath !== undefined && scopePath !== null && Object.keys(scopePath).length > 0) {
    body['scope'] = scopePath;
  }

  switch (scope) {
    case 'user': {
      body['user_id'] = userId as string;

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
      body['customer_id'] = customerId as string;
      break;
    case 'conversation':
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
export interface AnticipatedServe {
  response: RawContext;
  servedItemIds: string[];
  bundleId: string;
  /**
   * A semantic match below the sure bar. The server is asked too, and its
   * answer is used if it arrives within `raceDeadlineMs`. Python:
   * `_race_borderline`.
   */
  borderline?: boolean;
  raceDeadlineMs?: number;
}

export interface FetchHooks {
  /** Returns a response to short-circuit the network, or null to continue. */
  beforeFetch?: (
    scope: Scope,
    options: FetchOptions,
  ) => AnticipatedServe | null | Promise<AnticipatedServe | null>;
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
  /**
   * Called last on every path, with the shaped response, which it may modify:
   * the periodic user-summary splice Python runs at the end of each scoped
   * fetch.
   */
  onResponse?: (scope: Scope, options: FetchOptions, response: RawContext) => void;
  /**
   * True when this process already holds the conversation's compaction and
   * every turn after it, so the request asks the server not to assemble them
   * (`include_conversation_context: false`), as Python does on every scope.
   */
  skipServerShortTerm?: (conversationId: string) => boolean;
  /**
   * The answer to a fetch with no query, or null to make the fetch as before:
   * the conversation's state for the conversation scope, the profile for the
   * user scope. Python answers these inside the two scoped fetches; here the
   * client holds what they need, so it answers through this hook.
   */
  withoutQuery?: (scope: Scope, options: FetchOptions) => Promise<RawContext | null>;
}

const VALID_CONTEXT_MODES = ['in-conversation', 'conversation-summary'];

/** The shape asked for, validated as Python validates it, before any request. */
export function responseShapeOf(options: FetchOptions): string {
  const shape = opt<string>(options, 'response_shape', 'responseShape') ?? 'by_type';
  if (!(RESPONSE_SHAPES as readonly string[]).includes(shape)) {
    throw new InvalidInputError(
      `Invalid response_shape '${shape}'. Must be one of: ${RESPONSE_SHAPES.join(', ')}`,
    );
  }
  return shape;
}

const REQUIRED_ID: Record<Scope, readonly [keyof FetchOptions, keyof FetchOptions] | null> = {
  user: ['user_id', 'userId'],
  customer: ['customer_id', 'customerId'],
  conversation: ['conversation_id', 'conversationId'],
  client: null,
};

/**
 * The scope's required id and, at user scope, the summary-mode options,
 * validated as Python validates them: before the anticipation cache, so a bad
 * value fails the same way whether or not a prefetched bundle would have
 * answered. Until 0.5.12 these ran only on the network path.
 */
export function scopeOptionsOf(
  scope: Scope,
  options: FetchOptions,
): { contextMode: string; lastN: number } {
  const required = REQUIRED_ID[scope];
  if (required !== null && !opt<string>(options, required[0], required[1])) {
    throw new InvalidInputError(`${required[0]} is required`);
  }
  const contextMode = opt<string>(options, 'context_mode', 'contextMode') ?? 'in-conversation';
  const lastN = opt<number>(options, 'last_n_conversations', 'lastNConversations') ?? 1;
  if (scope !== 'user') return { contextMode, lastN };
  if (!VALID_CONTEXT_MODES.includes(contextMode)) {
    throw new InvalidInputError(
      `Invalid context_mode '${contextMode}'. Must be one of: ${VALID_CONTEXT_MODES.join(', ')}`,
    );
  }
  if (!Number.isInteger(lastN)) {
    throw new InvalidInputError('last_n_conversations must be an integer');
  }
  if (lastN < 0 || lastN > 20) {
    throw new InvalidInputError('last_n_conversations must be between 0 and 20');
  }
  return { contextMode, lastN };
}

const VALID_MODES = ['fast', 'accurate'];
const VALID_PRECISION = ['high', 'medium'];

/**
 * `mode` and `precision_level`, validated as Python validates them, before any
 * request. Without this a typo reached the server, which reads anything that
 * is not "fast" as "accurate": a slower, costlier read and no error.
 */
export function retrievalOptionsOf(options: FetchOptions): { mode: string; precisionLevel: string } {
  const mode = options.mode ?? 'fast';
  if (!VALID_MODES.includes(mode)) {
    throw new InvalidInputError(
      `Invalid mode '${mode}'. Must be one of: ${VALID_MODES.join(', ')}`,
    );
  }
  const precisionLevel = opt<string>(options, 'precision_level', 'precisionLevel') ?? 'high';
  if (!VALID_PRECISION.includes(precisionLevel)) {
    throw new InvalidInputError(
      `Invalid precision_level '${precisionLevel}'. Must be one of: ${VALID_PRECISION.join(', ')}`,
    );
  }
  return { mode, precisionLevel };
}

/** Keep local no-query answers subject to the wire request's query contract. */
function validateSearchQuery(searchQuery: unknown): void {
  if (searchQuery === undefined || searchQuery === null) return;
  if (!Array.isArray(searchQuery) || searchQuery.some((query) => typeof query !== 'string')) {
    throw new InvalidInputError('search_query must be an array of strings when provided');
  }
}

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

  // Identity first, as Python requires initialize() before any fetch: the
  // cache lookup below filters on client_id and the B2C check on the
  // instance's isolation, and both come from whoami. Resolved once, on the
  // first request of a client that never called initialize().
  await transport.ready?.();

  // Before the anticipation cache, not after: a cached hit would otherwise let
  // an illegal shape through silently, which is the exact failure mode this
  // contract exists to remove.
  await transport.checkCustomerId(
    opt<string>(options, 'customer_id', 'customerId'),
    `${scope}.context.fetch`,
  );

  // Validated before the cache, like customer_id above, so a bad value fails
  // the same way whether or not a prefetched bundle would have answered.
  const responseShape = responseShapeOf(options);
  retrievalOptionsOf(options);
  const { contextMode } = scopeOptionsOf(scope, options);
  const searchQuery = opt<unknown>(options, 'search_query', 'searchQuery');
  validateSearchQuery(searchQuery);
  // Summary mode is a different read: a caller profile and past-conversation
  // summaries, assembled by the server. A prefetched bundle holds neither, and
  // with no search query the cache answered it with the newest bundle, so the
  // call-start read came back without a profile and no request was sent.
  const summaryMode = scope === 'user' && contextMode === 'conversation-summary';

  // No query: there is nothing to search for, and the search the server ran
  // instead embedded an empty string. Answered before the cache, as Python
  // answers it before its cache, and without the user-summary splice, which
  // Python's early return skips too.
  if (!summaryMode && hooks.withoutQuery && !hasQuery(searchQuery)) {
    const answer = await hooks.withoutQuery(scope, options);
    if (answer !== null) return applyResponseShape(answer, responseShape);
  }

  const finish = (response: RawContext): RawContext => {
    hooks.onResponse?.(scope, options, response);
    return response;
  };

  const conversationId = opt<string>(options, 'conversation_id', 'conversationId');
  const skipServerSt = conversationId ? hooks.skipServerShortTerm?.(conversationId) === true : false;

  const network = async (): Promise<RawContext> => {
    const { endpoint, body } = buildFetchRequest(
      scope,
      skipServerSt ? { ...options, include_conversation_context: false } : options,
    );
    const result = await transport.request<Json>(endpoint, { body });
    // The wrapper returned `result.context || {}`; preserve that exactly.
    return applyResponseShape((result?.['context'] as RawContext) ?? {}, responseShape);
  };

  // Awaited: matching by meaning embeds the query with the local model first.
  const anticipated = summaryMode ? null : (await hooks.beforeFetch?.(scope, options)) ?? null;
  if (anticipated !== null && anticipated.borderline) {
    // A borderline match races the server: its answer if in time, the cached
    // one if not. A late server call finishes unobserved.
    const pending = network();
    pending.catch(() => undefined);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const late = new Promise<null>((resolve) => {
      timer = setTimeout(() => resolve(null), anticipated.raceDeadlineMs ?? 250);
    });
    const won = await Promise.race([pending.then((r) => r, () => null), late]);
    if (timer !== undefined) clearTimeout(timer);
    if (won !== null) {
      hooks.onAssembled?.(scope, options, won, startedAt);
      return finish(won);
    }
  }
  if (anticipated !== null) {
    hooks.onServedFromCache?.(scope, options, anticipated);
    hooks.onAssembled?.(scope, options, anticipated.response, startedAt);
    // A hit is items picked from several prefetched bundles, so a ranked hit
    // is ordered by each item's own rank_score.
    return finish(applyResponseShape(anticipated.response, responseShape));
  }

  const response = await network();
  hooks.onAssembled?.(scope, options, response, startedAt);
  return finish(response);
}
