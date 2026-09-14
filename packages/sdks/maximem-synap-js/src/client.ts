/**
 * SynapClient: the public entry point.
 *
 * Exposes two surfaces, both live in production:
 *
 *  - **Namespaced, snake_case** (`client.user.context.fetch(...)`) is the
 *    canonical surface and mirrors the Python SDK method for method. It
 *    returns raw snake_case.
 *  - **Top-level camelCase** (`client.fetchUserContext(...)`) is deprecated but
 *    fully supported. It returns the normalised camelCase shape, and it exists
 *    because the 0.3.x wrapper had it.
 *
 * They return different shapes on purpose. See gotcha G-B.
 */

import { InvalidInputError } from './errors.js';
import { getEnv } from './util/env.js';
import { HttpTransport, type Credentials, type HttpTransportOptions } from './transport/http.js';
import { fetchContext } from './context/fetch.js';
import { normalizeContextResponse } from './context/normalize.js';
import {
  AnticipationCache, type AnticipationCacheSnapshot,
} from './context/anticipation-cache.js';
import { flattenContextItems, type FlatMemory } from './context/flatten.js';
import {
  createMemoriesInterface, type CreateMemoryOptions, type CreateMemoryResult,
  type MemoriesInterface,
} from './memories/interface.js';
import {
  createConversationNamespace, CompactionSubscribers, type ConversationNamespace,
} from './conversation/interface.js';
import { createUserNamespace, type UserNamespace } from './user/interface.js';
import { createCreditsNamespace, type CreditsNamespace } from './credits/interface.js';
import { createCacheNamespace, type CacheNamespace } from './cache/interface.js';
import { createInstanceNamespace, type InstanceNamespace } from './instance/interface.js';
import type { FetchOptions, Json, NormalisedContext, RawContext } from './context/types.js';
import { SDK_VERSION } from './version.js';
import * as registry from './registry.js';
import { ShortTermStore } from './cache/short-term-store.js';
import { overlayLocalRecentTurns } from './context/overlay.js';
import { renderForPrompt, renderCompacted } from './context/local-prompt.js';
import { TurnCounter, mergeUserSummary } from './context/user-summary.js';
import { tryServeFromCache, servedItemIds, type Scope as AnticipationScope } from './context/anticipated.js';
import { rungKey } from './context/anticipation-cache.js';
import type { FetchHooks } from './context/fetch.js';
import { buildAssembledEvent } from './context/assembled.js';
import { newCorrelationId } from './util/correlation.js';
import { validateConversationId, validateInstanceId } from './util/validators.js';
import {
  mergeScopeResults, formatForPrompt, type UnifiedContext,
} from './context/unified.js';
import { buildTool, type AsToolOptions, type ToolDefinition } from './tool/as-tool.js';
import { checkCustomerId } from './scoping.js';

/** Levels the SDK emits. Mirrors the subset of Python's logging it uses. */
export type SynapLogLevel = 'debug' | 'info' | 'warn' | 'error';

/**
 * Receives every diagnostic the SDK would otherwise write to the console.
 *
 * Python routes these through the stdlib `logging` module, which is why it
 * takes `log_level` and `logger`. There is no global logger in JavaScript, so
 * the SDK takes the sink directly instead of a level plus a framework.
 */
export type SynapLogger = (level: SynapLogLevel, message: string) => void;

export interface SynapClientOptions {
  /**
   * The four identity fields are `| undefined` rather than plain optional so a
   * consumer with `exactOptionalPropertyTypes` can pass `process.env.X`
   * directly, which is what every example does. Undefined already means "fall
   * back to the environment" at runtime; this makes the type say so.
   */
  apiKey?: string | undefined;
  clientId?: string | undefined;
  instanceId?: string | undefined;
  baseUrl?: string | undefined;
  /** Python spells this `api_base_url`; both are accepted so one config
   *  object works against either SDK. */
  api_base_url?: string | undefined;
  timeouts?: HttpTransportOptions['timeouts'];
  retryPolicy?: HttpTransportOptions['retryPolicy'];
  keepAlive?: HttpTransportOptions['keepAlive'];
  /**
   * Keep the connection warm with a periodic /health ping.
   *
   * Off by default. A clear win for a long-lived process, pure waste in
   * serverless where the container is frozen between requests anyway.
   */
  heartbeat?: boolean;
  fetchImpl?: typeof fetch;
  /**
   * Splice locally-recorded turns into the next conversation-scope fetch, so a
   * just-written turn is visible before the server compacts. Defaults to true,
   * matching Python's `st_verbatim_overlay`. `SYNAP_ST_VERBATIM_OVERLAY`
   * overrides this either way.
   */
  st_verbatim_overlay?: boolean | undefined;
  stVerbatimOverlay?: boolean | undefined;

  /**
   * Let the SDK answer `get_context_for_prompt` and `get_compacted` from its
   * own short-term store when the conversation is warm, skipping the cloud
   * round trip. Off by default. `SYNAP_SDK_ST_AUTHORITATIVE` also enables it.
   *
   * This is the COST switch. `st_verbatim_overlay` is the correctness one and
   * is independent of it.
   */
  sdk_st_authoritative?: boolean | undefined;
  sdkStAuthoritative?: boolean | undefined;

  /**
   * Where the SDK's diagnostics go. Defaults to `console.warn` prefixed with
   * `[synap]`. Pass a no-op to silence them, or your own logger to route them.
   */
  logger?: SynapLogger | undefined;

  /**
   * Bypass the process registry and build a genuinely separate client.
   *
   * Underscored because it mirrors Python's `_force_new` and is not part of the
   * everyday surface. Needed in tests, and for the rare case of deliberately
   * running two clients on one identity with separate caches.
   */
  _force_new?: boolean;
}

interface ScopedContext {
  context: { fetch: (options?: FetchOptions) => Promise<RawContext> };
}

/**
 * Options for `configure()`.
 *
 * The Python-only keys are declared so a config object can be shared between
 * the two SDKs without a type error. They are accepted and ignored: there is
 * no SQLite cache backend and no global logger to replace here.
 */
export interface ConfigureOptions {
  timeouts?: HttpTransportOptions['timeouts'];
  retryPolicy?: HttpTransportOptions['retryPolicy'] | null;
  retry_policy?: HttpTransportOptions['retryPolicy'] | null;
  /** Accepted and ignored. Python-only. */
  storage_path?: string;
  /** Accepted and ignored. Python-only. */
  cache_backend?: string | null;
  /** Accepted and ignored. Python-only. */
  session_timeout_minutes?: number;
  /** Accepted and ignored. Python-only. */
  log_level?: string;
  /** Replaces the diagnostic sink, same as the constructor option. */
  logger?: SynapLogger | undefined;
}

/** Options for the cross-scope `fetch()`. Snake_case, matching Python's kwargs. */
export interface UnifiedFetchOptions {
  conversation_id?: string;
  user_id?: string;
  customer_id?: string;
  search_query?: string[];
  max_results?: number;
  types?: string[];
  mode?: string;
  precision_level?: string;
  include_conversation_context?: boolean;
  scopes?: string[];
  include_scope_labels?: boolean;
  context_mode?: string;
  include_profile?: boolean;
  last_n_conversations?: number;
}

export class SynapClient {
  // Native-private (`#`), not TypeScript `private`. TS `private` is erased at
  // compile time, so the field stays enumerable at runtime and a JS consumer
  // can still reach it. These two were public and made `client.transport
  // .credentials.apiKey` and the raw cache's `now`/`fire`/`invalidateEntity`
  // reachable, which is exactly what `client.cache` exists to prevent.
  // Definite-assignment assertions (`!`) because the constructor can RETURN an
  // existing client from the registry before reaching any of these. TypeScript
  // cannot follow a constructor return, so it sees a path where the fields are
  // unassigned. On that path the object is discarded and the incumbent is
  // handed back instead, so nothing observes them.
  readonly #transport!: HttpTransport;
  readonly #anticipationCache!: AnticipationCache;

  readonly user!: UserNamespace;
  readonly customer!: ScopedContext;
  readonly client!: ScopedContext;
  readonly conversation!: ConversationNamespace;
  readonly memories!: MemoriesInterface;
  readonly credits!: CreditsNamespace;
  readonly cache!: CacheNamespace;
  readonly instance!: InstanceNamespace;

  #closed = false;
  #initialized = false;
  readonly #registryKeys!: string[];
  readonly #compactionSubscribers!: CompactionSubscribers;
  readonly #shortTerm!: ShortTermStore;
  readonly #stVerbatimOverlay!: boolean | undefined;
  #logger!: SynapLogger | undefined;
  readonly #stAuthoritative!: boolean | undefined;
  readonly #turns!: TurnCounter;
  readonly #fetchHooks!: FetchHooks;

  constructor(options: SynapClientOptions = {}) {
    const credentials = resolveCredentials(options);
    const resolvedBaseUrl = resolveBaseUrl(options);
    // Fail locally on a malformed id rather than sending it upstream.
    validateInstanceId(credentials.instanceId);

    // ── Registry: one client per identity, per process ──────────────────────
    // Python has shared state here since 0.2. Without it, a framework
    // integration that builds a client per request or per agent gets a cold
    // anticipation cache every time and pays for a retrieval that Python would
    // have served locally.
    //
    // Returning from a constructor is unusual in JS but legal, and it is the
    // closest equivalent of Python's `self.__dict__ = existing.__dict__`. The
    // two callers then hold the SAME object rather than two objects sharing a
    // state bag, which is stricter than Python.
    const registryKey = registry.buildRegistryKey(credentials.instanceId, credentials.apiKey);
    if (options._force_new !== true) {
      const existing = registry.get(registryKey) as SynapClient | undefined;
      if (existing !== undefined) {
        // Handing back a client built on a DIFFERENT credential than the caller
        // just supplied. Only reachable on the explicit instanceId path, where
        // the id is the identity and the key is not part of it. The caller has
        // no other way to find out: the returned object looks exactly like the
        // one they asked for.
        const incumbent = existing.#transport.currentCredentials().apiKey;
        if (credentials.apiKey && incumbent && credentials.apiKey !== incumbent) {
          existing.#warn(
            `Instance ${registryKey} is already running in this process on a different ` +
              'API key. The key passed here is ignored; this client will authenticate ' +
              'as the existing one. Rotating a key this way has no effect. Construct ' +
              'with apiKey and no instanceId to get a client on your own credential.',
          );
        }
        return existing;
      }
    }
    this.#registryKeys = options._force_new === true ? [] : [registryKey];

    this.#transport = new HttpTransport({
      credentials,
      // SYNAP_BASE_URL is honoured, as Python does (sdk.py) and as the 0.3.x
      // wrapper did. Ignoring it was a wrong-DESTINATION bug rather than a
      // missing convenience: anyone pointing at a self-hosted or staging
      // deployment through the environment would silently have started sending
      // their data to production on upgrade.
      ...(resolvedBaseUrl !== undefined ? { baseUrl: resolvedBaseUrl } : {}),
      ...(options.timeouts !== undefined ? { timeouts: options.timeouts } : {}),
      ...(options.retryPolicy !== undefined ? { retryPolicy: options.retryPolicy } : {}),
      ...(options.keepAlive !== undefined ? { keepAlive: options.keepAlive } : {}),
      ...(options.fetchImpl !== undefined ? { fetchImpl: options.fetchImpl } : {}),
      userAgent: `@maximem/synap-js-sdk/${SDK_VERSION}`,
    });
    if (options.heartbeat) this.#transport.startHeartbeat();

    this.#anticipationCache = new AnticipationCache();

    this.#fetchHooks = {
      beforeFetch: (scope, options) => {
        const o = options as Record<string, unknown>;
        const attempt = tryServeFromCache(
          this.#anticipationCache,
          scope as AnticipationScope,
          {
            searchQuery: (o['search_query'] ?? o['searchQuery']) as string[] | undefined,
            entityId: (o['user_id'] ?? o['userId']) as string | undefined,
            customerId: (o['customer_id'] ?? o['customerId']) as string | undefined,
            clientId: this.#transport.currentCredentials().clientId || null,
            conversationId: (o['conversation_id'] ?? o['conversationId']) as string | undefined,
            // The rung this read is addressed to, in the same words the server
            // stamped on the bundles it pushed. Without it the cache matched
            // on entity and conversation alone, so a bundle prefetched for one
            // rung answered a caller standing at another, entirely inside this
            // process and invisible to every server log.
            scopeRung: rungKey(
              (o['scope_path'] ?? o['scopePath']) as Record<string, string> | undefined,
            ),
          },
          Number(o['max_results'] ?? o['maxResults'] ?? 10),
        );
        if (attempt.response === null || attempt.hit === null) return null;
        return {
          response: attempt.response,
          servedItemIds: servedItemIds(attempt.hit),
          bundleId: attempt.hit.bundleIds[0] ?? '',
        };
      },
      onAssembled: (scope, options, response, startedAt) => {
        this.#emitContextAssembled(scope, options as Record<string, unknown>, response, startedAt);
      },
      onServedFromCache: (_scope, options, served) => {
        // Fire-and-forget over the Listen stream. Drives the server's learning
        // loop: per-prefetch outcome scoring, per-pattern hit rates. Carries
        // ids and scope only, never prompt content.
        this.#emitContextUsed(options as Record<string, unknown>, _scope, served);
      },
    };

    this.user = createUserNamespace(this.#transport, this.#fetchHooks);
    // `async` arrows, not bare ones. fetchContext is already async so nothing
    // can throw synchronously today, but these are the only two namespace
    // methods that were not async, and the next person to add a validation
    // line above the call would reintroduce a throw that bypasses .catch().
    this.customer = { context: { fetch: async (o = {}) => fetchContext(this.#transport, 'customer', o, this.#fetchHooks) } };
    this.client = { context: { fetch: async (o = {}) => fetchContext(this.#transport, 'client', o, this.#fetchHooks) } };
    this.#compactionSubscribers = new CompactionSubscribers();
    this.#shortTerm = new ShortTermStore();
    this.#stVerbatimOverlay = options.st_verbatim_overlay ?? options.stVerbatimOverlay;
    this.#logger = options.logger;
    this.#stAuthoritative = options.sdk_st_authoritative ?? options.sdkStAuthoritative;
    this.#turns = new TurnCounter();
    this.conversation = createConversationNamespace(
      this.#transport,
      this.#compactionSubscribers,
      // Mirrors Python's `_invalidate_anticipation_on_write`, including its
      // gate: `invalidateEntity` is a no-op unless
      // SYNAP_SDK_CACHE_INVALIDATE_ON_WRITE is set, so default behaviour is
      // unchanged. Over-invalidating after a failed write is safe; it only
      // costs the next lookup a cold fetch.
      ({ userId, customerId }) => {
        for (const id of new Set([userId, customerId])) {
          if (id) this.#anticipationCache.invalidateEntity(id);
        }
      },
      // Buffer the turn locally so the next fetch sees it, ahead of the server
      // having persisted and compacted it.
      ({ conversationId, role, content, timestamp }) => {
        this.#shortTerm.appendTurn(conversationId, role, content, timestamp);
      },
      (response, conversationId, userId, scopePath) => {
        overlayLocalRecentTurns(response, this.#shortTerm, conversationId, this.#stVerbatimOverlay);
        this.#injectUserSummary(response, conversationId, userId, rungKey(scopePath));
      },
      {
        prompt: (conversationId, style) => this.#localPrompt(conversationId, style),
        compacted: (conversationId, format) => this.#localCompacted(conversationId, format),
      },
    );
    this.memories = createMemoriesInterface(this.#transport);
    this.credits = createCreditsNamespace(this.#transport);
    this.cache = createCacheNamespace(this.#anticipationCache);
    this.instance = createInstanceNamespace(
      () => this.#transport.currentCredentials(),
      this.#anticipationCache,
      ({ conversationId, role, content }) => {
        this.#shortTerm.appendTurn(conversationId, role, content);
      },
    );
    // A compaction folds turns into the server's summary, so the local copies
    // of those turns must go or they keep being spliced back in.
    this.instance._internal.setCompactionApplier((bundle) => {
      this.#shortTerm.applyCompaction(bundle);
    });

    if (options._force_new !== true) {
      // Claim the slot. A separate get-then-set would race in a threaded
      // runtime; JS is single-threaded between these statements, so this is
      // one operation for clarity rather than for locking.
      const incumbent = registry.registerIfAbsent(registryKey, this) as SynapClient | undefined;
      if (incumbent !== undefined) return incumbent;
    }
    // Route compaction bundles from the one Listen stream into the typed
    // per-conversation subscriptions. Wired here rather than inside either
    // namespace so the two do not have to import each other.
    this.instance._internal.setCompactionDispatcher((conversationId, bundle) => {
      this.#compactionSubscribers.dispatch(conversationId, bundle as Json, (error) => {
        this.#warn(`compaction subscriber threw: ${String(error)}`);
      });
    });
  }

  // ── Deprecated camelCase surface, carried over from the 0.3.x wrapper ───────
  // These return the NORMALISED shape, unlike their namespaced equivalents.

  /** @deprecated Use `client.user.context.fetch()`. Note it returns raw snake_case. */
  async fetchUserContext(options: FetchOptions = {}): Promise<NormalisedContext> {
    return normalizeContextResponse((await fetchContext(this.#transport, 'user', options)) as Json);
  }

  /** @deprecated Use `client.customer.context.fetch()`. */
  async fetchCustomerContext(options: FetchOptions = {}): Promise<NormalisedContext> {
    return normalizeContextResponse((await fetchContext(this.#transport, 'customer', options)) as Json);
  }

  /** @deprecated Use `client.client.context.fetch()`. */
  async fetchClientContext(options: FetchOptions = {}): Promise<NormalisedContext> {
    return normalizeContextResponse((await fetchContext(this.#transport, 'client', options)) as Json);
  }

  /** @deprecated Use `client.conversation.context.get_context_for_prompt()`. */
  async getContextForPrompt(
    options: { conversationId?: string; conversation_id?: string; style?: string } = {},
  ): Promise<Json> {
    return this.conversation.context.get_context_for_prompt(options);
  }

  /**
   * @deprecated Use `client.user.context.fetch()` and read the typed collections.
   *
   * Flattens every context type into one list, exactly as the 0.3.x bridge did.
   */
  async searchMemory(options: {
    userId?: string; user_id?: string;
    customerId?: string; customer_id?: string;
    query: string;
    maxResults?: number; max_results?: number;
    mode?: string;
    conversationId?: string; conversation_id?: string;
    types?: string[];
  }): Promise<{ results: FlatMemory[]; count: number }> {
    const userId = options.user_id ?? options.userId;
    if (!userId) throw new InvalidInputError('user_id is required');
    if (options.query === undefined || options.query === null) {
      throw new InvalidInputError('query is required');
    }
    const context = await fetchContext(this.#transport, 'user', {
      user_id: userId,
      ...(pick(options, 'customer_id', 'customerId') !== undefined
        ? { customer_id: pick(options, 'customer_id', 'customerId') as string }
        : {}),
      ...(pick(options, 'conversation_id', 'conversationId') !== undefined
        ? { conversation_id: pick(options, 'conversation_id', 'conversationId') as string }
        : {}),
      search_query: [options.query],
      max_results: (pick(options, 'max_results', 'maxResults') as number) ?? 10,
      types: options.types ?? ['all'],
      ...(options.mode !== undefined ? { mode: options.mode } : {}),
    });
    const results = flattenContextItems(context);
    return { results, count: results.length };
  }

  /**
   * @deprecated Use `client.user.context.fetch()` with an empty `search_query`.
   *
   * Note the default of 100, not 10: it matches the 0.3.x bridge.
   */
  async getMemories(options: {
    userId?: string; user_id?: string;
    customerId?: string; customer_id?: string;
    mode?: string;
    conversationId?: string; conversation_id?: string;
    maxResults?: number; max_results?: number;
    types?: string[];
  }): Promise<{ results: FlatMemory[]; count: number }> {
    const userId = options.user_id ?? options.userId;
    if (!userId) throw new InvalidInputError('user_id is required');
    const context = await fetchContext(this.#transport, 'user', {
      user_id: userId,
      ...(pick(options, 'customer_id', 'customerId') !== undefined
        ? { customer_id: pick(options, 'customer_id', 'customerId') as string }
        : {}),
      ...(pick(options, 'conversation_id', 'conversationId') !== undefined
        ? { conversation_id: pick(options, 'conversation_id', 'conversationId') as string }
        : {}),
      search_query: [],
      max_results: (pick(options, 'max_results', 'maxResults') as number) ?? 100,
      types: options.types ?? ['all'],
      ...(options.mode !== undefined ? { mode: options.mode } : {}),
    });
    const results = flattenContextItems(context);
    return { results, count: results.length };
  }

  /** @deprecated Use `client.memories.create()`. */
  async addMemory(options: CreateMemoryOptions): Promise<CreateMemoryResult> {
    return this.memories.create(options);
  }

  /**
   * @deprecated Use `client.memories.delete(memoryId)`.
   *
   * The `{ userId }` form is gone: it relied on a process-local id list and so
   * deleted nothing in serverless while reporting success (gotcha G-F).
   */
  async deleteMemory(options: { memoryId?: string; memory_id?: string }): Promise<Json> {
    return this.memories.delete(options.memory_id ?? options.memoryId ?? '');
  }

  /**
   * Resolve `client_id` and `instance_id` from the API key.
   *
   * Mirrors Python's `initialize()`. Calling it is optional here (unlike
   * Python, which raises "SDK not initialized" from `_ensure_initialized`)
   * because every HTTP route authenticates on the API key alone. It matters
   * for the anticipation cache: a bundle shared at client scope is filtered on
   * `client_id`, and an empty one fails that filter for every bundle. That was
   * a silent cache-scope bug in Python before whoami bootstrapping existed.
   *
   * Best-effort and idempotent. A whoami failure leaves whatever the
   * constructor resolved and is not fatal.
   */
  async initialize(): Promise<void> {
    if (this.#initialized) return;
    this.#initialized = true;
    const credentials = this.#transport.currentCredentials();
    if (credentials.clientId !== '' && credentials.instanceId !== '') return;
    try {
      const whoami = (await this.#transport.request('whoami', {})) as Json;
      const resolvedClientId = String(whoami['client_id'] ?? '');
      const resolvedInstanceId = String(whoami['instance_id'] ?? '');
      // The instance's scoping mode. Absent against any server older than this
      // field, and absent stays undefined, which every check treats as "do not
      // enforce". The server rejects independently either way.
      const iso = whoami['user_context_isolation'];
      // Stored on the transport, not on the client: every interface factory
      // already receives the transport, and surface-parity.test.ts is right
      // that a public member Python does not have is a divergence.
      this.#transport.userContextIsolation = typeof iso === 'string' ? iso : undefined;
      this.#transport.updateCredentials({
        ...(resolvedClientId !== '' && credentials.clientId === ''
          ? { clientId: resolvedClientId }
          : {}),
        ...(resolvedInstanceId !== '' && credentials.instanceId === ''
          ? { instanceId: resolvedInstanceId }
          : {}),
      });

      // Now that the real instance id is known, point it at this client too.
      // It was keyed on the credential because the id did not exist yet, so a
      // later `new SynapClient({ instanceId })` for the same instance would
      // otherwise miss and build a second client: two anticipation caches and
      // two Listen streams for one instance. The credential slot is kept, so
      // constructing by API key keeps resolving here as well.
      if (resolvedInstanceId !== '' && this.#registryKeys.length > 0) {
        if (registry.aliasIfAbsent(resolvedInstanceId, this)) {
          this.#registryKeys.push(resolvedInstanceId);
        }
      }
    } catch {
      // Non-fatal, exactly as in Python: identity resolution is an
      // optimisation, and every route still authenticates on the API key.
    }
  }

  /**
   * @deprecated Use `initialize()`, which is the Python name. `init()` was a
   * JS-only spelling with no Python counterpart; it now forwards, so a 0.3.x
   * startup script keeps working.
   */
  async init(): Promise<void> {
    return this.initialize();
  }

  /** The resolved instance id. Empty until `initialize()` resolves it. */
  get instance_id(): string {
    return this.#transport.currentCredentials().instanceId;
  }

  /** The resolved client id. Empty until `initialize()` resolves it. */
  get client_id(): string {
    return this.#transport.currentCredentials().clientId;
  }

  /**
   * Update configuration before `initialize()`.
   *
   * Mirrors Python's `configure()`, including its refusal to reconfigure after
   * initialization: the transport and cache are already built by then and a
   * half-applied config is worse than a rejected one.
   *
   * `logger` is honoured here as well as on the constructor. The remaining
   * Python-only keys (`storage_path`, `cache_backend`,
   * `session_timeout_minutes`, `log_level`) are accepted and ignored rather
   * than rejected, because this SDK has no SQLite cache backend and no global
   * logging framework whose level there would be to set. Ignoring them keeps
   * a shared config object portable between the two SDKs.
   */
  configure(options: ConfigureOptions = {}): void {
    if (this.#initialized) {
      throw new InvalidInputError('Cannot reconfigure after initialization');
    }
    if (options.timeouts !== undefined) this.#transport.setTimeouts(options.timeouts);
    if (options.retryPolicy !== undefined || options.retry_policy !== undefined) {
      this.#transport.setRetryPolicy(options.retryPolicy ?? options.retry_policy ?? null);
    }
    if (options.logger !== undefined) this.#logger = options.logger;
  }

  /**
   * Fetch and merge context across every scope with an identifier, in one call.
   *
   * Mirrors Python's `sdk.fetch()`. The three behaviours worth knowing:
   *
   *  - Scope fetches run in parallel and a FAILED scope is dropped with a
   *    warning rather than failing the call, so partial context still gets
   *    returned. The one exception is `InvalidInputError`: a bad request must
   *    surface instead of degrading into a silently-empty context.
   *  - `user_id` is threaded into the conversation-scope sub-fetch so the
   *    anticipation cache can apply its per-user filter.
   *  - The summary-mode parameters go ONLY to the user-scope sub-fetch. The
   *    other three routes return 422 for `context_mode="conversation-summary"`,
   *    and that rejection would be swallowed by the drop-failed-scope rule.
   */
  async fetch(options: UnifiedFetchOptions = {}): Promise<UnifiedContext> {
    const {
      conversation_id: conversationId,
      user_id: userId,
      customer_id: customerId,
      search_query: searchQuery,
      max_results: maxResults = 20,
      types,
      mode = 'fast',
      precision_level: precisionLevel = 'high',
      include_conversation_context: includeConversationContext = true,
      scopes,
      include_scope_labels: includeScopeLabels = false,
      context_mode: contextMode = 'in-conversation',
      include_profile: includeProfile = true,
      last_n_conversations: lastNConversations = 1,
    } = options;

    validateConversationId(conversationId);

    const wants = (scope: string): boolean => scopes === undefined || scopes.includes(scope);
    const shared = {
      search_query: searchQuery,
      max_results: maxResults,
      types,
      mode,
      precision_level: precisionLevel,
    };

    const labels: string[] = [];
    const tasks: Array<Promise<RawContext>> = [];

    if (conversationId !== undefined && conversationId !== '' && wants('conversation')) {
      labels.push('conversation');
      tasks.push(this.conversation.context.fetch({
        conversation_id: conversationId, ...shared,
        user_id: userId, customer_id: customerId,
      } as FetchOptions));
    }
    if (userId !== undefined && userId !== '' && wants('user')) {
      labels.push('user');
      tasks.push(this.user.context.fetch({
        user_id: userId, conversation_id: conversationId, ...shared,
        customer_id: customerId,
        context_mode: contextMode,
        include_profile: includeProfile,
        last_n_conversations: lastNConversations,
      } as FetchOptions));
    }
    if (customerId !== undefined && customerId !== '' && wants('customer')) {
      labels.push('customer');
      tasks.push(this.customer.context.fetch({
        customer_id: customerId, conversation_id: conversationId, ...shared,
      } as FetchOptions));
    }
    // Client scope needs no identifier (it is inferred from auth), but Python
    // only queries it when EXPLICITLY named in `scopes`. Preserved: including
    // it by default would add a billed fetch to every unified call.
    if (scopes !== undefined && scopes.includes('client')) {
      labels.push('client');
      tasks.push(this.client.context.fetch({
        conversation_id: conversationId, ...shared,
      } as FetchOptions));
    }

    if (tasks.length === 0) {
      const empty = mergeScopeResults([]);
      return empty;
    }

    const settled = await Promise.allSettled(tasks);
    const successful: Array<readonly [string, RawContext]> = [];
    for (const [i, result] of settled.entries()) {
      const label = labels[i] as string;
      if (result.status === 'fulfilled') {
        successful.push([label, result.value]);
        continue;
      }
      if (result.reason instanceof InvalidInputError) throw result.reason;
      this.#warn(`Scope '${label}' fetch failed in unified fetch (non-fatal): ${String(result.reason)}`);
    }

    const merged = mergeScopeResults(successful);

    // Read-your-writes: splice the locally buffered tail over whatever the
    // server returned for this conversation.
    for (const [, response] of successful) {
      overlayLocalRecentTurns(response, this.#shortTerm, conversationId, this.#stVerbatimOverlay);
    }

    if (includeConversationContext && conversationId !== undefined && conversationId !== '') {
      try {
        merged.conversation_context = (await this.conversation.context.get_context_for_prompt({
          conversation_id: conversationId,
        })) as Json;
      } catch {
        // Non-fatal in Python too: the merged items are still worth returning.
      }
    }

    merged.formatted_context = formatForPrompt(merged, {
      includeScope: includeScopeLabels,
      includeConversationContext,
    });
    return merged;
  }

  /**
   * An LLM-ready tool definition that fetches Synap context.
   *
   * Mirrors Python's `as_tool()`. The scope identifiers are closed over, so
   * the model cannot choose whose memory to read.
   */
  as_tool(options: AsToolOptions = {}): ToolDefinition {
    return buildTool(this as never, options, (message) => this.#warn(message));
  }

  /**
   * A read-only diagnostic view of the in-process anticipation cache.
   *
   * Mirrors Python's `anticipation_cache_snapshot()`. Returns the same
   * snake_case shape, since this output gets logged and compared across SDKs.
   */
  anticipation_cache_snapshot(): AnticipationCacheSnapshot {
    return this.#anticipationCache.snapshot();
  }

  /**
   * Report a locally served fetch back to the server.
   *
   * Only while listening: the event rides the Listen stream, and there is
   * nowhere to put it otherwise. Never throws, because a telemetry failure
   * must not fail a retrieval that already succeeded.
   */
  #emitContextUsed(
    options: Record<string, unknown>,
    scope: string,
    served: { servedItemIds: string[]; bundleId: string },
  ): void {
    try {
      if (!this.instance.is_listening) return;
      this.instance._internal.sendContextUsed({
        bundle_id: served.bundleId,
        conversation_id: String(options['conversation_id'] ?? options['conversationId'] ?? ''),
        user_id: String(options['user_id'] ?? options['userId'] ?? ''),
        customer_id: String(options['customer_id'] ?? options['customerId'] ?? ''),
        served_item_ids: served.servedItemIds,
        scope,
        source_bundle_ids: served.bundleId ? [served.bundleId] : [],
        timestamp_ms: Date.now(),
      });
    } catch {
      // Non-fatal by design.
    }
  }

  /**
   * Report what was actually composed for this turn.
   *
   * Fires whether the fetch was served from the cache or the network, because
   * the audit needs to distinguish them. Like the other stream events it is
   * fire-and-forget and only goes anywhere while listening.
   */
  #emitContextAssembled(
    scope: string,
    options: Record<string, unknown>,
    response: RawContext,
    startedAt: number,
  ): void {
    try {
      if (!this.instance.is_listening) return;
      this.instance._internal.sendContextAssembled({
        ...buildAssembledEvent({
          correlationId: newCorrelationId(),
          response,
          conversationId: (options['conversation_id'] ?? options['conversationId']) as string | undefined,
          userId: (options['user_id'] ?? options['userId']) as string | undefined,
          customerId: (options['customer_id'] ?? options['customerId']) as string | undefined,
          startedAt,
          sdkVersion: SDK_VERSION,
        }),
        scope,
      });
    } catch {
      // Non-fatal by design.
    }
  }

  /**
   * Fold a cached user summary in, every Nth turn.
   *
   * Only when the caller supplied a user_id. Without one the summary lookup
   * cannot be scoped, and an unscoped lookup is how user A's summary ends up in
   * user B's response.
   */
  #injectUserSummary(
    response: RawContext,
    conversationId: string | undefined,
    userId: string | undefined,
    scopeRung: string | null,
  ): void {
    this.#turns.increment(conversationId);
    if (userId === undefined || userId === '') return;
    if (!this.#turns.shouldInject(conversationId)) return;
    // The rung, for the same reason the user id is here: a summary prefetched
    // for one rung is not this caller's summary if they are standing at
    // another, and splicing it in is the same failure one level down.
    const summary = this.#anticipationCache.lookupUserSummary(userId, scopeRung);
    if (summary === null) return;
    mergeUserSummary(response, summary as Json);
  }

  /**
   * Whether the SDK is authoritative for short-term context.
   *
   * Resolution order matches Python's `_is_st_authoritative`: the explicit
   * option first, then `SYNAP_SDK_ST_AUTHORITATIVE` (truthy strings only).
   * Defaults to false, so nothing changes for a caller who never sets it.
   */
  #isStAuthoritative(): boolean {
    if (this.#stAuthoritative === true) return true;
    const raw = (getEnv('SYNAP_SDK_ST_AUTHORITATIVE') ?? '').trim().toLowerCase();
    return ['1', 'true', 'yes', 'on'].includes(raw);
  }

  /**
   * Render `get_context_for_prompt` from the local store, or null to fall
   * through to the network. Warm means a compaction landed OR raw turns are
   * buffered: with neither there is nothing to render and the server's answer
   * is the only real one.
   */
  #localPrompt(conversationId: string, style: string): Json | null {
    if (!this.#isStAuthoritative()) return null;
    const entry = this.#shortTerm.get(conversationId);
    if (entry === null) return null;
    if (entry.compactionId === null && entry.recentTurns.length === 0) return null;
    return renderForPrompt(entry, style) as unknown as Json;
  }

  /** Same, for `get_compacted`. Requires an actual compaction, not just turns. */
  #localCompacted(conversationId: string, format: string): Json | null {
    if (!this.#isStAuthoritative()) return null;
    const entry = this.#shortTerm.get(conversationId);
    if (entry === null || entry.compactionId === null) return null;
    return renderCompacted(entry, format);
  }

  /** Every diagnostic the SDK emits passes through here. */
  #warn(message: string): void {
    if (this.#logger !== undefined) {
      this.#logger('warn', message);
      return;
    }
    // eslint-disable-next-line no-console
    console.warn(`[synap] ${message}`);
  }

  /** Release timers and connections. Safe to call more than once. */
  async shutdown(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    this.#shortTerm.clear();
    this.#turns.clear();
    // Release every registry slot pointing here, so a later construction
    // builds a fresh client rather than adopting this shut-down one. Only
    // slots still owned by this client are dropped.
    for (const key of this.#registryKeys) registry.unregisterIfOwner(key, this);

    // Stop the stream first: it holds a socket and a heartbeat timer, and
    // leaving either alive is what keeps a finished process from exiting.
    await this.instance.stop_listening();
    this.#compactionSubscribers.removeAll();
    this.#transport.close();
    this.#anticipationCache.clear();
  }
}

function pick(o: Record<string, unknown>, snake: string, camel: string): unknown {
  const a = o[snake];
  return a !== undefined ? a : o[camel];
}

/** Explicit option wins, then the environment, then the SDK default. */
function resolveBaseUrl(options: SynapClientOptions): string | undefined {
  const explicit = options.baseUrl ?? options.api_base_url;
  if (explicit !== undefined) return explicit;
  const fromEnv = getEnv('SYNAP_BASE_URL')?.trim();
  return fromEnv !== undefined && fromEnv !== '' ? fromEnv : undefined;
}

function resolveCredentials(options: SynapClientOptions): Credentials {
  // Read at call time, not module load, so a late process.env mutation works.
  const apiKey = options.apiKey ?? getEnv('SYNAP_API_KEY');
  if (!apiKey) {
    throw new InvalidInputError(
      'No Synap API key found. Set SYNAP_API_KEY in your environment, or pass ' +
        'apiKey to the SynapClient constructor.',
    );
  }
  return {
    apiKey,
    clientId: options.clientId ?? getEnv('SYNAP_CLIENT_ID') ?? '',
    instanceId: options.instanceId ?? getEnv('SYNAP_INSTANCE_ID') ?? '',
  };
}
