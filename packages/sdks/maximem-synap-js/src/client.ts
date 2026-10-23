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

import type { AgentModel } from './agent-model.js';
import { AuthenticationError, InvalidInputError } from './errors.js';
import { getEnv } from './util/env.js';
import { HttpTransport, type Credentials, type HttpTransportOptions } from './transport/http.js';
import { fetchContext, responseShapeOf } from './context/fetch.js';
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
import { ShortTermStore, isComplete, type ShortTermTurn } from './cache/short-term-store.js';
import { TurnStore, type TurnInput } from './cache/turn-store.js';
import { renderExcerpts, searchTurns, toolResultText, type Encoder } from './cache/history-search.js';
import { SEMANTIC_MATCH } from './behavior/contract.js';
import { overlayLocalRecentTurns } from './context/overlay.js';
import { renderForPrompt, renderCompacted } from './context/local-prompt.js';
import { TurnCounter, mergeUserSummary } from './context/user-summary.js';
import {
  tryServeFromCache, servedItemIds, distinctQueryCount, type Scope as AnticipationScope,
} from './context/anticipated.js';
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
   * The model the agent runs on, and optionally its context window. Sent with
   * every recorded turn so the server sizes compaction to the agent's window
   * (it compacts at ~11.7% of it: 15K tokens on a 128K model). A name is
   * enough; the server knows common models' windows. Unset = 128K. Python's
   * `SDKConfig.agent_model` / `agent_context_window`; per-call `model` /
   * `context_window` override them.
   */
  agent_model?: string | undefined;
  agentModel?: string | undefined;
  agent_context_window?: number | undefined;
  agentContextWindow?: number | undefined;

  /**
   * Let the SDK answer `get_context_for_prompt` and `get_compacted` from its
   * own short-term store when the conversation is warm, skipping the cloud
   * round trip. Off by default. `SYNAP_SDK_ST_AUTHORITATIVE` overrides this in
   * either direction.
   *
   * Leave it off for an application that runs more than one process against
   * the same conversation. A process learns the server's message count only
   * when it writes. If it reads before it writes (build the prompt, then
   * record the turn), it cannot know that another process recorded a turn in
   * between, and it answers without that turn. Turn it on where one process
   * owns a conversation for its whole life, such as a voice agent.
   *
   * This is the COST switch. `st_verbatim_overlay` is the correctness one and
   * is independent of it.
   */
  sdk_st_authoritative?: boolean | undefined;
  sdkStAuthoritative?: boolean | undefined;

  /**
   * Your own embedding model for history search after compaction
   * (cache/history-search.ts), any `{ model, encode }`. It replaces the
   * SDK's model for history search; the anticipation cache uses it only if it
   * is the model its thresholds were tuned on (`history_search_model`).
   */
  history_search_encoder?: Encoder | undefined;
  historySearchEncoder?: Encoder | undefined;

  /**
   * The local embedding model, as Python's `history_search_model`: history
   * search after compaction, and matching fetch queries to prefetched items by
   * meaning. "auto" (default) runs the SDK's multilingual model in Node when
   * the optional `onnxruntime-node` and `@huggingface/tokenizers` packages are
   * installed: downloaded once (about 135 MB) into ~/.synap/models when the
   * first stream starts or the first history search runs, then loaded in the
   * background. An "onnx:<repo>@<revision>/<file>" names another (history
   * search only). null keeps keyword matching. Never loaded on Edge or in a
   * browser.
   */
  history_search_model?: string | null | undefined;
  historySearchModel?: string | null | undefined;

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
  /** The agent's model; see `SynapClientOptions.agent_model`. */
  agent_model?: string;
  agentModel?: string;
  agent_context_window?: number;
  agentContextWindow?: number;
  /**
   * The local model for history search and matching a fetch to prefetched
   * context by meaning: "auto" (default) or a model id, null for keyword
   * matching only. As Python's SDKConfig.history_search_model. Since 0.5.19.
   */
  history_search_model?: string | null;
  historySearchModel?: string | null;
  /** Replaces the diagnostic sink, same as the constructor option. */
  logger?: SynapLogger | undefined;
}

/** Options for the cross-scope `fetch()`. Snake_case, matching Python's kwargs.
 *
 * The three identifiers accept `null` as well as `undefined`, because that is
 * what callers actually pass: `conversation_id: convId || null` is the normal
 * way to say "there is no conversation here", and `validateConversationId`
 * has always treated `null` as absent. The declared type said otherwise, so
 * the `SynapClient`-to-`SynapSdkLike` assignability test in
 * `synap-claude-agent` (TS) could not typecheck, and the scope guards below
 * did not agree with the validator either. See the guards for that half.
 */
export interface UnifiedFetchOptions {
  conversation_id?: string | null;
  user_id?: string | null;
  customer_id?: string | null;
  search_query?: string[];
  max_results?: number;
  types?: string[];
  mode?: string;
  precision_level?: string;
  /** The moment the question is being asked. See FetchOptions.query_date. */
  query_date?: string;
  /** A share of the response per search query. See FetchOptions.max_results_per_query. */
  max_results_per_query?: number;
  include_conversation_context?: boolean;
  scopes?: string[];
  include_scope_labels?: boolean;
  context_mode?: string;
  include_profile?: boolean;
  last_n_conversations?: number;
  /**
   * `"by_type"` (default) or `"ranked"`. Ranked asks every scope for its
   * ranked list and merges them into `items`, best first across scopes and
   * types, by each item's `rank_score`; `formatted_context` is then one
   * numbered list in that order. The typed lists are filled either way.
   */
  response_shape?: string;
}

/** `SYNAP_SDK_LOCAL_MODEL=off` keeps the local embedding model from loading. Python: the same. */
function localModelOff(): boolean {
  return ['off', '0', 'false', 'no', 'none'].includes((getEnv('SYNAP_SDK_LOCAL_MODEL') ?? '').trim().toLowerCase());
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
  #identity: Promise<void> | undefined;
  readonly #registryKeys!: string[];
  readonly #compactionSubscribers!: CompactionSubscribers;
  readonly #shortTerm!: ShortTermStore;
  /** Every raw turn, past compaction, for local history search. */
  readonly #turnStore!: TurnStore;
  /** A local embedding model for history search; null = keywords only. */
  #localEncoder: Encoder | null = null;
  /** `history_search_model`: "auto", an "onnx:..." name, or null for none. */
  #localModelName: string | null = 'auto';
  #localModelStarted = false;
  readonly #backfillAfter = new Map<string, number>();
  readonly #stVerbatimOverlay!: boolean | undefined;
  #agentModel: AgentModel = {};
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
    this.#transport.identityResolver = () => this.#resolveIdentity();

    this.#anticipationCache = new AnticipationCache();

    this.#fetchHooks = {
      beforeFetch: async (scope, options) => {
        const o = options as Record<string, unknown>;
        // A prefetched bundle was assembled with no period restriction, so it
        // cannot answer a request that names one. Same class as the scope_path
        // defect described below: a bundle built for one question served to a
        // different one, entirely inside this process and invisible to every
        // server log. Skipping the cache costs one round trip; serving a bundle
        // that ignores the date costs the answer.
        if (o['query_date'] ?? o['queryDate']) return null;
        // A prefetched bundle was not assembled under a per-query share
        // either, and its items do not say which query found them.
        if (o['max_results_per_query'] ?? o['maxResultsPerQuery']) return null;
        const attempt = await tryServeFromCache(
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
          borderline: attempt.hit.band === 'borderline',
          raceDeadlineMs: SEMANTIC_MATCH.race_deadline_ms,
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
      // The periodic user-summary splice, as Python runs it at the end of the
      // user, customer and client fetches (the conversation fetch does its own
      // through `onFetched`). Until 0.5.12 only the conversation fetch had it,
      // so the same usage gave the model different context in JS.
      // Python's `_should_skip_server_st`: when this process already holds the
      // conversation's compaction and every turn after it, the server is told
      // not to assemble them again.
      skipServerShortTerm: (conversationId) => {
        if (!this.#isStAuthoritative()) return false;
        const entry = this.#shortTerm.get(conversationId);
        return entry !== null && entry.compactionId !== null && isComplete(entry);
      },
      onResponse: (scope, options, response) => {
        if (scope === 'conversation') return;
        const o = options as Record<string, unknown>;
        const owner = scope === 'user'
          ? (o['user_id'] ?? o['userId'])
          : scope === 'customer' ? (o['customer_id'] ?? o['customerId']) : '_client';
        this.#injectUserSummary(
          response,
          (o['conversation_id'] ?? o['conversationId']) as string | undefined,
          owner as string | undefined,
          rungKey((o['scope_path'] ?? o['scopePath']) as Record<string, string> | undefined),
        );
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
    // Every raw turn, kept past compaction for local search (turn-store.ts).
    this.#turnStore = new TurnStore();
    this.#shortTerm.attachArchive(this.#turnStore);
    this.#stVerbatimOverlay = options.st_verbatim_overlay ?? options.stVerbatimOverlay;
    this.#agentModel = {
      model: options.agent_model ?? options.agentModel,
      contextWindow: options.agent_context_window ?? options.agentContextWindow,
    };
    this.#logger = options.logger;
    this.#stAuthoritative = options.sdk_st_authoritative ?? options.sdkStAuthoritative;
    this.#localEncoder = options.history_search_encoder ?? options.historySearchEncoder ?? null;
    const modelOption = options.history_search_model !== undefined
      ? options.history_search_model
      : options.historySearchModel;
    this.#localModelName = modelOption === undefined ? 'auto' : modelOption;
    if (this.#localEncoder?.model === SEMANTIC_MATCH.model) this.#anticipationCache.setEncoder(this.#localEncoder);
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
      ({ conversationId, role, content, timestamp, messageId, serverMessageCount, inFlight }) => {
        this.#shortTerm.appendTurn(conversationId, role, content, timestamp, {
          messageId,
          serverMessageCount,
          inFlight,
        });
      },
      (response, conversationId, userId, scopePath) => {
        overlayLocalRecentTurns(response, this.#shortTerm, conversationId, this.#stVerbatimOverlay);
        this.#injectUserSummary(response, conversationId, userId, rungKey(scopePath));
      },
      {
        prompt: (conversationId, style) => this.#localPrompt(conversationId, style),
        compacted: (conversationId, format) => this.#localCompacted(conversationId, format),
        reseed: (conversationId, response) => this.#reseedShortTerm(conversationId, response),
        historyExcerpts: (conversationId, response, query) => this.#withHistoryExcerpts(conversationId, response, query),
      },
      () => this.#agentModel,
      // The same hooks as the other three scopes: the anticipation cache first,
      // as Python's conversation fetch does, and the audit events. Until 0.5.13
      // the conversation fetch was built without them, so every JS conversation
      // fetch was a billed round trip, including ones the stream had answered.
      this.#fetchHooks,
    );
    // After a forget that deleted something, drop that user's cached bundles so
    // they are not served until the TTL runs out. Unconditional, unlike the
    // invalidate-on-write hook: a forgotten memory must not be served.
    this.memories = createMemoriesInterface(this.#transport, ({ userId, customerId }) => {
      for (const id of new Set([userId, customerId])) {
        if (id) this.#anticipationCache.dropEntity(id);
      }
    });
    this.credits = createCreditsNamespace(this.#transport);
    this.cache = createCacheNamespace(this.#anticipationCache);
    this.instance = createInstanceNamespace(
      () => this.#transport.currentCredentials(),
      this.#anticipationCache,
      ({ conversationId, role, content }) => {
        this.#shortTerm.appendTurn(conversationId, role, content);
      },
      () => this.#agentModel,
      ({ conversationId, result, toolName }) => this.archive_tool_result(conversationId, result, { tool_name: toolName }),
      (customerId, where) => this.#transport.checkCustomerId(customerId, where),
    );
    // A compaction folds turns into the server's summary, so the local copies
    // of those turns must go or they keep being spliced back in.
    // Prefetched bundles start arriving with the stream: load the local model
    // then, not at construction, so a process that never listens or searches
    // never downloads it.
    this.instance._internal.setOnListen(() => this.#startLocalModel());
    this.instance._internal.setCompactionApplier((bundle) => {
      this.#shortTerm.applyCompaction(bundle);
    });
    // The server's message count after each streamed write, which is how the
    // store learns whether it holds the whole conversation.
    this.instance._internal.setConversationStateHandler((conversationId, messageCount) => {
      this.#shortTerm.noteServerCount(conversationId, messageCount);
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
   * Mirrors Python's `initialize()`. Python raises "SDK not initialized" on
   * any call made before it; here the first request runs the same resolution
   * instead (`identityResolver` on the transport), so skipping this call costs
   * nothing and breaks no existing caller. It matters for the anticipation
   * cache: a bundle shared at client scope is filtered on `client_id`, and an
   * empty one fails that filter for every bundle.
   *
   * Calling it explicitly still locks `configure()`, as in Python. Resolving
   * on first use does not, because a caller who never called it could always
   * configure after a request, and that keeps working.
   *
   * Best-effort and idempotent. A whoami failure leaves whatever the
   * constructor resolved and is not fatal.
   */
  async initialize(): Promise<void> {
    this.#initialized = true;
    await this.#resolveIdentity();
  }

  /** Runs whoami at most once, however many calls arrive together. */
  #resolveIdentity(): Promise<void> {
    this.#identity ??= this.#whoami();
    return this.#identity;
  }

  async #whoami(): Promise<void> {
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
    // As Python's SDKConfig.history_search_model. Until 0.5.19 only the
    // constructor took it here. It applies to the model the SDK loads when it
    // first needs one; once that has started, changing it does nothing.
    const historyModel = options.history_search_model !== undefined
      ? options.history_search_model
      : options.historySearchModel;
    if (historyModel !== undefined && !this.#localModelStarted) this.#localModelName = historyModel;
    const model = options.agent_model ?? options.agentModel;
    const window = options.agent_context_window ?? options.agentContextWindow;
    if (model !== undefined || window !== undefined) {
      this.#agentModel = {
        model: model ?? this.#agentModel.model,
        contextWindow: window ?? this.#agentModel.contextWindow,
      };
    }
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
      query_date: queryDate,
      max_results_per_query: maxResultsPerQuery,
      include_conversation_context: includeConversationContext = true,
      scopes,
      include_scope_labels: includeScopeLabels = false,
      context_mode: contextMode = 'in-conversation',
      include_profile: includeProfile = true,
      last_n_conversations: lastNConversations = 1,
      response_shape: responseShape = 'by_type',
    } = options;

    // First, as Python checks it: on B2C a customer_id is refused even when
    // `scopes` leaves out every sub-fetch that would have carried it.
    await this.#transport.checkCustomerId(customerId, 'sdk.fetch');
    validateConversationId(conversationId);
    // Up front, as Python does: a bad value inside a scope fetch would be an
    // InvalidInputError too, but only once a scope is queried.
    responseShapeOf({ response_shape: responseShape });

    const wants = (scope: string): boolean => scopes === undefined || scopes.includes(scope);
    const shared = {
      search_query: searchQuery,
      max_results: maxResults,
      types,
      mode,
      precision_level: precisionLevel,
      // In `shared` so it reaches all four branches at once. The Python edit
      // threaded three call sites by hand and missed the client branch, which
      // sits one indent deeper; spreading removes that whole class of miss.
      query_date: queryDate,
      max_results_per_query: maxResultsPerQuery,
      response_shape: responseShape,
    };

    const labels: string[] = [];
    const tasks: Array<Promise<RawContext>> = [];

    // `null` means absent, exactly as `validateConversationId` already reads
    // it. The guards below used to test only `!== undefined && !== ''`, so a
    // caller writing `conversation_id: convId || null` -- which is what the
    // Claude Agent (TS) integration writes, and the obvious way to spell "no
    // conversation" -- sailed past them and fired a conversation-scope fetch
    // carrying `conversation_id: null`. The validator waved it through and the
    // scope was queried for a conversation that does not exist.
    const present = (v: string | null | undefined): v is string =>
      v !== undefined && v !== null && v !== '';

    if (present(conversationId) && wants('conversation')) {
      labels.push('conversation');
      tasks.push(this.conversation.context.fetch({
        conversation_id: conversationId, ...shared,
        user_id: userId, customer_id: customerId,
      } as FetchOptions));
    }
    if (present(userId) && wants('user')) {
      labels.push('user');
      tasks.push(this.user.context.fetch({
        user_id: userId, conversation_id: conversationId, ...shared,
        customer_id: customerId,
        context_mode: contextMode,
        include_profile: includeProfile,
        last_n_conversations: lastNConversations,
      } as FetchOptions));
    }
    if (present(customerId) && wants('customer')) {
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

    if (includeConversationContext && present(conversationId)) {
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
        // How many distinct search queries the fetch carried. A hit never
        // reaches the fetch route, so this event is how the server learns it
        // happened and what it is charged by. The count only: this event must
        // not carry the caller's words.
        query_count: distinctQueryCount(options['search_query'] ?? options['searchQuery']),
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
   * Resolution order matches Python's `_is_st_authoritative`:
   * `SYNAP_SDK_ST_AUTHORITATIVE` first, as an explicit override in either
   * direction, then the option. Off by default. Being enabled does not mean
   * the local store is used: that also takes an entry that holds the whole
   * conversation, see `isComplete`.
   */
  #isStAuthoritative(): boolean {
    const raw = (getEnv('SYNAP_SDK_ST_AUTHORITATIVE') ?? '').trim().toLowerCase();
    if (['0', 'false', 'no', 'off'].includes(raw)) return false;
    if (['1', 'true', 'yes', 'on'].includes(raw)) return true;
    return this.#stAuthoritative === true;
  }

  /**
   * Render `get_context_for_prompt` from the local store, or null to fall
   * through to the network.
   *
   * The store answers only when it holds the whole conversation: the server
   * has confirmed a count the entry accounts for. An entry that is merely
   * present is not enough. After a restart or an eviction it holds one turn,
   * and rendering that as the conversation drops everything said before it.
   */
  /**
   * `response` with the earlier turns that best match `query` (by default the
   * latest user message) appended, when a compaction exists. The search runs
   * here, over the turn archive. Python: `_with_history_excerpts`.
   */
  async #withHistoryExcerpts(conversationId: string, response: Json, query: string | undefined): Promise<Json> {
    const r = (response ?? {}) as Record<string, unknown>;
    const store = this.#turnStore;
    const recent = (Array.isArray(r['recent_messages']) ? r['recent_messages'] : []) as Array<Record<string, unknown>>;
    store.addMany(
      conversationId,
      recent.map((m) => ({
        role: String(m['role'] ?? 'user'),
        content: String(m['content'] ?? ''),
        timestamp: String(m['timestamp'] ?? ''),
        message_id: typeof m['message_id'] === 'string' ? m['message_id'] : null,
      })),
    );
    const entry = this.#shortTerm.get(conversationId);
    const total = Math.max(Number(r['total_message_count'] ?? 0) || 0, Number(entry?.serverMessageCount ?? 0) || 0);
    this.#maybeBackfillTurns(conversationId, total);
    if (!r['context_block']) return response;
    const turns = store.turns(conversationId);
    let q = query;
    if (!q) {
      const latest = [...recent].reverse().find((m) => m['role'] === 'user');
      q = latest ? String(latest['content'] ?? '') : ([...turns].reverse().find((t) => t.role === 'user')?.content ?? '');
    }
    if (!q) return response;
    const exclude = new Set<string>();
    for (const m of recent) {
      if (typeof m['message_id'] === 'string' && m['message_id']) exclude.add(m['message_id']);
      exclude.add(String(m['content'] ?? ''));
    }
    this.#startLocalModel();
    const hits = await searchTurns(turns, q, { exclude, encoder: this.#localEncoder, store });
    if (hits.length === 0) return response;
    const block = renderExcerpts(hits, turns.length ? turns[turns.length - 1]!.timestamp : '');
    return {
      ...r,
      history_excerpts: hits.map((h) => ({ position: h.position, role: h.role, text: h.text })),
      history_block: block,
      formatted_context: (String(r['formatted_context'] ?? '').trimEnd() + '\n\n' + block).trim(),
    } as Json;
  }

  /**
   * Keep what a tool returned in this process's turn archive, so history search
   * can find it after a compaction. Never sent anywhere: the server's
   * conversation holds what people said, and a compaction is made from that.
   * `instance.record_tool_result` calls this too. Python: `archive_tool_result`.
   */
  archive_tool_result(
    conversationId: string,
    result: unknown,
    options: { tool_name?: string | null; toolName?: string | null; timestamp?: string } = {},
  ): void {
    if (!conversationId || result === null || result === undefined) return;
    try {
      this.#turnStore.add(conversationId, 'tool', toolResultText(result, options.tool_name ?? options.toolName),
        options.timestamp ?? new Date().toISOString());
    } catch {
      // The archive is a convenience: never fail the agent's call.
    }
  }

  static readonly BACKFILL_COOLDOWN_MS = 60_000;
  static readonly BACKFILL_PAGE = 500;

  /** Fetch, in the background, turns the server holds and the archive does not. */
  /**
   * Load the local embedding model in the background, once. Searches and
   * lookups before it is ready match by words. Python: `_start_history_encoder`.
   */
  #startLocalModel(): void {
    if (this.#localModelStarted) return;
    this.#localModelStarted = true;
    if (localModelOff()) return;
    const name = this.#localModelName === 'auto' ? SEMANTIC_MATCH.model : this.#localModelName;
    if (!name) return;
    const callerEncoder = this.#localEncoder;
    void import('./cache/onnx-encoder.js')
      .then(({ canRunLocalModel, loadOnnxEncoder }) => (canRunLocalModel() ? loadOnnxEncoder(name) : null))
      .then((encoder) => {
        if (encoder === null) return;
        // A caller's own encoder keeps history search.
        if (callerEncoder === null) this.#localEncoder = encoder;
        // The anticipation thresholds hold for one model only.
        if (encoder.model === SEMANTIC_MATCH.model) this.#anticipationCache.setEncoder(encoder);
      })
      .catch(() => {
        /* words only, as without a model */
      });
  }

  #maybeBackfillTurns(conversationId: string, total: number): void {
    if (total <= this.#turnStore.count(conversationId)) return;
    const now = Date.now();
    if (now < (this.#backfillAfter.get(conversationId) ?? 0)) return;
    this.#backfillAfter.set(conversationId, now + SynapClient.BACKFILL_COOLDOWN_MS);
    void this.#backfillTurns(conversationId, total);
  }

  async #backfillTurns(conversationId: string, total: number): Promise<void> {
    try {
      let start = 1;
      while (start <= total) {
        const result = await this.#transport.request<Json>('conversation_turns', {
          pathParams: { conversation_id: conversationId },
          query: { start, limit: SynapClient.BACKFILL_PAGE },
        });
        const page = (((result ?? {}) as Record<string, unknown>)['turns'] ?? []) as TurnInput[];
        this.#turnStore.addMany(conversationId, page);
        if (page.length < SynapClient.BACKFILL_PAGE) break;
        start += SynapClient.BACKFILL_PAGE;
      }
    } catch (error) {
      this.#warn(`Turn backfill for ${conversationId} failed: ${String(error)}`);
    }
  }

  #localPrompt(conversationId: string, style: string): Json | null {
    if (!this.#isStAuthoritative()) return null;
    const entry = this.#shortTerm.get(conversationId);
    if (entry === null || !isComplete(entry)) return null;
    if (entry.compactionId === null && entry.recentTurns.length === 0) return null;
    return renderForPrompt(entry, style) as unknown as Json;
  }

  /** Same, for `get_compacted`. Requires an actual compaction, not just turns. */
  #localCompacted(conversationId: string, format: string): Json | null {
    if (!this.#isStAuthoritative()) return null;
    const entry = this.#shortTerm.get(conversationId);
    if (entry === null || entry.compactionId === null || !isComplete(entry)) return null;
    return renderCompacted(entry, format);
  }

  /**
   * Bring the local store back in step from the server's answer.
   *
   * Only for a conversation this process writes to. The server's answer lists
   * every message its summary does not cover, which is exactly what the entry
   * was missing.
   */
  #reseedShortTerm(conversationId: string, response: Json): void {
    if (!this.#isStAuthoritative()) return;
    const entry = this.#shortTerm.get(conversationId);
    if (entry === null || !entry.participant) return;
    const r = (response ?? {}) as Record<string, unknown>;
    const recent = Array.isArray(r['recent_messages']) ? r['recent_messages'] : [];
    this.#shortTerm.reseed(conversationId, {
      compactionId: typeof r['compaction_id'] === 'string' ? r['compaction_id'] : null,
      recentMessages: recent as ShortTermTurn[],
      compactedMessageCount: Number(r['compacted_message_count'] ?? 0) || 0,
      totalMessageCount: Number(r['total_message_count'] ?? 0) || 0,
    });
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
  // `||`, not `??`: an empty apiKey falls back to the environment, as
  // Python's `api_key or os.environ.get(...)` does.
  const apiKey = options.apiKey || getEnv('SYNAP_API_KEY');
  // An AuthenticationError, as Python raises it (auth/manager.py), so one
  // `catch` handles "no usable key" in both SDKs. Until 0.5.13 this was an
  // InvalidInputError.
  if (!apiKey) {
    throw new AuthenticationError(
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
