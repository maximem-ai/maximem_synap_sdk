/**
 * Bidirectional gRPC stream client for the Synap anticipation channel.
 *
 * Mirrors Python's `GRPCTransport` and the synap-vercel-adk client:
 *   - `/synap.v1.SynapService/Listen`, StreamEvent in, StreamResponse out
 *   - Auth via `authorization: Bearer <key>` plus `x-client-id` / `x-instance-id`
 *   - 30s heartbeat, exponential-backoff reconnect
 *   - Server pushes ContextBundleProto, which lands in the AnticipationCache
 *
 * Node-only, and reached exclusively through a lazy `import()` so that merely
 * importing the SDK on Edge or in a Worker stays safe. grpc-js is built on
 * `node:http2`/`node:net`/`node:tls`; a static import fails the BUILD there.
 *
 * The proto arrives as an inlined descriptor (see ./descriptor.ts) rather than
 * being read from disk, so bundlers and Vercel file tracing cannot lose it.
 */

import type { AnticipationCache } from '../context/anticipation-cache.js';
import { SYNAP_PROTO_DESCRIPTOR } from './descriptor.js';
import { newCorrelationId } from '../util/correlation.js';
import { InsufficientCreditsError, RateLimitError, type SynapError } from '../errors.js';

export type StreamState =
  | 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'closed';

/**
 * The ONLY bundle type the cache refuses.
 *
 * NOTE: this deliberately differs from the synap-vercel-adk stream client,
 * which caches an allowlist of `anticipation` and `user_summary` and drops
 * compaction updates. Python's `_handle_anticipated_bundle` stores everything
 * except `reactive`, and Python is the reference implementation, so an
 * allowlist here would silently cache less than the pip SDK does.
 *
 * Reactive bundles are excluded because they answer a fetch that already
 * happened; caching one serves a stale answer to a different question.
 */
const UNCACHEABLE_BUNDLE_TYPE = 'reactive';

// Mirrors the Python GRPCTransport constants.
const MAX_RECONNECT_ATTEMPTS = 10;
const BACKOFF_BASE_MS = 1_000;
const BACKOFF_MAX_MS = 30_000;
const HEARTBEAT_INTERVAL_MS = 30_000;
const HEARTBEAT_TIMEOUT_MS = 10_000;
const MAX_MISSED_HEARTBEATS = 3;

// The outbound buffer, same shape and same numbers as Python's transport
// (SEND_QUEUE_MAX_DEPTH / SEND_QUEUE_MAX_AGE). Events written while the stream
// is down were silently dropped here: `#write` returned early and said nothing,
// so a reconnect lost every turn that happened during it.
const SEND_QUEUE_MAX_DEPTH = 100;
const SEND_QUEUE_MAX_AGE_MS = 5 * 60 * 1_000;
// How long disconnect() waits for the buffer to go out. Short: a shutdown path
// that blocks is worse than a lost event.
const CLOSE_FLUSH_TIMEOUT_MS = 2_000;
/** The server may only SHORTEN the SDK default, never extend it past this floor. */
const MIN_TTL_SECONDS = 60;

export interface StreamCredentials {
  apiKey: string;
  clientId: string;
  instanceId: string;
}

export interface StreamClientOptions {
  host?: string;
  port?: number;
  useTls?: boolean;
  /** Injected in tests so reconnect backoff does not make the suite slow. */
  sleep?: (ms: number) => Promise<void>;
  /** How long to wait for the channel to become ready. Default 10s. */
  connectTimeoutMs?: number;
  onStateChange?: (state: StreamState) => void;
  /** Fires for every bundle, before the cache filter. Backs `listen({onContext})`. */
  onContext?: (bundle: Record<string, unknown>) => void;
  /** Fires only for `compaction_update` bundles. Backs the compaction subscriptions. */
  onCompactionUpdate?: (conversationId: string, bundle: Record<string, unknown>) => void;
}

/**
 * grpc-js status code for RESOURCE_EXHAUSTED. Hard-coded rather than read from
 * `grpc.status`, because this module is loaded before grpc-js on some paths and
 * the numbers are fixed by the gRPC spec.
 */
const GRPC_RESOURCE_EXHAUSTED = 8;

/**
 * The credit gate refuses a call with RESOURCE_EXHAUSTED, these details, and
 * the balance and reason in trailing metadata. See
 * synap/cloud/application/credits/grpc_gate.py.
 */
export const CREDIT_ABORT_DETAILS = 'insufficient_credits';

/**
 * Reasons that end the stream. A reconnect re-runs the same gate with the same
 * balance, so retrying only delays the answer the caller needs. Mirrors
 * `CREDIT_STOP_REASONS` in `transport/grpc_client.py`.
 */
export const CREDIT_STOP_REASONS = new Set([
  'overages_disabled',
  'trial_limit_reached',
  'subscription_inactive',
]);

function metadataValue(metadata: unknown, key: string): string | null {
  if (metadata === null || typeof metadata !== 'object') return null;
  const getter = (metadata as { get?: (k: string) => unknown }).get;
  const raw = typeof getter === 'function'
    ? getter.call(metadata, key)
    : (metadata as Record<string, unknown>)[key];
  const value = Array.isArray(raw) ? raw[0] : raw;
  if (value === undefined || value === null) return null;
  const text = typeof value === 'string' ? value : String(value);
  return text === '' ? null : text;
}

function numberOrNull(value: string | null): number | null {
  if (value === null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * Translate a credit refusal into the same error HTTP would have raised.
 *
 * RESOURCE_EXHAUSTED on its own is indistinguishable from server overload, so
 * this keys on `credit-reason`: `overages_disabled` is an
 * `InsufficientCreditsError` (permanent, a paid plan at zero that has not
 * allowed usage past zero), while `trial_limit_reached` and
 * `subscription_inactive` are a `RateLimitError`. Returns null for anything
 * else, including a RESOURCE_EXHAUSTED that is not a credit refusal and a
 * reason this SDK does not know: the caller then keeps its existing behaviour
 * of treating the abort as transient. Mirrors `credit_error_from_rpc`.
 */
export function creditErrorFromRpc(
  error: unknown,
  correlationId: string | null = null,
): SynapError | null {
  if (error === null || typeof error !== 'object') return null;
  if ((error as { code?: unknown }).code !== GRPC_RESOURCE_EXHAUSTED) return null;

  const metadata = (error as { metadata?: unknown }).metadata;
  const reason = metadataValue(metadata, 'credit-reason');
  if (reason === null || !CREDIT_STOP_REASONS.has(reason)) return null;

  const rawDetails = (error as { details?: unknown }).details;
  const details = typeof rawDetails === 'string' && rawDetails !== ''
    ? rawDetails
    : CREDIT_ABORT_DETAILS;
  const requestId = metadataValue(metadata, 'request-id') ?? correlationId;
  const manageUrl = metadataValue(metadata, 'credit-manage-url');

  if (reason === 'overages_disabled') {
    return new InsufficientCreditsError(details, {
      correlationId: requestId,
      balanceCredits: numberOrNull(metadataValue(metadata, 'credit-balance')),
      requiredCredits: numberOrNull(metadataValue(metadata, 'credit-minimum-required')),
      recoveryUrl: metadataValue(metadata, 'credit-recovery-url'),
      redeemUrl: metadataValue(metadata, 'credit-redeem-url'),
      reason,
      manageUrl,
    });
  }

  return new RateLimitError(details, {
    correlationId: requestId,
    reason,
    upgradeUrl: metadataValue(metadata, 'credit-upgrade-url'),
    manageUrl,
  });
}

/** True for an error this module raised from a credit refusal. */
export function isCreditStop(error: unknown): boolean {
  if (!(error instanceof InsufficientCreditsError) && !(error instanceof RateLimitError)) {
    return false;
  }
  const reason = (error as { reason?: unknown }).reason;
  return typeof reason === 'string' && CREDIT_STOP_REASONS.has(reason);
}

interface DuplexCall {
  write: (msg: unknown) => void;
  on: (event: string, handler: (arg?: unknown) => void) => void;
  end: () => void;
}

export interface ConversationEventInput {
  event_type: string;
  conversation_id?: string;
  user_id?: string;
  customer_id?: string;
  role?: string;
  content?: string;
  session_id?: string;
  metadata?: Record<string, string>;
  timestamp_ms?: number;
  /** proto: ConversationEvent.tool_name */
  tool_name?: string;
  /** proto: ConversationEvent.tool_args_json. Python sends json.dumps(tool_args). */
  tool_args_json?: string;
  /**
   * proto: ConversationEvent.tool_result_json. A result is not a message, so
   * it does not ride in `content`; a string is sent as itself rather than as a
   * quoted JSON string, matching Python.
   */
  tool_result_json?: string;
  /** proto: ConversationEvent.tool_call_id. Ties a result to its call. */
  tool_call_id?: string;
  search_queries?: string[];
  context_types?: string[];
  /**
   * proto: ConversationEvent.event_id. Minted here when absent, and the SAME
   * id travels on every replay — that is what the server's dedupe matches, so
   * a retry cannot double a turn.
   */
  event_id?: string;
  /** proto: ConversationEvent.sent_at_ms. When we FIRST tried, not when it happened. */
  sent_at_ms?: number;
}

export class GrpcStreamClient {
  readonly host: string;
  readonly port: number;
  readonly useTls: boolean;

  #state: StreamState = 'disconnected';
  #call: DuplexCall | null = null;
  #reconnectAttempts = 0;
  #lastPongAt = 0;
  #missedHeartbeats = 0;
  #heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  #shutdownRequested = false;
  // Events that could not be written because there was no stream.
  #sendQueue: Array<{ queuedAt: number; event: ConversationEventInput }> = [];
  // Events written to a live stream that the server has not acknowledged. The
  // queue above covers "the stream was down"; this covers the half that
  // actually loses sessions — the write succeeded, the stream broke before the
  // server processed it, and nobody knows.
  #unacked = new Map<string, { queuedAt: number; event: ConversationEventInput }>();
  // Why the stream stopped, when it stopped for a reason the caller can act on
  // (today: a credit refusal). Mirrors Python's `last_error`.
  #lastError: SynapError | null = null;

  #grpc: typeof import('@grpc/grpc-js') | null = null;
  #ServiceStub: (new (target: string, creds: unknown, opts: unknown) => Record<string, unknown>) | null = null;

  readonly #credentials: StreamCredentials;
  readonly #cache: AnticipationCache;
  readonly #options: StreamClientOptions;
  readonly #sleep: (ms: number) => Promise<void>;
  readonly #connectTimeoutMs: number;

  constructor(
    credentials: StreamCredentials,
    cache: AnticipationCache,
    options: StreamClientOptions = {},
  ) {
    this.#credentials = credentials;
    this.#cache = cache;
    this.#options = options;
    this.host = options.host ?? 'synap-cloud-prod.maximem.ai';
    this.port = options.port ?? 443;
    this.useTls = options.useTls ?? true;
    this.#sleep = options.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
    this.#connectTimeoutMs = options.connectTimeoutMs ?? 10_000;
  }

  get currentState(): StreamState { return this.#state; }
  get isConnected(): boolean { return this.#state === 'connected'; }

  /**
   * The error that ended the stream, if one did.
   *
   * Set when the server refused the stream for credits, so an `onDisconnect`
   * handler can read the balance, the reason and the URL to fix it. Null for an
   * ordinary disconnect.
   */
  get lastError(): SynapError | null { return this.#lastError; }

  async connect(): Promise<void> {
    await this.#loadGrpc();
    this.#shutdownRequested = false;
    this.#lastError = null;
    await this.#establish();
    this.#startHeartbeat();
  }

  async disconnect(): Promise<void> {
    // What is still buffered goes out first. The buffer lives in memory, so
    // whatever remains when this returns is gone for good.
    this.#flushBeforeClose();
    this.#shutdownRequested = true;
    this.#stopHeartbeat();
    this.#closeCall();
    this.#setState('closed');
  }

  /**
   * Never fails the caller's turn, and no longer loses it either.
   *
   * A write that cannot go out is buffered and replayed on reconnect, and a
   * write that does go out is held until the server acknowledges it. Before
   * this, both cases ended at a silent `return`.
   */
  sendConversationEvent(event: ConversationEventInput): void {
    const stamped: ConversationEventInput = {
      ...event,
      event_id: event.event_id && event.event_id !== '' ? event.event_id : newCorrelationId(),
      sent_at_ms: event.sent_at_ms ?? Date.now(),
    };
    if (!this.isConnected || this.#call === null) {
      this.#enqueue(stamped);
      return;
    }
    this.#write({ conversation_event: { ...stamped } });
    this.#trackUnacked(stamped);
  }

  #enqueue(event: ConversationEventInput): void {
    this.#pruneQueue();
    if (this.#sendQueue.length >= SEND_QUEUE_MAX_DEPTH) {
      const evicted = this.#sendQueue.shift();
      console.warn(
        `[synap] send queue full (max=${SEND_QUEUE_MAX_DEPTH}); dropping oldest `
        + `event (event_type=${evicted?.event.event_type})`,
      );
    }
    this.#sendQueue.push({ queuedAt: Date.now(), event });
  }

  #pruneQueue(): void {
    const cutoff = Date.now() - SEND_QUEUE_MAX_AGE_MS;
    while (this.#sendQueue.length > 0 && (this.#sendQueue[0]?.queuedAt ?? 0) < cutoff) {
      const expired = this.#sendQueue.shift();
      console.warn(
        `[synap] queued event expired (age>${SEND_QUEUE_MAX_AGE_MS}ms); dropping `
        + `(event_type=${expired?.event.event_type})`,
      );
    }
  }

  #trackUnacked(event: ConversationEventInput): void {
    const id = event.event_id;
    if (id === undefined || id === '') return;
    const cutoff = Date.now() - SEND_QUEUE_MAX_AGE_MS;
    for (const [oldestId, held] of this.#unacked) {
      if (held.queuedAt >= cutoff && this.#unacked.size < SEND_QUEUE_MAX_DEPTH) break;
      this.#unacked.delete(oldestId);
      console.warn(
        '[synap] unacknowledged event dropped: the server never confirmed it '
        + `(event_type=${held.event.event_type})`,
      );
    }
    this.#unacked.set(id, { queuedAt: Date.now(), event });
  }

  /**
   * Replay what the server never confirmed, then what never went out.
   *
   * That order is the order things happened in: an unacked event was written
   * to the previous stream, so it is older than anything that piled up while
   * there was no stream at all. Safe to repeat — every event carries the id it
   * was first sent with and the server drops a second sighting of it.
   */
  #replayBuffered(): void {
    const unacked = [...this.#unacked.values()];
    const queued = this.#sendQueue;
    this.#sendQueue = [];
    for (const held of [...unacked, ...queued]) {
      this.#write({ conversation_event: { ...held.event } });
    }
    if (unacked.length > 0 || queued.length > 0) {
      console.warn(
        `[synap] replayed ${unacked.length} unacknowledged and ${queued.length} `
        + 'queued event(s) after reconnect',
      );
    }
  }

  sendSessionControl(input: {
    action: string;
    session_id?: string;
    conversation_id?: string;
    user_id?: string;
    customer_id?: string;
  }): void {
    this.#write({ session_control: { ...input } });
  }

  sendContextUsed(event: Record<string, unknown>): void {
    this.#write({ context_used: { ...event } });
  }

  sendContextAssembled(event: Record<string, unknown>): void {
    this.#write({ context_assembled: { ...event } });
  }

  #flushBeforeClose(): void {
    if (!this.isConnected || this.#call === null) {
      if (this.#sendQueue.length > 0 || this.#unacked.size > 0) {
        console.warn(
          `[synap] closing with ${this.#sendQueue.length} queued and `
          + `${this.#unacked.size} unacknowledged event(s) and no stream to send `
          + 'them on; they are lost',
        );
      }
      return;
    }
    const queued = this.#sendQueue;
    this.#sendQueue = [];
    for (const held of queued) {
      this.#write({ conversation_event: { ...held.event } });
    }
  }

  #write(msg: unknown): void {
    if (!this.isConnected || this.#call === null) return;
    try {
      this.#call.write(msg);
    } catch {
      // Non-fatal. These are telemetry and control events; losing one is
      // strictly better than surfacing a stream hiccup to the caller.
    }
  }

  async #loadGrpc(): Promise<void> {
    if (this.#grpc !== null) return;
    const [grpcMod, protoMod] = await Promise.all([
      import(/* @vite-ignore */ '@grpc/grpc-js'),
      import(/* @vite-ignore */ '@grpc/proto-loader'),
    ]);
    this.#grpc = grpcMod;

    // `fromJSON`, not `loadSync`: no path, no disk read, nothing for a bundler
    // to fail to trace. `keepCase` keeps the snake_case field names the server
    // and the Python SDK both use.
    const pkgDef = protoMod.fromJSON(
      SYNAP_PROTO_DESCRIPTOR as unknown as Parameters<typeof protoMod.fromJSON>[0],
      { keepCase: true, longs: Number, enums: String, defaults: true, oneofs: true },
    );
    const pkg = grpcMod.loadPackageDefinition(pkgDef) as unknown as {
      synap: { v1: { SynapService: new (t: string, c: unknown, o: unknown) => Record<string, unknown> } };
    };
    this.#ServiceStub = pkg.synap.v1.SynapService;
  }

  async #establish(): Promise<void> {
    this.#setState('connecting');
    const grpc = this.#grpc;
    const Stub = this.#ServiceStub;
    if (grpc === null || Stub === null) throw new Error('gRPC not loaded');

    const channelCreds = this.useTls
      ? grpc.credentials.createSsl()
      : grpc.credentials.createInsecure();

    const stub = new Stub(`${this.host}:${this.port}`, channelCreds, {
      'grpc.keepalive_time_ms': 30_000,
      'grpc.keepalive_timeout_ms': 10_000,
      'grpc.keepalive_permit_without_calls': 1,
    });

    // Wait for the channel before reporting success.
    //
    // grpc-js connects LAZILY: constructing the stub and creating the call both
    // return immediately, and a bad host surfaces later through the call's
    // 'error' event. Without this, `await listen()` resolved against an
    // unreachable port, `is_listening` read true for a stream that never
    // connected, and `send_message()` then dropped every event silently,
    // because sends are deliberately fire-and-forget. Failing loudly here is
    // the difference between a clear error and an agent that looks like it has
    // no memory.
    await new Promise<void>((resolve, reject) => {
      const deadline = new Date(Date.now() + this.#connectTimeoutMs);
      const waitForReady = (stub as unknown as {
        waitForReady?: (d: Date, cb: (e?: Error) => void) => void;
      }).waitForReady;
      if (typeof waitForReady !== 'function') {
        // Older grpc-js, or a stub that does not expose it. Fall back to the
        // lazy behaviour rather than failing outright.
        resolve();
        return;
      }
      waitForReady.call(stub, deadline, (error?: Error) => {
        if (error) reject(error);
        else resolve();
      });
    });

    const metadata = new grpc.Metadata();
    metadata.add('authorization', `Bearer ${this.#credentials.apiKey}`);
    metadata.add('x-client-id', this.#credentials.clientId);
    metadata.add('x-instance-id', this.#credentials.instanceId);

    const call = (stub['Listen'] as (m: unknown) => DuplexCall)(metadata);
    call.on('data', (msg) => this.#handleMessage(msg as Record<string, unknown>));
    call.on('error', (err) => {
      // A credit refusal arrives here, not from waitForReady: the channel is
      // healthy and the gate aborts the call itself.
      const creditError = creditErrorFromRpc(err);
      if (creditError !== null) {
        this.#handleCreditStop(creditError);
        return;
      }
      void this.#handleDisconnect(`error: ${String(err)}`);
    });
    call.on('end', () => { void this.#handleDisconnect('server_close'); });

    this.#call = call;
    this.#setState('connected');
    this.#reconnectAttempts = 0;
    this.#lastPongAt = Date.now();
    // Before anything else on the new stream: the turns that happened while
    // there was not one.
    this.#replayBuffered();
  }

  #handleMessage(msg: Record<string, unknown>): void {
    const ack = msg['event_ack'] as { event_ids?: string[] } | undefined;
    if (ack) {
      for (const id of ack.event_ids ?? []) this.#unacked.delete(id);
      return;
    }
    if (msg['context_bundle']) {
      this.#handleBundle(msg['context_bundle'] as Record<string, unknown>);
      return;
    }
    if (msg['heartbeat_pong']) {
      this.#lastPongAt = Date.now();
      this.#missedHeartbeats = 0;
      return;
    }
    const signal = msg['signal'] as { signal_type?: string } | undefined;
    if (signal?.signal_type === 'closing') {
      void this.#handleDisconnect('server_signal_closing');
    }
  }

  #handleBundle(bundle: Record<string, unknown>): void {
    const bundleType = String(bundle['bundle_type'] ?? '') || 'anticipation';
    const conversationId = String(bundle['anticipation_conversation_id'] ?? '');

    if (bundleType !== UNCACHEABLE_BUNDLE_TYPE) this.#store(bundle, bundleType, conversationId);

    if (bundleType === 'compaction_update' && conversationId !== '') {
      // Python also deletes this conversation's `compacted_full` /
      // `compacted_context` entries from its separate CacheManager. This SDK
      // has no such cache, and calling dropConversation() here would delete
      // the bundle just stored above, so there is nothing to invalidate.
      this.#options.onCompactionUpdate?.(conversationId, bundle);
    }

    // Fires for every type, after storing, matching Python's ordering.
    this.#options.onContext?.(bundle);
  }

  #store(bundle: Record<string, unknown>, bundleType: string, conversationId: string): void {
    const itemsByType: Record<string, Array<Record<string, unknown>>> = {};
    for (const [type, list] of Object.entries(bundle['items_by_type'] ?? {})) {
      const items = (list as { items?: Array<Record<string, unknown>> })?.items;
      itemsByType[type] = items ?? [];
    }

    const ttlHint = Number(bundle['ttl_hint_seconds'] ?? 0);
    this.#cache.store({
      bundleId: String(bundle['bundle_id'] ?? ''),
      // The cache keys on ONE entity. User id wins when present, so a bundle
      // pushed for a user cannot be served to a different user in the same
      // customer; falling back to customer id keeps org-scope bundles usable.
      entityId:
        String(bundle['anticipation_user_id'] ?? '') ||
        String(bundle['anticipation_customer_id'] ?? '') ||
        null,
      conversationId: conversationId || null,
      // The rung the server retrieved this bundle at. Absent from anything an
      // older server sent, and absent reads as "named no rung" rather than as
      // "matches anything".
      scopeRung: String(bundle['anticipation_scope_rung'] ?? '') || null,
      itemsByType,
      // 0 means "use the SDK default"; anything else is floored at 60s.
      ttlHintSeconds: ttlHint > 0 ? Math.max(ttlHint, MIN_TTL_SECONDS) : null,
      bundleType,
      searchQueries: [
        ...((bundle['search_queries'] as string[] | undefined) ?? []),
        ...((bundle['search_keywords'] as string[] | undefined) ?? []),
      ],
    });
  }

  /**
   * End the stream because the server refused it for credits.
   *
   * Reconnecting cannot clear a credit refusal, so the stream closes and the
   * typed error is kept on `lastError` for the application to read. Mirrors
   * Python's `_handle_credit_stop`.
   */
  #handleCreditStop(error: SynapError): void {
    this.#lastError = error;
    this.#shutdownRequested = true;
    this.#stopHeartbeat();
    this.#closeCall();
    this.#setState('disconnected');
  }

  async #handleDisconnect(_reason: string): Promise<void> {
    if (this.#shutdownRequested) return;

    this.#closeCall();
    this.#setState('reconnecting');

    if (this.#reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      this.#setState('disconnected');
      return;
    }

    // Jittered exponential backoff. The jitter matters with many SDK instances
    // behind one deploy: without it they all retry in lockstep.
    const delay = Math.min(
      BACKOFF_BASE_MS * 2 ** this.#reconnectAttempts + Math.random() * 1_000,
      BACKOFF_MAX_MS,
    );
    this.#reconnectAttempts += 1;

    await this.#sleep(delay);
    if (this.#shutdownRequested) return;
    try {
      await this.#establish();
    } catch (error) {
      if (isCreditStop(error)) {
        this.#handleCreditStop(error as SynapError);
        return;
      }
      void this.#handleDisconnect('reconnect_failed');
    }
  }

  #startHeartbeat(): void {
    this.#heartbeatTimer = setInterval(() => {
      if (!this.isConnected || this.#call === null) return;
      const now = Date.now();

      if (now - this.#lastPongAt > HEARTBEAT_TIMEOUT_MS) {
        this.#missedHeartbeats += 1;
        if (this.#missedHeartbeats >= MAX_MISSED_HEARTBEATS) {
          void this.#handleDisconnect('heartbeat_timeout');
          return;
        }
      }
      try {
        this.#call.write({ heartbeat_ping: { timestamp_ms: now } });
      } catch {
        void this.#handleDisconnect('heartbeat_write_error');
      }
    }, HEARTBEAT_INTERVAL_MS);
    // Never hold the process open on the heartbeat alone: a CLI that finishes
    // its work should exit, not hang for 30s.
    this.#heartbeatTimer.unref?.();
  }

  #stopHeartbeat(): void {
    if (this.#heartbeatTimer !== null) {
      clearInterval(this.#heartbeatTimer);
      this.#heartbeatTimer = null;
    }
  }

  #closeCall(): void {
    try {
      this.#call?.end();
    } catch {
      /* already torn down */
    }
    this.#call = null;
  }

  #setState(state: StreamState): void {
    this.#state = state;
    this.#options.onStateChange?.(state);
  }
}
