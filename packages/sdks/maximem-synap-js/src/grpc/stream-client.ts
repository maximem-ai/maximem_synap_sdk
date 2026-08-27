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
  search_queries?: string[];
  context_types?: string[];
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

  async connect(): Promise<void> {
    await this.#loadGrpc();
    this.#shutdownRequested = false;
    await this.#establish();
    this.#startHeartbeat();
  }

  async disconnect(): Promise<void> {
    this.#shutdownRequested = true;
    this.#stopHeartbeat();
    this.#closeCall();
    this.#setState('closed');
  }

  /** Fire-and-forget: a dropped event must never fail the caller's turn. */
  sendConversationEvent(event: ConversationEventInput): void {
    this.#write({ conversation_event: { ...event } });
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
    call.on('error', (err) => { void this.#handleDisconnect(`error: ${String(err)}`); });
    call.on('end', () => { void this.#handleDisconnect('server_close'); });

    this.#call = call;
    this.#setState('connected');
    this.#reconnectAttempts = 0;
    this.#lastPongAt = Date.now();
  }

  #handleMessage(msg: Record<string, unknown>): void {
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
    } catch {
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
