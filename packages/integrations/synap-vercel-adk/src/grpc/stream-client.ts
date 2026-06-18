/**
 * gRPC bidirectional stream client for the Synap anticipation channel.
 *
 * Mirrors the Python GRPCTransport exactly:
 *   - /synap.v1.SynapService/Listen  (stream StreamEvent → stream StreamResponse)
 *   - Auth: Authorization: Bearer {api_key}  +  x-client-id  +  x-instance-id
 *   - Heartbeat every 30s, reconnect with exponential backoff on failure
 *   - Server pushes ContextBundleProto → stored in AnticipationCache
 *
 * Node.js only — dynamically imported so Edge / browser bundles are unaffected.
 */

import path from 'path';
import { fileURLToPath } from 'url';
import type {
  Credentials,
  ContextBundleMsg,
  ConversationEventMsg,
  ContextUsedEventMsg,
  ContextAssembledEventMsg,
  CachedBundleType,
} from '../types.js';
import type { AnticipationCache } from '../context/anticipation-cache.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Anticipation cache TTL — matches the Python SDK default (1800s / 30 min).
// ttl_hint_seconds from the server is honored when present, with a 60s floor.
const DEFAULT_TTL_MS = 1_800_000;
const MIN_TTL_SECONDS = 60;

// Backoff + heartbeat settings — mirrors Python GRPCTransport constants
const MAX_RECONNECT_ATTEMPTS = 10;
const BACKOFF_BASE_MS = 1_000;
const BACKOFF_MAX_MS = 30_000;
const HEARTBEAT_INTERVAL_MS = 30_000;
const HEARTBEAT_TIMEOUT_MS = 10_000;
const MAX_MISSED_HEARTBEATS = 3;

type GrpcLib = typeof import('@grpc/grpc-js');
type ProtoLib = typeof import('@grpc/proto-loader');

export type StreamState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'closed';

export interface StreamClientOptions {
  host?: string;
  port?: number;
  useTls?: boolean;
}

export class GrpcStreamClient {
  readonly host: string;
  readonly port: number;
  readonly useTls: boolean;

  private state: StreamState = 'disconnected';
  private channel: unknown = null;
  private call: unknown = null;
  private reconnectAttempts = 0;
  private lastPongAt = 0;
  private missedHeartbeats = 0;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private shutdownRequested = false;

  private grpc: GrpcLib | null = null;
  private ServiceStub: unknown = null;

  constructor(
    private readonly credentials: Credentials,
    private readonly cache: AnticipationCache,
    private readonly onStateChange?: (state: StreamState) => void,
    opts: StreamClientOptions = {},
  ) {
    this.host = opts.host ?? 'synap-cloud-prod.maximem.ai';
    this.port = opts.port ?? 443;
    this.useTls = opts.useTls ?? true;
  }

  get currentState(): StreamState { return this.state; }
  get isConnected(): boolean { return this.state === 'connected'; }

  async connect(): Promise<void> {
    await this.loadGrpc();
    this.shutdownRequested = false;
    await this.establish();
    this.startHeartbeat();
  }

  async disconnect(): Promise<void> {
    this.shutdownRequested = true;
    this.stopHeartbeat();
    this.closeCall();
    this.setState('closed');
  }

  async sendConversationEvent(event: ConversationEventMsg): Promise<void> {
    if (!this.isConnected || !this.call) return;
    try {
      const msg = {
        conversation_event: {
          event_type: event.event_type,
          conversation_id: event.conversation_id,
          user_id: event.user_id,
          role: event.role,
          content: event.content,
          customer_id: event.customer_id,
          session_id: event.session_id,
          metadata: event.metadata,
          timestamp_ms: event.timestamp_ms,
          search_queries: event.search_queries,
          context_types: event.context_types,
        },
      };
      (this.call as { write: (msg: unknown) => void }).write(msg);
    } catch {
      // Non-fatal — fire-and-forget telemetry event
    }
  }

  /** Emit ContextUsedEvent after a fetch is served from the anticipation cache.
   *  Drives the server's learning loop. IDs only — never prompt/item content. */
  async sendContextUsedEvent(event: ContextUsedEventMsg): Promise<void> {
    if (!this.isConnected || !this.call) return;
    try {
      (this.call as { write: (msg: unknown) => void }).write({ context_used: { ...event } });
    } catch {
      // Non-fatal — fire-and-forget telemetry event
    }
  }

  /** Emit ContextAssembledEvent after a fetch resolves (any source). Drives the
   *  Requests-page audit enrichment. IDs + composition metadata only. */
  async sendContextAssembledEvent(event: ContextAssembledEventMsg): Promise<void> {
    if (!this.isConnected || !this.call) return;
    try {
      (this.call as { write: (msg: unknown) => void }).write({ context_assembled: { ...event } });
    } catch {
      // Non-fatal — fire-and-forget telemetry event
    }
  }

  // ─── Private ────────────────────────────────────────────────────────────────

  private async loadGrpc(): Promise<void> {
    if (this.grpc) return;

    try {
      const [grpcMod, protoMod] = await Promise.all([
        import('@grpc/grpc-js') as Promise<GrpcLib>,
        import('@grpc/proto-loader') as Promise<ProtoLib>,
      ]);
      this.grpc = grpcMod;

      // Load proto bundled alongside the package
      const protoPath = path.resolve(__dirname, '../../proto/synap_service.proto');
      const pkgDef = protoMod.loadSync(protoPath, {
        keepCase: true,
        longs: Number,
        enums: String,
        defaults: true,
        oneofs: true,
      });

      const pkg = grpcMod.loadPackageDefinition(pkgDef) as unknown as {
        synap: { v1: { SynapService: new (addr: string, creds: unknown, opts: unknown) => unknown } };
      };
      this.ServiceStub = pkg.synap.v1.SynapService;
    } catch (err) {
      throw new Error(
        `@grpc/grpc-js not available. Install it: npm i @grpc/grpc-js @grpc/proto-loader\n${err}`
      );
    }
  }

  private async establish(): Promise<void> {
    this.setState('connecting');
    if (!this.grpc || !this.ServiceStub) throw new Error('gRPC not loaded');

    const target = `${this.host}:${this.port}`;
    const channelCreds = this.useTls
      ? this.grpc.credentials.createSsl()
      : this.grpc.credentials.createInsecure();

    const channelOptions = {
      'grpc.keepalive_time_ms': 30_000,
      'grpc.keepalive_timeout_ms': 10_000,
      'grpc.keepalive_permit_without_calls': 1,
    };

    const stub = new (this.ServiceStub as new (t: string, c: unknown, o: unknown) => unknown)(
      target, channelCreds, channelOptions
    ) as Record<string, unknown>;

    const metadata = new this.grpc.Metadata();
    metadata.add('authorization', `Bearer ${this.credentials.api_key}`);
    metadata.add('x-client-id', this.credentials.client_id);
    metadata.add('x-instance-id', this.credentials.instance_id);

    const call = (stub['Listen'] as Function)(metadata) as {
      write: (msg: unknown) => void;
      on: (event: string, handler: Function) => void;
      end: () => void;
    };

    call.on('data', (msg: unknown) => this.handleMessage(msg as Record<string, unknown>));
    call.on('error', (err: unknown) => { void this.handleDisconnect(`error: ${err}`); });
    call.on('end', () => { void this.handleDisconnect('server_close'); });

    this.call = call;
    this.setState('connected');
    this.reconnectAttempts = 0;
    this.lastPongAt = Date.now();
  }

  private handleMessage(msg: Record<string, unknown>): void {
    if (msg['context_bundle']) {
      this.handleBundle(msg['context_bundle'] as ContextBundleMsg);
    } else if (msg['heartbeat_pong']) {
      this.lastPongAt = Date.now();
      this.missedHeartbeats = 0;
    } else if (msg['signal']) {
      const signal = msg['signal'] as { signal_type: string; reason: string };
      if (signal.signal_type === 'closing') {
        void this.handleDisconnect('server_signal_closing');
      }
    }
  }

  private handleBundle(bundle: ContextBundleMsg): void {
    const bundleType = (bundle.bundle_type as CachedBundleType) || 'anticipation';

    // Compaction updates are a control signal, not cached context: invalidate the
    // conversation's entries and stop. Reactive bundles are not cached either
    // (matches the Python SDK — only anticipation/user_summary are stored).
    if (bundleType === 'compaction_update') {
      if (bundle.anticipation_conversation_id) {
        this.cache.invalidateConversation(bundle.anticipation_conversation_id);
      }
      return;
    }
    if (bundleType === 'reactive') return;

    // Normalise items_by_type from proto shape → cache shape
    const itemsByType: Record<string, import('../types.js').RawContextItem[]> = {};
    for (const [type, list] of Object.entries(bundle.items_by_type ?? {})) {
      itemsByType[type] = (list as { items: import('../types.js').RawContextItem[] }).items ?? [];
    }

    // Honor server ttl_hint_seconds (floored); fall back to the SDK default.
    const ttlHint = bundle.ttl_hint_seconds ?? 0;
    const ttl = ttlHint > 0 ? Math.max(ttlHint, MIN_TTL_SECONDS) * 1000 : DEFAULT_TTL_MS;

    this.cache.store({
      bundleId: bundle.bundle_id ?? '',
      itemsByType,
      conversationContext: bundle.conversation_context ?? null,
      bundleType,
      userId: bundle.anticipation_user_id ?? '',
      customerId: bundle.anticipation_customer_id ?? '',
      conversationId: bundle.anticipation_conversation_id ?? '',
      searchKeywords: bundle.search_keywords ?? [],
      searchQueries: bundle.search_queries ?? [],
      sourceBundleIds: bundle.bundle_id ? [bundle.bundle_id] : [],
      totalTokens: bundle.total_tokens ?? 0,
      bundleConfidence: bundle.bundle_confidence ?? 0,
      originPatternId: bundle.origin_pattern_id ?? '',
      storedAt: Date.now(),
      ttl,
    });
  }

  private async handleDisconnect(reason: string): Promise<void> {
    if (this.shutdownRequested) return;

    this.closeCall();
    this.setState('reconnecting');

    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      this.setState('disconnected');
      return;
    }

    const delay = Math.min(
      BACKOFF_BASE_MS * Math.pow(2, this.reconnectAttempts) + Math.random() * 1_000,
      BACKOFF_MAX_MS,
    );
    this.reconnectAttempts++;

    await new Promise(r => setTimeout(r, delay));
    if (!this.shutdownRequested) {
      try { await this.establish(); }
      catch { void this.handleDisconnect('reconnect_failed'); }
    }
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (!this.isConnected || !this.call) return;

      const now = Date.now();
      const elapsed = now - this.lastPongAt;

      if (elapsed > HEARTBEAT_TIMEOUT_MS) {
        this.missedHeartbeats++;
        if (this.missedHeartbeats >= MAX_MISSED_HEARTBEATS) {
          void this.handleDisconnect('heartbeat_timeout');
          return;
        }
      }

      try {
        (this.call as { write: (msg: unknown) => void }).write({
          heartbeat_ping: { timestamp_ms: now },
        });
      } catch {
        void this.handleDisconnect('heartbeat_write_error');
      }
    }, HEARTBEAT_INTERVAL_MS);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private closeCall(): void {
    try { (this.call as { end: () => void } | null)?.end(); } catch { /**/ }
    this.call = null;
  }

  private setState(s: StreamState): void {
    this.state = s;
    this.onStateChange?.(s);
  }
}
