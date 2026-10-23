/**
 * The gRPC stream, opened on first use.
 *
 * The stream is the default path: a caller who never heard of `listen()` still
 * gets turns, tool calls and reasoning to the server, and anticipation bundles
 * back. The first model call opens it; every later one reuses it.
 *
 * - Lazy and deduped: concurrent first calls share one open.
 * - Bounded: an open that takes longer than 10s is abandoned, and after a
 *   failure nothing retries for 60s. Until then callers get `null` and use
 *   REST.
 * - Never throws: a model call must not fail because telemetry could not
 *   connect. Edge runtimes, where gRPC cannot load, degrade to REST silently.
 * - Opt out with `SYNAP_AUTO_LISTEN=false` or `autoListen: false`. A stream
 *   the caller opened with `listen()` is used either way.
 */

import type { GrpcStreamClient, StreamState } from './grpc/stream-client.js';
import type { AnticipationCache } from './context/anticipation-cache.js';
import type { Credentials, SynapProviderOptions } from './types.js';

const OPEN_TIMEOUT_MS = 10_000;
const RETRY_AFTER_MS = 60_000;

type StreamClientModule = typeof import('./grpc/stream-client.js');

function isNode(): boolean {
  return typeof process !== 'undefined' && process.versions?.node != null;
}

function envAutoListen(): boolean {
  if (typeof process === 'undefined') return true;
  const v = process.env?.['SYNAP_AUTO_LISTEN'];
  if (v === undefined) return true;
  return !['0', 'false', 'no', 'off'].includes(v.trim().toLowerCase());
}

export class StreamManager {
  private client: GrpcStreamClient | null = null;
  private mod: StreamClientModule | null = null;
  private opening: Promise<GrpcStreamClient | null> | null = null;
  private failedAt: number | null = null;
  private warned = false;
  /** Set by an explicit stopListening(): the caller wants it closed. */
  private stopped = false;

  constructor(
    private readonly credentials: () => Credentials | null,
    private readonly cache: AnticipationCache,
    private readonly opts: Pick<SynapProviderOptions, 'grpcHost' | 'grpcPort' | 'grpcUseTls' | 'autoListen'>,
  ) {}

  /** The live stream, if there is one. Opens nothing. */
  get current(): GrpcStreamClient | null {
    return this.client?.isConnected ? this.client : null;
  }

  private autoListenEnabled(): boolean {
    if (this.stopped) return false;
    if (this.opts.autoListen !== undefined) return this.opts.autoListen;
    return envAutoListen();
  }

  /**
   * The stream, opened if there is none and opening is allowed. Null means
   * this call cannot use one: write over REST. Never throws.
   */
  async ensure(): Promise<GrpcStreamClient | null> {
    const live = this.current;
    if (live) return live;
    if (!this.autoListenEnabled() || !isNode()) return null;
    if (this.opening) return this.opening;
    const state = this.client?.currentState;
    // Recovering by itself: REST for now, and no second client beside it.
    if (state === 'connecting' || state === 'reconnecting') return null;
    if (this.client) {
      // It gave up (or was closed): drop it so a fresh one can open.
      const dead = this.client;
      this.client = null;
      await dead.disconnect().catch(() => undefined);
    }
    if (this.failedAt !== null && Date.now() - this.failedAt < RETRY_AFTER_MS) return null;
    return this.open(false);
  }

  /** Explicit `listen()`: open now, ignoring the cooldown and any earlier stop. */
  async listen(): Promise<void> {
    this.stopped = false;
    this.failedAt = null;
    if (this.current) return;
    await (this.opening ?? this.open(true));
  }

  async stop(): Promise<void> {
    this.stopped = true;
    const client = this.client;
    this.client = null;
    await client?.disconnect();
  }

  private open(loud: boolean): Promise<GrpcStreamClient | null> {
    const attempt = this.connect(loud).finally(() => { this.opening = null; });
    this.opening = attempt;
    return attempt;
  }

  private async connect(loud: boolean): Promise<GrpcStreamClient | null> {
    const creds = this.credentials();
    if (!creds) return null;
    let candidate: GrpcStreamClient | null = null;
    try {
      this.mod ??= await import('./grpc/stream-client.js');
      candidate = new this.mod.GrpcStreamClient(
        creds,
        this.cache,
        (state: StreamState) => {
          // Non-fatal: HTTP context fetch and REST turn writes remain active.
          void state;
        },
        { host: this.opts.grpcHost, port: this.opts.grpcPort, useTls: this.opts.grpcUseTls },
      );
      await candidate.connect({ waitForReadyMs: OPEN_TIMEOUT_MS });
      this.client = candidate;
      this.failedAt = null;
      return candidate;
    } catch (err) {
      await candidate?.disconnect().catch(() => undefined);
      this.client = null;
      this.failedAt = Date.now();
      if (loud || !this.warned) {
        this.warned = true;
        console.warn(
          '[synap] could not open the gRPC stream; turns go over REST until it opens. '
          + 'SYNAP_AUTO_LISTEN=false stops trying.',
          err,
        );
      }
      return null;
    }
  }
}
