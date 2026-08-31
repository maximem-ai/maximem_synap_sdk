/**
 * Native HTTP transport.
 *
 * Port of `maximem_synap/transport/http_client.py`. Uses global `fetch` so the
 * same code runs on Node, Edge, Workers and Bun, with an optional undici
 * dispatcher on Node to restore Python's connection reuse (see keepalive.ts).
 */

import {
  AgentUnavailableError,
  AuthenticationError,
  ContextNotFoundError,
  ConflictError,
  TranscriptConflictError,
  InsufficientCreditsError,
  InvalidInputError,
  NetworkTimeoutError,
  RateLimitError,
  ServiceUnavailableError,
  SynapError,
} from '../errors.js';
import { newCorrelationId } from '../util/correlation.js';
import { ENDPOINTS, resolvePath, type EndpointName } from './endpoints.js';
import {
  DEFAULT_RETRY_POLICY,
  shouldRetry,
  type RetryPolicy,
} from './retry.js';
import {
  getKeepAliveDispatcher,
  Heartbeat,
  HEARTBEAT_INTERVAL_MS,
  type KeepAliveOptions,
} from './keepalive.js';
import { checkCustomerId } from '../scoping.js';

export const DEFAULT_BASE_URL = 'https://synap-cloud-prod.maximem.ai';

export interface TimeoutConfig {
  /** Seconds. Mirrors TimeoutConfig in the Python SDK. */
  connect: number;
  read: number;
  write: number;
  /**
   * gRPC stream idle timeout. Not used by the HTTP transport; the stream
   * client reads it. Present so a config object round-trips between the two
   * SDKs, and because Python's TimeoutConfig has it.
   */
  streamIdle: number;
}

export const DEFAULT_TIMEOUTS: TimeoutConfig = {
  connect: 5.0, read: 30.0, write: 10.0, streamIdle: 60.0,
};

export interface Credentials {
  apiKey: string;
  clientId: string;
  instanceId: string;
}

export interface HttpTransportOptions {
  credentials: Credentials;
  baseUrl?: string;
  timeouts?: Partial<TimeoutConfig>;
  /** `null` disables retries entirely, as Python's `retry_policy=None` does. */
  retryPolicy?: Partial<RetryPolicy> | null;
  keepAlive?: KeepAliveOptions & { heartbeatIntervalMs?: number };
  userAgent?: string;
  /** Injected in tests. Defaults to global fetch. */
  fetchImpl?: typeof fetch;
}

export interface RequestOptions {
  pathParams?: Record<string, string | number>;
  query?: Record<string, string | number | boolean | undefined>;
  body?: unknown;
  /**
   * Send `body` as-is instead of JSON-encoding it, and drop the JSON
   * Content-Type so the runtime can set `multipart/form-data` with its own
   * boundary. Setting Content-Type by hand for FormData produces a body the
   * server cannot parse, because the boundary will not match.
   */
  rawBody?: boolean;
  correlationId?: string;
  /** Overrides the endpoint's declared idempotency. */
  idempotent?: boolean;
  signal?: AbortSignal;
}

interface AttemptOutcome {
  error: unknown;
  /** True when the server may have processed the request despite the failure. */
  outcomeUnknown: boolean;
}

export class HttpTransport {
  readonly baseUrl: string;
  // Mutable so `configure()` can adjust them before the first request, and so
  // `initialize()` can write back the identity resolved from whoami.
  timeouts: TimeoutConfig;
  retryPolicy: RetryPolicy;

  #credentials: Credentials;
  private readonly userAgent: string;
  private readonly fetchImpl: typeof fetch;
  private readonly keepAliveOptions: KeepAliveOptions;
  private readonly heartbeat: Heartbeat;
  private closed = false;

  constructor(options: HttpTransportOptions) {
    this.#credentials = options.credentials;
    // Trailing slash would produce `//v1/...`, which some proxies 404.
    this.baseUrl = (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, '');
    this.timeouts = { ...DEFAULT_TIMEOUTS, ...options.timeouts };
    this.retryPolicy = options.retryPolicy === null
      ? { ...DEFAULT_RETRY_POLICY, maxAttempts: 1 }
      : { ...DEFAULT_RETRY_POLICY, ...options.retryPolicy };
    this.userAgent = options.userAgent ?? '@maximem/synap-js-sdk';
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis);
    this.keepAliveOptions = options.keepAlive ?? {};
    this.heartbeat = new Heartbeat(
      () => this.pingHealth(),
      options.keepAlive?.heartbeatIntervalMs ?? HEARTBEAT_INTERVAL_MS,
    );
  }

  /** Start keeping the connection warm. Opt-in: it is pointless in serverless. */
  startHeartbeat(): void {
    if (!this.closed) this.heartbeat.start();
  }

  /**
   * Release timers. Safe to call more than once.
   *
   * This is the supported shutdown path, deliberately explicit rather than
   * hooked to `process.on('exit')`: those handlers leak a listener per client,
   * never fire in Lambda, and `process` does not exist in Workers (gotcha G-I).
   */
  /**
   * A COPY of the current credentials.
   *
   * A copy, not the live object: handing out the internal reference is how
   * `client.transport.credentials.apiKey` became readable and mutable from
   * outside in the first place.
   */
  currentCredentials(): Credentials {
    return { ...this.#credentials };
  }

  /** Write back identity resolved from `GET /api/v1/auth/whoami`. */
  updateCredentials(patch: Partial<Credentials>): void {
    this.#credentials = { ...this.#credentials, ...patch };
  }

  /**
   * The instance's scoping mode, from whoami. Lives here rather than on the
   * client because every interface factory already receives the transport,
   * and it is per-client, so a module-level global would be wrong the moment a
   * process holds two clients.
   *
   * undefined means the server did not report it, which is true of any
   * deployment older than the field, and must mean "enforce nothing".
   */
  userContextIsolation: string | undefined = undefined;

  /** Refuse a customer_id on a B2C instance, at the call site. */
  checkCustomerId(customerId: unknown, where: string): void {
    checkCustomerId(
      this.userContextIsolation, customerId, where, this.#credentials.instanceId,
    );
  }

  setTimeouts(timeouts: Partial<TimeoutConfig>): void {
    this.timeouts = { ...this.timeouts, ...timeouts };
  }

  /**
   * `null` DISABLES retries, matching Python.
   *
   * Python's `SDKConfig(retry_policy=None)` leaves `self.retry_policy` unset and
   * `max_attempts` falls to 1 (http_client.py:180). We restored the default
   * policy instead, so a caller following the "Disabling Retries" documentation
   * got three attempts and no way to turn them off.
   */
  setRetryPolicy(policy: Partial<RetryPolicy> | null): void {
    this.retryPolicy = policy === null
      ? { ...DEFAULT_RETRY_POLICY, maxAttempts: 1 }
      : { ...this.retryPolicy, ...policy };
  }

  close(): void {
    this.closed = true;
    this.heartbeat.stop();
  }

  async request<T = unknown>(name: EndpointName, options: RequestOptions = {}): Promise<T> {
    const spec = ENDPOINTS[name];
    const correlationId = options.correlationId ?? newCorrelationId();
    const idempotent = options.idempotent ?? spec.idempotent;

    let attempt = 0;
    let last: AttemptOutcome | null = null;

    while (attempt < this.retryPolicy.maxAttempts) {
      attempt += 1;
      try {
        return await this.attempt<T>(name, options, correlationId);
      } catch (error) {
        last = { error, outcomeUnknown: isOutcomeUnknown(error) };
        const decision = shouldRetry({
          error,
          attempt,
          policy: this.retryPolicy,
          method: spec.method,
          outcomeUnknown: last.outcomeUnknown,
          idempotent,
        });
        if (!decision.retry) throw error;
        await sleep(decision.delaySeconds * 1000, options.signal);
      }
    }
    throw last?.error ?? new SynapError('Request failed with no recorded error');
  }

  private async attempt<T>(
    name: EndpointName,
    options: RequestOptions,
    correlationId: string,
  ): Promise<T> {
    const spec = ENDPOINTS[name];
    const url = this.buildUrl(resolvePath(name, options.pathParams), options.query);

    const dispatcher = await getKeepAliveDispatcher(this.keepAliveOptions);
    const signal = withTimeout(this.timeouts.read * 1000, options.signal);

    const init: RequestInit = {
      method: spec.method,
      headers: this.buildHeaders(correlationId, { json: options.rawBody !== true }),
      signal: signal.signal,
    };
    if (options.body !== undefined && spec.method !== 'GET') {
      // `BodyInit` is a DOM-lib type and this package does not include that lib,
      // so the field's own type is used instead of naming it.
      init.body = options.rawBody === true
        ? (options.body as NonNullable<RequestInit['body']>)
        : JSON.stringify(options.body);
    }
    // `dispatcher` is a Node/undici extension. Its ambient type is only
    // present when undici's types are installed, so it is set structurally
    // rather than declared -- otherwise this file fails to compile for anyone
    // who has not installed the optional dependency.
    if (dispatcher) (init as Record<string, unknown>)['dispatcher'] = dispatcher;

    let response: Response;
    try {
      response = await this.fetchImpl(url, init);
    } catch (error) {
      throw toTransportError(error, correlationId);
    } finally {
      signal.dispose();
    }

    if (!response.ok) {
      throw await toHttpError(response, correlationId);
    }

    if (response.status === 204) return undefined as T;
    const text = await response.text();
    if (!text) return undefined as T;
    try {
      return JSON.parse(text) as T;
    } catch {
      throw new SynapError(`Malformed JSON in response from ${url}`, { correlationId });
    }
  }

  private buildUrl(path: string, query?: RequestOptions['query']): string {
    const url = new URL(this.baseUrl + path);
    for (const [k, v] of Object.entries(query ?? {})) {
      if (v !== undefined) url.searchParams.set(k, String(v));
    }
    return url.toString();
  }

  private buildHeaders(
    correlationId: string,
    options: { json?: boolean } = {},
  ): Record<string, string> {
    return {
      Authorization: `Bearer ${this.#credentials.apiKey}`,
      'X-Client-ID': this.#credentials.clientId,
      'X-Instance-ID': this.#credentials.instanceId,
      'X-Correlation-ID': correlationId,
      ...(options.json === false ? {} : { 'Content-Type': 'application/json' }),
      'User-Agent': this.userAgent,
    };
  }

  /** Unauthenticated and unmetered, exactly like the Python heartbeat. */
  private async pingHealth(): Promise<void> {
    const signal = withTimeout(10_000);
    try {
      const dispatcher = await getKeepAliveDispatcher(this.keepAliveOptions);
      const init: RequestInit = { signal: signal.signal };
      if (dispatcher) (init as Record<string, unknown>)['dispatcher'] = dispatcher;
      await this.fetchImpl(`${this.baseUrl}/health`, init);
    } finally {
      signal.dispose();
    }
  }
}

// ─── Error classification ─────────────────────────────────────────────────────

/**
 * Whether the server may have processed the request despite this error.
 *
 * This is the input to the double-billing gate in retry.ts, so the bias is
 * deliberate: anything we cannot positively rule out counts as ambiguous.
 */
export function isOutcomeUnknown(error: unknown): boolean {
  // A definitive HTTP status means the server answered. Nothing ambiguous.
  if (error instanceof RateLimitError) return false;
  if (error instanceof ServiceUnavailableError) return false;
  if (error instanceof AuthenticationError) return false;
  if (error instanceof InvalidInputError) return false;
  if (error instanceof ConflictError) return false;
  if (error instanceof InsufficientCreditsError) return false;
  if (error instanceof ContextNotFoundError) return false;

  // Timeouts are the ambiguous case: the request may have been handled and
  // only the response lost.
  if (error instanceof NetworkTimeoutError) return true;

  // Connection-level failures mean the request never reached the application.
  if (isConnectionRefusal(error)) return false;

  return true;
}

function isConnectionRefusal(error: unknown): boolean {
  const code = (error as { cause?: { code?: string }; code?: string })?.cause?.code
    ?? (error as { code?: string })?.code;
  return (
    code === 'ECONNREFUSED' ||
    code === 'ENOTFOUND' ||
    code === 'EAI_AGAIN' ||
    code === 'ECONNRESET' ||
    code === 'EHOSTUNREACH'
  );
}

function toTransportError(error: unknown, correlationId: string): SynapError {
  if (error instanceof SynapError) return error;
  const name = (error as { name?: string })?.name;
  if (name === 'AbortError' || name === 'TimeoutError') {
    return new NetworkTimeoutError('Request timed out', { correlationId, cause: error });
  }
  return new ServiceUnavailableError(
    `Network request failed: ${(error as Error)?.message ?? String(error)}`,
    { correlationId, cause: error },
  );
}

async function toHttpError(response: Response, correlationId: string): Promise<SynapError> {
  const body = await response.text().catch(() => '');
  let parsed: Record<string, unknown> = {};
  try {
    parsed = JSON.parse(body) as Record<string, unknown>;
  } catch {
    /* body is not JSON; the raw text is still used in the message */
  }
  // `detail` is a string on most routes but a structured object on the ones
  // that need a machine-readable code: {"detail": {"code": ..., "message": ...}}.
  // Casting it straight to string rendered those as "[object Object]".
  const rawDetail = parsed['detail'] ?? parsed['message'] ?? body ?? '';
  const detailObject =
    typeof rawDetail === 'object' && rawDetail !== null
      ? (rawDetail as Record<string, unknown>)
      : null;
  const detail =
    detailObject !== null
      ? String(detailObject['message'] ?? JSON.stringify(detailObject))
      : String(rawDetail);
  const message = `HTTP ${response.status}: ${detail || response.statusText}`;
  const opts = { correlationId };

  switch (response.status) {
    case 400:
    case 422:
      return new InvalidInputError(message, opts);
    case 401:
    case 403:
      return new AuthenticationError(message, opts);
    case 404:
      return new ContextNotFoundError(message, opts);
    case 402:
      // The server sends `minimum_required_credits` (see
      // synap/cloud/application/credits/enforcement.py). We read
      // `required_credits` for a while, a key that never exists, so
      // `requiredCredits` was always null. `recovery_url` and `redeem_url`
      // were dropped entirely, leaving no way to surface a top-up link.
      return new InsufficientCreditsError(message, {
        ...opts,
        balanceCredits: numberOrNull(parsed['balance_credits']),
        requiredCredits:
          numberOrNull(parsed['minimum_required_credits']) ??
          numberOrNull(parsed['required_credits']),
        recoveryUrl: stringOrNull(parsed['recovery_url']),
        redeemUrl: stringOrNull(parsed['redeem_url']),
      });
    case 409:
      // Conflict is PERMANENT and never retried. Python discriminates on the
      // structured body so a transcript-immutability conflict becomes
      // TranscriptConflictError and anything else (e.g. compact()'s "already
      // in progress") stays the generic ConflictError. Without this, a
      // customer's `catch (e) { if (e instanceof TranscriptConflictError) }`
      // never fires in JS while the identical Python code works.
      if (detailObject !== null && detailObject['code'] === 'transcript_conflict') {
        return new TranscriptConflictError(message, opts);
      }
      return new ConflictError(message, opts);
    case 429: {
      const header = response.headers.get('retry-after');
      const retryAfter = header !== null && header.trim() !== '' ? Number(header) : null;
      return new RateLimitError(message, {
        ...opts,
        retryAfterSeconds: retryAfter !== null && Number.isFinite(retryAfter) ? retryAfter : null,
      });
    }
    case 503:
      return new ServiceUnavailableError(message, opts);
    case 504:
      return new AgentUnavailableError(message, opts);
    default:
      if (response.status >= 500) return new ServiceUnavailableError(message, opts);
      return new SynapError(message, opts);
  }
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringOrNull(v: unknown): string | null {
  return typeof v === 'string' && v !== '' ? v : null;
}

// ─── Utilities ────────────────────────────────────────────────────────────────

interface DisposableSignal {
  signal: AbortSignal;
  dispose(): void;
}

/**
 * Compose a timeout with a caller-supplied signal.
 *
 * Hand-rolled rather than `AbortSignal.timeout` + `AbortSignal.any` because
 * `any` is Node 20.3+ and this must also run on Workers and older Bun. The
 * timer is cleared in `dispose` so a fast response does not hold the event
 * loop open for the full timeout.
 */
function withTimeout(ms: number, external?: AbortSignal): DisposableSignal {
  const controller = new AbortController();
  const onExternalAbort = () => controller.abort(external?.reason);
  const timer = setTimeout(() => {
    controller.abort(new NetworkTimeoutError(`Timed out after ${ms}ms`));
  }, ms);
  (timer as { unref?: () => void }).unref?.();

  if (external) {
    if (external.aborted) onExternalAbort();
    else external.addEventListener('abort', onExternalAbort, { once: true });
  }

  return {
    signal: controller.signal,
    dispose() {
      clearTimeout(timer);
      external?.removeEventListener('abort', onExternalAbort);
    },
  };
}

/**
 * Retry backoff.
 *
 * Deliberately NOT `unref`'d, unlike the heartbeat timer in keepalive.ts. That
 * distinction is the whole point: a heartbeat is background work and must not
 * hold the process open, but a backoff is FOREGROUND work that a caller is
 * awaiting. With it unref'd, a process whose only pending work was this sleep
 * exited during the backoff and the awaited request never settled -- Node
 * reports it as "Detected unsettled top-level await". Invisible in a server
 * with other traffic; fatal in a CLI or an idle serverless handler.
 */
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    function onAbort() {
      clearTimeout(timer);
      reject(signal?.reason ?? new Error('Aborted'));
    }
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}
