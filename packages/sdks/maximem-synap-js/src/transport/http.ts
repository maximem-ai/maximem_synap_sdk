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
  InstanceNotReadyError,
  InstanceBuildFailedError,
  InsufficientCreditsError,
  InvalidInputError,
  NetworkTimeoutError,
  RateLimitError,
  ServiceUnavailableError,
  SynapError,
  SynapPermanentError,
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

/**
 * Client-side cooldown for `instance_not_ready`.
 *
 * The server refuses memory calls while an instance's memory architecture is
 * being built, and says how long to wait. After one refusal the SDK answers the
 * same refusal locally, with no network call, until that wait has passed.
 *
 * Only the routes the server gates (`require_instance_ready` in
 * `synap/cloud/application/admin_api/routes/sdk_*.py` and `migrations.py`) are
 * covered. Everything else (whoami, health, auth, status, credits, telemetry,
 * forget, compaction, profile reads) is not gated by the server and is never
 * short-circuited here. Matching is on the path (or a sub-path) AND the POST
 * method. Same list as `GATED_POST_PATHS` in `http_client.py`.
 */
export const GATED_POST_PATHS: readonly string[] = [
  '/v1/context/conversation/fetch', // sdk_context.py
  '/v1/context/user/fetch', // sdk_context.py
  '/v1/context/customer/fetch', // sdk_context.py
  '/v1/context/client/fetch', // sdk_context.py
  '/v1/conversations/messages', // sdk_conversations.py (+ /messages/batch)
  '/v1/conversations/ingest', // sdk_conversations.py
  '/v1/events/batch', // sdk_events.py
  '/api/v1/memories/create', // sdk_memories.py
  '/api/v1/memories/batch', // sdk_memories.py
  '/api/v1/memories/upload', // sdk_memories.py
  '/v1/migrations', // migrations.py (create, /{id}/approve, /{id}/run)
];

/** The most the SDK will ever wait on a server hint (seconds). */
export const COOLDOWN_MAX_SECONDS = 300;

/** True when the server refuses this call while the instance is building. */
export function isGatedPath(method: string, path: string): boolean {
  if (method.toUpperCase() !== 'POST') return false;
  const bare = (path.split('?', 1)[0] ?? '').replace(/\/+$/, '');
  return GATED_POST_PATHS.some((p) => bare === p || bare.startsWith(p + '/'));
}

export interface TimeoutConfig {
  /**
   * Seconds. Same fields as TimeoutConfig in the Python SDK, but only `read`
   * is applied here: it limits the whole request. `connect` and `write` are
   * accepted so a config round-trips between the SDKs, and have no effect.
   */
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
  /** Monotonic clock in milliseconds. Injected in tests; defaults to `performance.now`. */
  clock?: () => number;
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
  private readonly clock: () => number;
  // instance id -> when the refusal's wait ends (monotonic ms), and what was said.
  // One JS thread runs every request, so a plain Map needs no lock.
  private readonly notReadyCooldown = new Map<
    string, { until: number; message: string; state: string | null }
  >();

  constructor(options: HttpTransportOptions) {
    this.clock = options.clock ?? (() => performance.now());
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

  /**
   * Set by the client to its identity resolution (`initialize()`'s whoami).
   * Python refuses every call before `initialize()`; here the first request
   * runs it instead, so a caller who skips it gets the same identity-dependent
   * behaviour (client-scope cache hits, the B2C check) rather than an error.
   */
  identityResolver: (() => Promise<void>) | undefined = undefined;

  /** Resolve identity once, before the first request. Never throws. */
  async ready(): Promise<void> {
    if (this.identityResolver !== undefined) await this.identityResolver();
  }

  /**
   * Refuse a customer_id on a B2C instance, at the call site. Resolves identity
   * first: the isolation comes from whoami, so on the first call of a client
   * that skipped initialize() the check would otherwise see nothing and pass.
   */
  async checkCustomerId(customerId: unknown, where: string): Promise<void> {
    await this.ready();
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

  /** Throw InstanceNotReadyError locally if this instance is cooling down. */
  private cooldownCheck(instanceId: string, correlationId: string): void {
    const entry = this.notReadyCooldown.get(instanceId);
    if (entry === undefined) return;
    const remainingMs = entry.until - this.clock();
    if (remainingMs <= 0) {
      this.notReadyCooldown.delete(instanceId);
      return;
    }
    throw new InstanceNotReadyError(entry.message, {
      correlationId,
      retryAfterSeconds: Math.max(1, Math.ceil(remainingMs / 1000)),
      instanceId,
      state: entry.state,
      fromCooldown: true,
    });
  }

  /** Remember a server refusal. Never longer than the server's hint. */
  private cooldownStart(instanceId: string, error: InstanceNotReadyError): void {
    const hint = error.retryAfterSeconds;
    if (hint === null || !(hint > 0)) return; // no hint: nothing to base a wait on
    const waitMs = Math.min(hint, COOLDOWN_MAX_SECONDS) * 1000;
    this.notReadyCooldown.set(instanceId, {
      until: this.clock() + waitMs,
      message: error.message,
      state: error.state,
    });
  }

  async request<T = unknown>(name: EndpointName, options: RequestOptions = {}): Promise<T> {
    const spec = ENDPOINTS[name];
    const correlationId = options.correlationId ?? newCorrelationId();
    const idempotent = options.idempotent ?? spec.idempotent;
    // whoami IS the identity resolution, so it must not wait on itself.
    if (name !== 'whoami') await this.ready();

    const gated = isGatedPath(spec.method, resolvePath(name, options.pathParams));
    const cooldownKey = this.#credentials.instanceId;
    if (gated) this.cooldownCheck(cooldownKey, correlationId);

    let attempt = 0;
    let last: AttemptOutcome | null = null;

    // At least one attempt: maxAttempts 0 used to send nothing and throw
    // "Request failed with no recorded error".
    while (attempt < Math.max(1, this.retryPolicy.maxAttempts)) {
      attempt += 1;
      try {
        const result = await this.attempt<T>(name, options, correlationId);
        // The server accepted a memory call: the instance is ready.
        if (gated) this.notReadyCooldown.delete(cooldownKey);
        return result;
      } catch (error) {
        // Remember the refusal so the next calls need no round trip.
        if (gated && error instanceof InstanceNotReadyError && !error.fromCooldown) {
          this.cooldownStart(cooldownKey, error);
        }
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

  // Connection-level failures mean the request never reached the application.
  // Checked before the class: since 0.5.13 they are NetworkTimeoutErrors too,
  // and they must stay safe to retry.
  if (isConnectionRefusal(error)) return false;

  // Timeouts are the ambiguous case: the request may have been handled and
  // only the response lost.
  if (error instanceof NetworkTimeoutError) return true;

  return true;
}

const CONNECTION_REFUSAL_CODES = new Set([
  'ECONNREFUSED', 'ENOTFOUND', 'EAI_AGAIN', 'ECONNRESET', 'EHOSTUNREACH',
]);

/**
 * A refusal code anywhere on the cause chain. fetch nests the socket error
 * under its own, and a SynapError carries a `code` of its own
 * ('network_timeout'), so the walk cannot stop at the first `code` it sees.
 */
function isConnectionRefusal(error: unknown): boolean {
  let cur: unknown = error;
  for (let depth = 0; depth < 4 && cur !== null && typeof cur === 'object'; depth += 1) {
    const code = (cur as { code?: unknown }).code;
    if (typeof code === 'string' && CONNECTION_REFUSAL_CODES.has(code)) return true;
    cur = (cur as { cause?: unknown }).cause;
  }
  return false;
}

function toTransportError(error: unknown, correlationId: string): SynapError {
  if (error instanceof SynapError) return error;
  const name = (error as { name?: string })?.name;
  if (name === 'AbortError' || name === 'TimeoutError') {
    return new NetworkTimeoutError('Request timed out', { correlationId, cause: error });
  }
  // A refused, unresolvable or dropped connection is a NetworkTimeoutError,
  // as Python raises it (httpx ConnectError and RemoteProtocolError), so one
  // `catch` handles "could not reach Synap" in both SDKs. Until 0.5.13 these
  // were ServiceUnavailableErrors, the class for a 5xx the server sent.
  if (isConnectionRefusal(error)) {
    return new NetworkTimeoutError(
      `Connection failed: ${(error as Error)?.message ?? String(error)}`,
      { correlationId, cause: error },
    );
  }
  return new ServiceUnavailableError(
    `Network request failed: ${(error as Error)?.message ?? String(error)}`,
    { correlationId, cause: error },
  );
}

/**
 * Values the credit gate sends as the body's `error` field. Only these are read
 * as a reason, so an unrelated 429 carrying some other `error` string is not
 * mistaken for a credit stop. Mirrors `_CREDIT_ERRORS` in `http_client.py`.
 */
const CREDIT_ERRORS = new Set(['insufficient_credits', 'trial_limit_reached']);

/**
 * The credit gate's structured body, however the server nested it.
 *
 * The gate raises an HTTPException, so FastAPI wraps its payload in `detail`;
 * the burn-service path returns the same fields at the top level. Reading only
 * the top level (what this transport did) meant the gate's balance, reason and
 * URLs never reached the caller.
 */
function creditPayload(
  parsed: Record<string, unknown>,
  detailObject: Record<string, unknown> | null,
): Record<string, unknown> {
  return detailObject ?? parsed;
}

/** Why the request was refused, or null if this was not a credit stop. */
function creditReason(payload: Record<string, unknown>): string | null {
  const reason = stringOrNull(payload['reason']);
  if (reason !== null) return reason;
  const error = stringOrNull(payload['error']);
  return error !== null && CREDIT_ERRORS.has(error) ? error : null;
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
    case 402: {
      // The server sends `minimum_required_credits` (see
      // synap/cloud/application/credits/enforcement.py). We read
      // `required_credits` for a while, a key that never exists, so
      // `requiredCredits` was always null. `recovery_url` and `redeem_url`
      // were dropped entirely, leaving no way to surface a top-up link.
      const credit = creditPayload(parsed, detailObject);
      return new InsufficientCreditsError(message, {
        ...opts,
        balanceCredits: numberOrNull(credit['balance_credits']),
        requiredCredits:
          numberOrNull(credit['minimum_required_credits']) ??
          numberOrNull(credit['required_credits']),
        recoveryUrl: stringOrNull(credit['recovery_url']),
        redeemUrl: stringOrNull(credit['redeem_url']),
        reason: creditReason(credit),
        manageUrl: stringOrNull(credit['manage_url']),
      });
    }
    case 409:
      // Conflict is PERMANENT and never retried. Python discriminates on the
      // structured body so a transcript-immutability conflict becomes
      // TranscriptConflictError and anything else (e.g. compact()'s "already
      // in progress") stays the generic ConflictError. Without this, a
      // customer's `catch (e) { if (e instanceof TranscriptConflictError) }`
      // never fires in JS while the identical Python code works.
      if (
        parsed['error'] === 'instance_build_failed' ||
        detailObject?.['code'] === 'instance_build_failed' ||
        detailObject?.['error'] === 'instance_build_failed'
      ) {
        // Not time based: no cooldown, no retry. Mirrors `_instance_build_failed`.
        const pick = (key: string): unknown => parsed[key] ?? detailObject?.[key];
        const msg = pick('message');
        const attempts = Number(pick('attempts'));
        return new InstanceBuildFailedError(
          typeof msg === 'string' && msg !== ''
            ? msg
            : "This instance's memory architecture build failed.",
          {
            ...opts,
            instanceId: stringOrNull(pick('instance_id')),
            attempts: pick('attempts') !== undefined && pick('attempts') !== null
              && Number.isFinite(attempts) ? Math.trunc(attempts) : null,
            hint: stringOrNull(pick('hint')),
          },
        );
      }
      if (
        parsed['error'] === 'instance_not_ready' ||
        detailObject?.['code'] === 'instance_not_ready' ||
        detailObject?.['error'] === 'instance_not_ready'
      ) {
        // Fields come from the flat copy, then the `detail` copy, then the
        // Retry-After header. Mirrors `_instance_not_ready` in http_client.py.
        const pick = (key: string): unknown => parsed[key] ?? detailObject?.[key];
        const rawWait = pick('retry_after_seconds') ?? response.headers.get('retry-after');
        const wait =
          rawWait !== null && rawWait !== undefined && String(rawWait).trim() !== ''
            ? Number(rawWait)
            : NaN;
        const msg = pick('message');
        return new InstanceNotReadyError(
          typeof msg === 'string' && msg !== ''
            ? msg
            : "This instance's memory architecture is still being built.",
          {
            ...opts,
            retryAfterSeconds: Number.isFinite(wait) ? Math.trunc(wait) : null,
            instanceId: stringOrNull(pick('instance_id')),
            state: stringOrNull(pick('state')) ?? 'building',
          },
        );
      }
      if (detailObject !== null && detailObject['code'] === 'transcript_conflict') {
        return new TranscriptConflictError(message, opts);
      }
      return new ConflictError(message, opts);
    case 429: {
      // Either an ordinary rate limit or a credit stop that time or a plan
      // change clears: the spent Trial cap (`trial_limit_reached`) and a paid
      // plan whose subscription lapsed (`subscription_inactive`). The credit
      // fields are absent on the plain rate limit.
      const header = response.headers.get('retry-after');
      const retryAfter = header !== null && header.trim() !== '' ? Number(header) : null;
      const credit = creditPayload(parsed, detailObject);
      return new RateLimitError(message, {
        ...opts,
        retryAfterSeconds: retryAfter !== null && Number.isFinite(retryAfter) ? retryAfter : null,
        reason: creditReason(credit),
        upgradeUrl: stringOrNull(credit['upgrade_url']),
        manageUrl: stringOrNull(credit['manage_url']),
      });
    }
    case 503:
      return new ServiceUnavailableError(message, opts);
    case 504:
      return new AgentUnavailableError(message, opts);
    default:
      if (response.status >= 500) return new ServiceUnavailableError(message, opts);
      // Any other 4xx is the request's fault and fails the same way again:
      // permanent, as Python raises it. It was a bare SynapError.
      if (response.status >= 400) return new SynapPermanentError(message, opts);
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
