/**
 * Synap SDK error taxonomy.
 *
 * Mirrors `maximem_synap/models/errors.py` 1:1, including the four
 * backward-compatibility aliases at the bottom. If you add an error to one
 * SDK, add it to the other in the same change.
 *
 * ## Why every error carries a `.code`
 *
 * This package ships dual ESM + CJS. A consumer whose dependency graph pulls
 * in both formats gets *two* copies of every class in this file, and then:
 *
 *     try { ... } catch (e) { if (e instanceof SynapError) { ... } }
 *
 * silently evaluates to `false` against the copy from the other format. Error
 * handling stops working and nothing throws to tell you. `.code` is a plain
 * string, so it is immune. Prefer it at package boundaries:
 *
 *     if (isSynapError(e) && e.code === 'rate_limit') { ... }
 *
 * `Symbol.hasInstance` is also wired up on the base class, so `instanceof`
 * works across realms too -- but `.code` is the documented contract.
 */

/** Stable, format-independent discriminators. Never renumber or reuse. */
export type SynapErrorCode =
  | 'synap_error'
  | 'transient'
  | 'permanent'
  | 'network_timeout'
  | 'rate_limit'
  | 'insufficient_credits'
  | 'service_unavailable'
  | 'invalid_input'
  | 'invalid_instance_id'
  | 'invalid_conversation_id'
  | 'authentication'
  | 'context_not_found'
  | 'conflict'
  | 'transcript_conflict'
  | 'session_expired'
  | 'listening_already_active'
  | 'listening_not_active'
  | 'agent_unavailable';

const BRAND = Symbol.for('@maximem/synap-js-sdk.SynapError');

export interface SynapErrorOptions {
  correlationId?: string | null;
  cause?: unknown;
}

export class SynapError extends Error {
  /**
   * The single declaration site for each subclass's code.
   *
   * It is `static` rather than a class field on purpose. Under
   * `useDefineForClassFields` (the default at our ES2022 target) a class field
   * is an *instance* property, so it is invisible from the constructor object.
   * `Symbol.hasInstance` below only has the constructor to work with, so a
   * field would leave it unable to tell `AuthenticationError` from
   * `RateLimitError` -- it would report every Synap error as an instance of
   * every other one.
   */
  static readonly errorCode: SynapErrorCode = 'synap_error';
  /** True for errors where a retry may succeed. Static for the same reason. */
  static readonly isTransient: boolean = false;

  /** Own, enumerable, so logs and JSON.stringify include it. */
  readonly code: SynapErrorCode;
  readonly correlationId: string | null;
  readonly transient: boolean;

  constructor(message: string, options: SynapErrorOptions = {}) {
    super(message, options.cause !== undefined ? { cause: options.cause } : undefined);
    const ctor = new.target as typeof SynapError;
    this.name = new.target.name;
    this.code = ctor.errorCode;
    this.transient = ctor.isTransient;
    this.correlationId = options.correlationId ?? null;
    Object.defineProperty(this, BRAND, { value: true, enumerable: false });
    // Without this, subclassing a built-in under a downlevel target breaks the
    // prototype chain and `instanceof` fails even within a single copy.
    Object.setPrototypeOf(this, new.target.prototype);
  }

  static override [Symbol.hasInstance](instance: unknown): boolean {
    if (instance === null || typeof instance !== 'object') return false;
    // Cross-realm / dual-format tolerant: identify by brand, then narrow by the
    // subclass's own code so `e instanceof RateLimitError` stays meaningful.
    if (!(BRAND in instance)) return false;
    const expected = (this as unknown as { errorCode?: SynapErrorCode }).errorCode;
    // Called on something that is not one of our classes: fall back to the brand.
    if (expected === undefined) return true;
    // Everything branded descends from the root.
    if (expected === 'synap_error') return true;
    const actual = (instance as { code?: string }).code;
    return actual === expected || descendsFrom(actual, expected);
  }
}

/** Type guard that works across ESM/CJS copies and realms. */
export function isSynapError(e: unknown): e is SynapError {
  return typeof e === 'object' && e !== null && BRAND in e;
}

export class SynapTransientError extends SynapError {
  static override readonly errorCode: SynapErrorCode = 'transient';
  static override readonly isTransient: boolean = true;
}

export class SynapPermanentError extends SynapError {
  static override readonly errorCode: SynapErrorCode = 'permanent';
  static override readonly isTransient: boolean = false;
}

// ─── Transient ────────────────────────────────────────────────────────────────

export class NetworkTimeoutError extends SynapTransientError {
  static override readonly errorCode: SynapErrorCode = 'network_timeout';
}

export class RateLimitError extends SynapTransientError {
  static override readonly errorCode: SynapErrorCode = 'rate_limit';
  readonly retryAfterSeconds: number | null;

  constructor(
    message: string,
    options: SynapErrorOptions & { retryAfterSeconds?: number | null } = {},
  ) {
    super(message, options);
    this.retryAfterSeconds = options.retryAfterSeconds ?? null;
  }
}

export class ServiceUnavailableError extends SynapTransientError {
  static override readonly errorCode: SynapErrorCode = 'service_unavailable';
}

export class AgentUnavailableError extends SynapTransientError {
  static override readonly errorCode: SynapErrorCode = 'agent_unavailable';
  constructor(message = 'Agent unavailable', options: SynapErrorOptions = {}) {
    super(message, options);
  }
}

// ─── Permanent ────────────────────────────────────────────────────────────────

/**
 * The caller's credit wallet cannot cover this request.
 *
 * HTTP 402, and gRPC RESOURCE_EXHAUSTED with credit-related trailing metadata.
 * Permanent by design: retrying without topping up burns quota and never
 * succeeds. This is the error behind "the agent forgot" reports where retrieval
 * silently returns nothing.
 */
export class InsufficientCreditsError extends SynapPermanentError {
  static override readonly errorCode: SynapErrorCode = 'insufficient_credits';
  readonly balanceCredits: number | null;
  readonly requiredCredits: number | null;
  /** Where the balance can be viewed. Python calls this `recovery_url`. */
  readonly recoveryUrl: string | null;
  /** Where a redeem code can be entered. Python calls this `redeem_url`. */
  readonly redeemUrl: string | null;

  constructor(
    message: string,
    options: SynapErrorOptions & {
      balanceCredits?: number | null;
      requiredCredits?: number | null;
      recoveryUrl?: string | null;
      redeemUrl?: string | null;
    } = {},
  ) {
    super(message, options);
    this.balanceCredits = options.balanceCredits ?? null;
    this.requiredCredits = options.requiredCredits ?? null;
    this.recoveryUrl = options.recoveryUrl ?? null;
    this.redeemUrl = options.redeemUrl ?? null;
  }
}

export class InvalidInputError extends SynapPermanentError {
  static override readonly errorCode: SynapErrorCode = 'invalid_input';
}

/**
 * Raised by `validateInstanceId` before any network call, matching Python's
 * `utils/validators.py`. The message text is identical in both SDKs.
 */
export class InvalidInstanceIdError extends InvalidInputError {
  static override readonly errorCode: SynapErrorCode = 'invalid_instance_id';
}

/** See the note on InvalidInstanceIdError. Raised by `validateConversationId`. */
export class InvalidConversationIdError extends InvalidInputError {
  static override readonly errorCode: SynapErrorCode = 'invalid_conversation_id';
}

export class AuthenticationError extends SynapPermanentError {
  static override readonly errorCode: SynapErrorCode = 'authentication';
}

export class ContextNotFoundError extends SynapPermanentError {
  static override readonly errorCode: SynapErrorCode = 'context_not_found';
}

export class ConflictError extends SynapPermanentError {
  static override readonly errorCode: SynapErrorCode = 'conflict';
}

export class TranscriptConflictError extends ConflictError {
  static override readonly errorCode: SynapErrorCode = 'transcript_conflict';
}

export class SessionExpiredError extends SynapPermanentError {
  static override readonly errorCode: SynapErrorCode = 'session_expired';
}

export class ListeningAlreadyActiveError extends SynapPermanentError {
  static override readonly errorCode: SynapErrorCode = 'listening_already_active';
}

export class ListeningNotActiveError extends SynapPermanentError {
  static override readonly errorCode: SynapErrorCode = 'listening_not_active';
}

// ─── Hierarchy metadata (drives the cross-copy instanceof above) ──────────────

const PARENT: Partial<Record<SynapErrorCode, SynapErrorCode>> = {
  transient: 'synap_error',
  permanent: 'synap_error',
  network_timeout: 'transient',
  rate_limit: 'transient',
  service_unavailable: 'transient',
  agent_unavailable: 'transient',
  insufficient_credits: 'permanent',
  invalid_input: 'permanent',
  invalid_instance_id: 'invalid_input',
  invalid_conversation_id: 'invalid_input',
  authentication: 'permanent',
  context_not_found: 'permanent',
  conflict: 'permanent',
  transcript_conflict: 'conflict',
  session_expired: 'permanent',
  listening_already_active: 'permanent',
  listening_not_active: 'permanent',
};

function descendsFrom(actual: string | undefined, expected: string): boolean {
  let cur = actual as SynapErrorCode | undefined;
  const seen = new Set<string>();
  while (cur && !seen.has(cur)) {
    seen.add(cur);
    if (cur === expected) return true;
    cur = PARENT[cur];
  }
  return false;
}

// ─── Backward-compatibility aliases (mirrors the tail of errors.py) ──────────
// `ConnectionError` deliberately shadows the global in a module scope. Python
// does the same thing; keeping the name means ported code reads identically.

export const SDKError = SynapError;
export const TransientError = SynapTransientError;
export const PermanentError = SynapPermanentError;
export const ConnectionError = NetworkTimeoutError;
