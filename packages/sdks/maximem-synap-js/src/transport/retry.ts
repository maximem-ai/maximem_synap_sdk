/**
 * Retry policy, mirroring `maximem_synap/resilience/retry.py` plus the
 * idempotency gate added there for defect D.
 *
 * ## Why idempotency is part of the retry decision
 *
 * The obvious retry rule is "retry transient errors". That rule double-bills.
 *
 * `POST /api/v1/memories/create` is metered ingestion. If the request reached
 * the server, was processed, and the *response* was lost, a retry ingests the
 * same content twice and bills for it twice. The client cannot tell that case
 * apart from "the request never arrived" by looking at the error type alone.
 *
 * So retries are gated on two independent questions:
 *
 *   1. Is the error transient?          (worth retrying at all)
 *   2. Is the outcome *knowable*?       (safe to retry a non-idempotent call)
 *
 * A connection error means the request provably never reached the application,
 * so the outcome is known: nothing happened. A read timeout means the request
 * may well have been processed. That is the ambiguous case, and for a
 * non-idempotent method it is where we stop.
 *
 * The residual window -- server accepted the request, began processing, then
 * died before responding -- cannot be closed from the client at all. Closing it
 * requires a server-honoured idempotency key on the ingest route. Until that
 * exists, this policy narrows the window rather than eliminating it.
 */

import { RateLimitError, SynapError, isSynapError } from '../errors.js';

/** HTTP methods that are idempotent by specification. */
const IDEMPOTENT_METHODS: ReadonlySet<string> = new Set([
  'GET',
  'HEAD',
  'OPTIONS',
  'PUT',
  'DELETE',
]);

export interface RetryPolicy {
  maxAttempts: number;
  /** Base delay in seconds. */
  backoffBase: number;
  /** Delay cap in seconds. */
  backoffMax: number;
  /** Randomise the delay to avoid a thundering herd. */
  backoffJitter: boolean;
}

export const DEFAULT_RETRY_POLICY: RetryPolicy = {
  maxAttempts: 3,
  backoffBase: 1.0,
  backoffMax: 10.0,
  backoffJitter: true,
};

export interface RetryDecisionInput {
  error: unknown;
  /** 1-based. */
  attempt: number;
  policy: RetryPolicy;
  method: string;
  /**
   * True when the request may have been processed by the server despite the
   * error. Read timeouts and mid-flight disconnects are ambiguous; connection
   * errors and definitive HTTP status responses are not.
   */
  outcomeUnknown: boolean;
  /** Overrides idempotency for a route, e.g. a read-only POST. */
  idempotent?: boolean;
}

export interface RetryDecision {
  retry: boolean;
  /** Seconds to wait before the next attempt. */
  delaySeconds: number;
  reason: string;
}

export function shouldRetry(input: RetryDecisionInput): RetryDecision {
  const { error, attempt, policy, method, outcomeUnknown } = input;

  if (attempt >= policy.maxAttempts) {
    return { retry: false, delaySeconds: 0, reason: 'max_attempts_exhausted' };
  }

  if (!isSynapError(error) || !(error as SynapError).transient) {
    return { retry: false, delaySeconds: 0, reason: 'not_transient' };
  }

  const idempotent = input.idempotent ?? IDEMPOTENT_METHODS.has(method.toUpperCase());
  if (outcomeUnknown && !idempotent) {
    // The request may have been processed. For ingestion that means a retry
    // would store and bill twice. Surfacing the error is the cheaper mistake.
    return {
      retry: false,
      delaySeconds: 0,
      reason: 'non_idempotent_outcome_unknown',
    };
  }

  return {
    retry: true,
    delaySeconds: backoffDelay(error, attempt, policy),
    reason: 'transient_retryable',
  };
}

/** Exponential backoff, honouring Retry-After when the server sent one. */
export function backoffDelay(error: unknown, attempt: number, policy: RetryPolicy): number {
  if (error instanceof RateLimitError && error.retryAfterSeconds != null) {
    // The server told us when to come back. Guessing shorter just gets us
    // rejected again and burns quota.
    return Math.min(error.retryAfterSeconds, policy.backoffMax);
  }
  const exponential = policy.backoffBase * 2 ** (attempt - 1);
  const capped = Math.min(exponential, policy.backoffMax);
  if (!policy.backoffJitter) return capped;
  // Full jitter. Equal-jitter still synchronises a fleet after a shared outage.
  return Math.random() * capped;
}

export const _IDEMPOTENT_METHODS = IDEMPOTENT_METHODS;
