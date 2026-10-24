/**
 * Data namespace. Mirrors `maximem_synap.data.interface.DataInterface`.
 *
 * Delete personal data by scope (user, conversation, customer). Validation
 * messages are identical to Python's, and the body has the same keys in the
 * same order, nulls included.
 */

import { InvalidInputError } from '../errors.js';
import type { HttpTransport } from '../transport/http.js';
import type { Json } from '../context/types.js';

export const DATA_SCOPES = ['user', 'conversation', 'customer'] as const;
export type DataScope = (typeof DATA_SCOPES)[number];

/** The most calls `wait` makes. Same as Python's MAX_WAIT_CALLS. */
export const MAX_WAIT_CALLS = 50;

const ID_FIELD: Record<DataScope, string> = {
  user: 'user_id',
  customer: 'customer_id',
  conversation: 'conversation_id',
};

export interface DeleteDataOptions {
  /** `"user"`, `"conversation"` or `"customer"`. Client-wide delete is not supported. */
  scope: DataScope | string;
  user_id?: string | null;
  userId?: string | null;
  customer_id?: string | null;
  customerId?: string | null;
  conversation_id?: string | null;
  conversationId?: string | null;
  /** Count what would be removed and change nothing. No `confirm` needed. */
  dry_run?: boolean;
  dryRun?: boolean;
  /** Required unless `dryRun`. Must equal the id of the scope being deleted. */
  confirm?: string | null;
  /** Repeat the call while the server says `continue` (default true, max 50 calls). */
  wait?: boolean;
}

/** Result of `data.delete` / `data.status`. Mirrors Python's `DeleteDataResult`. */
export interface DeleteDataResult {
  deletion_id: string | null;
  /** `complete`, `partial`, `in_progress` or `dry_run`. */
  status: string;
  scope: string | null;
  dry_run: boolean;
  found: boolean;
  counts: Record<string, unknown>;
  stores: Record<string, unknown>;
  still_present_in: string[];
  fully_removed: boolean;
  /** True when the server's size cap stopped the call. */
  continue: boolean;
  kept: string[];
  note: string | null;
  memories_kept: boolean;
  audit: string | null;
  /** What the SDK cleared on this machine after the server finished. */
  local_cleared: string[];
  [key: string]: unknown;
}

function either<T>(o: Record<string, unknown>, snake: string, camel: string): T | undefined {
  const a = o[snake];
  if (a !== undefined) return a as T;
  const b = o[camel];
  return b === undefined ? undefined : (b as T);
}

/** Validate and build the request body. Throws before any request is sent. */
export function buildDeleteBody(options: DeleteDataOptions, instanceId: string): Json {
  const o = options as unknown as Record<string, unknown>;
  const scope = options.scope as unknown;
  if (scope !== null && typeof scope === 'object') {
    throw new InvalidInputError(
      'scope must be one of: user, conversation, customer. A dict is not ' +
        'accepted here (the rung-path dict used by memories.create and fetch ' +
        'is a different parameter).',
    );
  }
  if (typeof scope !== 'string' || !(DATA_SCOPES as readonly string[]).includes(scope)) {
    throw new InvalidInputError(
      `Invalid scope '${String(scope)}'. Expected one of: ${DATA_SCOPES.join(', ')}`,
    );
  }
  const dryRun = either<boolean>(o, 'dry_run', 'dryRun') ?? false;
  if (typeof dryRun !== 'boolean') throw new InvalidInputError('dry_run must be a boolean');

  const ids: Record<string, string | null | undefined> = {
    user_id: either<string | null>(o, 'user_id', 'userId'),
    customer_id: either<string | null>(o, 'customer_id', 'customerId'),
    conversation_id: either<string | null>(o, 'conversation_id', 'conversationId'),
  };
  for (const [name, value] of Object.entries(ids)) {
    if (value !== undefined && value !== null && (typeof value !== 'string' || value.trim() === '')) {
      throw new InvalidInputError(`${name} must not be empty`);
    }
  }
  const field = ID_FIELD[scope as DataScope];
  const target = ids[field];
  if (target === undefined || target === null) {
    throw new InvalidInputError(`${field} is required for scope='${scope}'`);
  }
  const confirm = options.confirm ?? null;
  if (!dryRun) {
    if (confirm === null || confirm === '') {
      throw new InvalidInputError(
        `confirm is required unless dry_run=True: pass the ${field} being deleted as confirm`,
      );
    }
    if (confirm !== target) {
      throw new InvalidInputError(`confirm must equal the ${field} being deleted`);
    }
  }
  return {
    scope,
    instance_id: instanceId,
    user_id: ids['user_id'] ?? null,
    customer_id: ids['customer_id'] ?? null,
    conversation_id: ids['conversation_id'] ?? null,
    dry_run: dryRun,
    confirm,
  };
}

/** Fill every documented field so callers can rely on the shape. */
function normalise(raw: Record<string, unknown>): DeleteDataResult {
  return {
    deletion_id: null,
    status: 'unknown',
    scope: null,
    dry_run: false,
    found: false,
    counts: {},
    stores: {},
    still_present_in: [],
    // Pessimistic: true only when the server said so.
    fully_removed: false,
    continue: false,
    kept: [],
    note: null,
    memories_kept: false,
    audit: null,
    local_cleared: [],
    ...raw,
  } as DeleteDataResult;
}

export interface DataInterface {
  /**
   * Delete data for one user, one conversation or one customer. Permanent.
   * scope 'conversation' removes turns, summaries, telemetry and the turn buffer ONLY;
   * memories learned from it are kept (use scope 'user' or memories.delete). Data in
   * another instance is never touched; stores that cannot be scoped by instance are
   * skipped and named in still_present_in.
   *
   * scope 'customer' needs a management key (management:* or *); a runtime key gets 403.
   * LLM call traces are not removed (llm_call_bodies up to 14 days, llm_calls up to 180 days).
   *
   * A store that could not be cleared is in `still_present_in` and
   * `fully_removed` is false; it is a result, not an exception. Errors:
   * `InvalidInputError` (bad arguments, caught before any request, or 400),
   * `AuthenticationError` (401/403), `ConflictError` (409), `RateLimitError`.
   */
  delete(options: DeleteDataOptions): Promise<DeleteDataResult>;
  /** The audit record of a delete (status, counts, still_present_in). */
  status(deletionId: string): Promise<DeleteDataResult>;
}

export type DataDeletedHook = (info: {
  scope: string;
  userId: string | null;
  customerId: string | null;
  conversationId: string | null;
}) => string[];

export function createDataInterface(
  transport: HttpTransport,
  // Drops this machine's copies; returns what it cleared. Mirrors Python's `_evict_deleted`.
  onDeleted?: DataDeletedHook,
): DataInterface {
  return {
    async delete(options) {
      const o = options as unknown as Record<string, unknown>;
      await transport.checkCustomerId(
        either<string>(o, 'customer_id', 'customerId'), 'data.delete',
      );
      const body = buildDeleteBody(options, transport.currentCredentials().instanceId);
      const dryRun = body['dry_run'] === true;
      const wait = options.wait ?? true;

      let result: DeleteDataResult | null = null;
      for (let i = 0; i < (wait ? MAX_WAIT_CALLS : 1); i += 1) {
        result = normalise(
          await transport.request<Record<string, unknown>>('data_delete', { body }),
        );
        if (dryRun || !result.continue) break;
      }
      const final = result as DeleteDataResult;
      if (!dryRun && ['complete', 'partial', 'in_progress'].includes(final.status) && onDeleted) {
        try {
          final.local_cleared = onDeleted({
            scope: body['scope'] as string,
            userId: (body['user_id'] as string | null) ?? null,
            customerId: (body['customer_id'] as string | null) ?? null,
            conversationId: (body['conversation_id'] as string | null) ?? null,
          });
        } catch { /* eviction is best effort */ }
      }
      return final;
    },

    async status(deletionId) {
      if (typeof deletionId !== 'string' || deletionId.trim() === '') {
        throw new InvalidInputError('deletion_id is required');
      }
      return normalise(
        await transport.request<Record<string, unknown>>('data_delete_status', {
          pathParams: { deletion_id: deletionId },
          query: { instance_id: transport.currentCredentials().instanceId },
        }),
      );
    },
  };
}
