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
  /**
   * Delete in several instances of the client at once: `"all"` or a list of
   * instance ids. Needs a management key not bound to one instance (a bound key
   * may only name its own). Omit it to delete in this SDK's instance only.
   * Mirrors Python's `instances`.
   */
  instances?: 'all' | string[] | null;
  /**
   * Remove the memories now instead of after the deletion period. Needs a
   * management-level key (delete:permanent, delete:*, management:* or *).
   */
  permanent?: boolean;
}

/** Result of `data.delete` / `data.status`. Mirrors Python's `DeleteDataResult`. */
export interface DeleteDataResult {
  deletion_id: string | null;
  /**
   * `complete`, `pending_purge`, `partial`, `in_progress` or `dry_run`.
   * `pending_purge`: hidden from every read now, restorable with
   * `data.restore` until `purge_after`, then removed for good.
   */
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
  /**
   * Stores this delete does not reach at all, in plain words. `complete` means every
   * store the delete targets was cleared, not that nothing is left in these. Empty
   * from servers that predate the field.
   */
  not_covered: string[];
  note: string | null;
  memories_kept: boolean;
  audit: string | null;
  /** What the SDK cleared on this machine after the server finished. */
  local_cleared: string[];
  /** Only with `instances`: one result per instance, plus `instance_id`. */
  instances: Record<string, unknown>[];
  /** Only with `instances`: instances the time budget did not reach yet. */
  not_started: string[];
  /** Only with `instances`: every audit id. */
  deletion_ids: string[];
  /** With status `pending_purge`: when the deleted memories are purged (ISO 8601, UTC). */
  purge_after: string | null;
  [key: string]: unknown;
}

/** The most ids one `data.restore` call takes. Same as Python's MAX_RESTORE_IDS. */
export const MAX_RESTORE_IDS = 200;

/** Result of `data.restore`. Mirrors Python's `RestoreResult`. */
export interface RestoreResult {
  /** Live again; served by the next fetch. */
  restored_ids: string[];
  /** Not found in this instance, not deleted, or past the purge date. */
  not_restorable_ids: string[];
  /** A store that could not be brought back in line; the memory is restored either way. */
  still_missing_in: string[];
  reindexed_ids: string[];
  fully_restored: boolean;
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
  const instances =
    options.instances === undefined || options.instances === null
      ? null
      : checkInstances(options.instances);
  const permanent = options.permanent ?? false;
  if (typeof permanent !== 'boolean') throw new InvalidInputError('permanent must be a boolean');
  const body: Record<string, unknown> = {
    scope,
    // Without `instances` the body is byte-identical to earlier versions.
    instance_id: instances === null ? instanceId : null,
    user_id: ids['user_id'] ?? null,
    customer_id: ids['customer_id'] ?? null,
    conversation_id: ids['conversation_id'] ?? null,
    dry_run: dryRun,
    confirm,
  };
  if (instances !== null) body['instance_ids'] = instances;
  // Sent only when true, so every other body is byte-identical to older
  // versions (the server refuses unknown fields). Same as Python.
  if (permanent) body['permanent'] = true;
  return body as Json;
}

function checkInstances(instances: unknown): 'all' | string[] {
  if (instances === 'all') return 'all';
  if (!Array.isArray(instances)) {
    throw new InvalidInputError('instances must be "all" or a list of instance ids');
  }
  const out: string[] = [];
  for (const i of instances) {
    if (typeof i !== 'string' || i.trim() === '') {
      throw new InvalidInputError('instances must not contain an empty instance id');
    }
    if (!out.includes(i.trim())) out.push(i.trim());
  }
  if (out.length === 0) {
    throw new InvalidInputError('instances is empty: name at least one instance, or "all"');
  }
  return out;
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
    not_covered: [],
    note: null,
    memories_kept: false,
    audit: null,
    local_cleared: [],
    instances: [],
    not_started: [],
    deletion_ids: [],
    purge_after: null,
    ...raw,
  } as DeleteDataResult;
}

/** What to restore: memory ids, or every memory one `data.delete` hid. */
export interface RestoreOptions {
  memory_ids?: string[] | null;
  memoryIds?: string[] | null;
  /** The `deletion_id` of a `data.delete` result. Send exactly one of the two. */
  deletion_id?: string | null;
  deletionId?: string | null;
}

/**
 * Validate and build the restore body. Throws before any request is sent;
 * messages match Python's `build_restore_body`.
 */
export function buildRestoreBody(
  memoryIds: unknown, instanceId: string, deletionId: unknown = null,
): Json {
  const noIds = memoryIds === null || memoryIds === undefined;
  const noDeletion = deletionId === null || deletionId === undefined;
  if (noIds === noDeletion) {
    throw new InvalidInputError('send exactly one of memory_ids or deletion_id');
  }
  if (!noDeletion) {
    if (typeof deletionId !== 'string' || deletionId.trim() === '') {
      throw new InvalidInputError('deletion_id must not be empty');
    }
    return { deletion_id: deletionId.trim(), instance_id: instanceId };
  }
  if (typeof memoryIds === 'string' || !Array.isArray(memoryIds)) {
    throw new InvalidInputError('memory_ids must be a list of memory ids');
  }
  const ids = memoryIds.map((i) => String(i));
  if (ids.length === 0) throw new InvalidInputError('memory_ids must not be empty');
  if (ids.length > MAX_RESTORE_IDS) {
    throw new InvalidInputError('at most ' + String(MAX_RESTORE_IDS) + ' memory_ids per call');
  }
  if (ids.some((i) => i.trim() === '')) {
    throw new InvalidInputError('memory_ids must not contain empty ids');
  }
  return { memory_ids: ids, instance_id: instanceId };
}

export interface DataInterface {
  /**
   * Delete data for one user, one conversation or one customer. Memories are
   * soft-deleted (restorable until `purge_after`) unless `permanent` is true.
   * scope 'conversation' removes turns, summaries, telemetry and the turn buffer ONLY;
   * memories learned from it are kept (use scope 'user' or memories.delete). Data in
   * another instance is never touched; stores that cannot be scoped by instance are
   * skipped and named in still_present_in.
   *
   * scope 'customer' needs a management key (management:* or *); a runtime key gets 403.
   *
   * A store that could not be cleared is in `still_present_in` and
   * `fully_removed` is false; it is a result, not an exception. Errors:
   * `InvalidInputError` (bad arguments, caught before any request, or 400),
   * `AuthenticationError` (401/403), `ConflictError` (409), `RateLimitError`.
   */
  delete(options: DeleteDataOptions): Promise<DeleteDataResult>;
  // Memories are soft-deleted by default: hidden now, restorable with
  // restore({ deletionId }) until purge_after, then purged; status is
  // 'pending_purge'. Conversations, profile and source text go at once.
  /** The audit record of a delete (status, counts, still_present_in). */
  status(deletionId: string): Promise<DeleteDataResult>;
  /**
   * Bring deleted memories back, until their purge date (the instance's deletion
   * period, 0 to 30 days, 30 by default). Pass up to 200 ids, or
   * `{ deletionId }` for every memory one `data.delete` hid. Needs the same key
   * permissions as delete and reaches only this instance's memories. Ids that
   * could not be restored are in `not_restorable_ids`; that is a result, not an
   * exception.
   */
  restore(target: string[] | RestoreOptions): Promise<RestoreResult>;
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
      if (!dryRun && ['complete', 'pending_purge', 'partial', 'in_progress'].includes(final.status) && onDeleted) {
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

    async restore(target) {
      const instanceId = transport.currentCredentials().instanceId;
      let body: Json;
      if (Array.isArray(target) || typeof target === 'string' || target === null
          || target === undefined) {
        body = buildRestoreBody(target, instanceId);
      } else {
        const o = target as unknown as Record<string, unknown>;
        body = buildRestoreBody(
          either<unknown>(o, 'memory_ids', 'memoryIds') ?? null, instanceId,
          either<unknown>(o, 'deletion_id', 'deletionId') ?? null);
      }
      const raw = await transport.request<Record<string, unknown>>('data_restore', { body });
      return {
        restored_ids: [],
        not_restorable_ids: [],
        still_missing_in: [],
        reindexed_ids: [],
        fully_restored: false,
        ...raw,
      } as RestoreResult;
    },
  };
}
