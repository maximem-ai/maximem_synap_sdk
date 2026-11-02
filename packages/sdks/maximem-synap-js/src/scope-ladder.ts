/**
 * The account's scope ladder, as the SDK sees it. Mirrors
 * `maximem_synap/scope_ladder.py`; if you change a rule here, change it there
 * in the same piece of work.
 *
 * A ladder is the ordered list of levels an account files memory under. The
 * default has two non-root levels, `customer` then `user` (B2C: `user` alone).
 * An account may add levels in between, for example `team`.
 *
 * This module turns "values, outermost first" into the arguments the SDK calls
 * take (`user_id`, `customer_id` and, on a deeper ladder, `scope_path` for
 * reads or `scope` for memories), and refuses before any network call when the
 * values cannot be addressed.
 *
 * The rules, each read from the server (`synap/cloud/shared/scope_ladder.py`
 * `_normalise`, `synap/cloud/shared/scope.py` `resolve_identity`):
 *
 *  - A path never names the root level, must be contiguous from the first
 *    level, and a shorter path (a prefix) means that level.
 *  - The server reads the path keys `customer` and `user` over the flat ids,
 *    so ids are always sent as well.
 *  - A path is honoured only when nested scoping is on for the instance AND the
 *    account's ladder is ratified; otherwise the server refuses it. So on a
 *    default ladder this module never produces a path.
 *  - B2C (`user_context_isolation == "equals_customer"`): `resolve_identity`
 *    collapses customer and user onto the user's id, and `customer_id` is
 *    refused by the SDK. The tenant level of a path therefore carries the
 *    identity's value, and only `user_id` is sent.
 */

import { ScopeLadderError } from './errors.js';

export const ROLE_TENANT = 'tenant_of_record';
export const ROLE_IDENTITY = 'identity';

/** How long a ladder is trusted before whoami is read again. As the server's cache. */
export const LADDER_TTL_MS = 300_000;
/** After a failed re-read, how long to keep serving the last known ladder. */
export const LADDER_RETRY_BACKOFF_MS = 60_000;

/**
 * Minimum time between re-reads that an invalidation triggers. A caller whose
 * every request is refused would otherwise read whoami once per request.
 */
export const LADDER_INVALIDATION_MIN_INTERVAL_MS = 5_000;

export interface ScopeLevel {
  readonly key: string;
  readonly label: string;
  readonly position: number;
  readonly role: string | null;
  readonly required: boolean;
}

/** Where a call reads or writes: ids, plus a path on a deeper ladder. `scope_path` is null on a default ladder. */
export class ScopeAddress {
  readonly user_id: string | null;
  readonly customer_id: string | null;
  readonly scope_path: Readonly<Record<string, string>> | null;

  constructor(init: {
    user_id?: string | null;
    customer_id?: string | null;
    scope_path?: Record<string, string> | null;
  } = {}) {
    this.user_id = init.user_id ?? null;
    this.customer_id = init.customer_id ?? null;
    this.scope_path = init.scope_path ?? null;
  }

  /** Options for reads (`fetch`, `*.context.fetch`, `send_message`, ...): the path is `scope_path`. */
  read_kwargs(): { user_id?: string; customer_id?: string; scope_path?: Record<string, string> } {
    const out: { user_id?: string; customer_id?: string; scope_path?: Record<string, string> } = {};
    if (this.user_id) out.user_id = this.user_id;
    if (this.customer_id) out.customer_id = this.customer_id;
    if (this.scope_path && Object.keys(this.scope_path).length > 0) {
      out.scope_path = { ...this.scope_path };
    }
    return out;
  }

  /** Options for `memories.create` / `create_from_file` / `forget`: the path is `scope`. */
  write_kwargs(): { user_id?: string; customer_id?: string; scope?: Record<string, string> } {
    const out: { user_id?: string; customer_id?: string; scope?: Record<string, string> } = {};
    if (this.user_id) out.user_id = this.user_id;
    if (this.customer_id) out.customer_id = this.customer_id;
    if (this.scope_path && Object.keys(this.scope_path).length > 0) {
      out.scope = { ...this.scope_path };
    }
    return out;
  }
}

const REFRESH_HINT =
  "The account's ladder may not be accepted yet, or nested scoping may be off for " +
  'this instance. If that changed recently, ' +
  '`await sdk.get_scope_ladder({ refresh: true })` re-reads it.';

export class ScopeLadder {
  readonly levels: readonly ScopeLevel[];
  readonly is_default: boolean;
  readonly ladder_ratified: boolean;
  readonly nested_scoping_enabled: boolean;
  /** "server" | "inferred" */
  readonly source: 'server' | 'inferred';
  /** True when the instance is B2C: the tenant and identity level are one node. */
  readonly b2c: boolean;

  constructor(init: {
    levels: readonly ScopeLevel[];
    is_default: boolean;
    ladder_ratified: boolean;
    nested_scoping_enabled: boolean;
    source: 'server' | 'inferred';
    b2c?: boolean;
  }) {
    this.levels = Object.freeze([...init.levels]);
    this.is_default = init.is_default;
    this.ladder_ratified = init.ladder_ratified;
    this.nested_scoping_enabled = init.nested_scoping_enabled;
    this.source = init.source;
    this.b2c = init.b2c ?? false;
  }

  get keys(): readonly string[] {
    return this.levels.map((l) => l.key);
  }

  /**
   * Map `values` (outermost level first) to ids and, on a deeper ladder, the
   * full prefix path. A shorter list addresses a higher level. Throws
   * `ScopeLadderError` before any call when the values cannot be addressed.
   */
  address(values: readonly string[]): ScopeAddress {
    if (!Array.isArray(values)) {
      throw new ScopeLadderError(
        `address() takes a list of values, outermost level first, got ${typeof values}.`,
      );
    }
    const vals = [...values];
    if (vals.length === 0) {
      throw new ScopeLadderError(
        'address() needs at least one value. The client level is addressed ' +
          "explicitly elsewhere (scope='client'), not through the ladder. " +
          `Levels, outermost first: ${JSON.stringify(this.keys)}.`,
      );
    }
    for (const v of vals) {
      if (typeof v !== 'string' || v === '') {
        throw new ScopeLadderError(`address() values must be non-empty strings, got ${JSON.stringify(v)}.`);
      }
    }

    const levels = this.levels;
    // B2C: the tenant level takes the identity's value, so the caller gives one
    // value per level except the tenant level.
    const expected = this.b2c ? levels.filter((l) => l.role !== ROLE_TENANT) : levels;
    if (vals.length > expected.length) {
      throw new ScopeLadderError(
        `${vals.length} values for ${expected.length} levels ` +
          `(${JSON.stringify(expected.map((l) => l.key))}, outermost first).`,
      );
    }

    if (!this.is_default && (!this.nested_scoping_enabled || !this.ladder_ratified)) {
      throw new ScopeLadderError(
        `This account has a deeper scope ladder (${JSON.stringify(this.keys)}), but the server ` +
          'would refuse a scope path ' +
          `(nested_scoping_enabled=${this.nested_scoping_enabled}, ` +
          `ladder_ratified=${this.ladder_ratified}). ` + REFRESH_HINT,
      );
    }

    let userId: string | null = null;
    let customerId: string | null = null;
    const path: Record<string, string> = {};

    if (this.b2c) {
      const identity = levels.find((l) => l.role === ROLE_IDENTITY);
      if (identity === undefined) {
        throw new ScopeLadderError('This ladder has no identity level, so a B2C address cannot be built.');
      }
      if (vals.length < expected.length) {
        // The tenant level carries the identity's value, which a shorter list
        // does not reach. Guessing would file under the wrong node.
        throw new ScopeLadderError(
          "On a B2C instance the tenant level takes the identity's value, so give one value " +
            `for every level after it (${JSON.stringify(expected.map((l) => l.key))}); got ${vals.length}.`,
        );
      }
      const byKey = new Map<string, string>();
      expected.forEach((l, i) => byKey.set(l.key, vals[i] as string));
      const identityValue = byKey.get(identity.key) as string;
      userId = identityValue;
      if (!this.is_default) {
        for (const l of levels) {
          path[l.key] = l.role === ROLE_TENANT ? identityValue : (byKey.get(l.key) as string);
        }
      }
    } else {
      levels.slice(0, vals.length).forEach((l, i) => {
        const v = vals[i] as string;
        if (l.role === ROLE_TENANT) customerId = v;
        else if (l.role === ROLE_IDENTITY) userId = v;
        if (!this.is_default) path[l.key] = v;
      });
    }

    return new ScopeAddress({
      user_id: userId,
      customer_id: customerId,
      scope_path: Object.keys(path).length > 0 ? path : null,
    });
  }
}

const STRICT_LEVELS: readonly ScopeLevel[] = [
  { key: 'customer', label: 'Customer', position: 1, role: ROLE_TENANT, required: true },
  { key: 'user', label: 'User', position: 2, role: ROLE_IDENTITY, required: true },
];
const B2C_LEVELS: readonly ScopeLevel[] = [
  { key: 'user', label: 'User', position: 1, role: ROLE_IDENTITY, required: true },
];

function parseLevels(raw: unknown): ScopeLevel[] | null {
  if (!Array.isArray(raw) || raw.length === 0) return null;
  const out: ScopeLevel[] = [];
  for (const item of raw) {
    if (typeof item !== 'object' || item === null || Array.isArray(item)) return null;
    const o = item as Record<string, unknown>;
    const key = o['key'];
    if (typeof key !== 'string' || key === '') return null;
    const position = Number(o['position']);
    if (!Number.isFinite(position) || o['position'] === null || o['position'] === '') return null;
    const label = o['label'];
    const role = o['role'];
    out.push({
      key,
      label: typeof label === 'string' && label !== '' ? label : key,
      position: Math.trunc(position),
      role: typeof role === 'string' && role !== '' ? role : null,
      required: o['required'] === undefined ? true : Boolean(o['required']),
    });
  }
  out.sort((a, b) => a.position - b.position);
  return out;
}

/**
 * The ladder a whoami answer describes, or null when it does not say.
 *
 * Uses the `scope_ladder` block when present and well formed. A missing key and
 * an explicit null both mean "no block"; anything else that is not a well
 * formed block is treated as absent with a warning, never an exception. With no
 * usable block the default ladder is inferred from `user_context_isolation`
 * (`strict`: customer, user; `equals_customer`: user), `source: "inferred"`.
 */
export function ladderFromWhoami(
  whoami: Record<string, unknown>,
  fallbackIsolation?: string,
  warn: (message: string) => void = () => {},
): ScopeLadder | null {
  const rawIso = whoami['user_context_isolation'];
  const isolation = (typeof rawIso === 'string' && rawIso !== '' ? rawIso : undefined) ?? fallbackIsolation;
  const b2c = isolation === 'equals_customer';

  const block = whoami['scope_ladder'];
  if (block !== undefined && block !== null) {
    const levels =
      typeof block === 'object' && !Array.isArray(block)
        ? parseLevels((block as Record<string, unknown>)['levels'])
        : null;
    if (levels === null) {
      warn('whoami returned a malformed scope_ladder block; ignoring it and inferring the default ladder.');
    } else {
      const b = block as Record<string, unknown>;
      return new ScopeLadder({
        levels,
        is_default: Boolean(b['is_default']),
        ladder_ratified: Boolean(b['ladder_ratified']),
        nested_scoping_enabled: Boolean(b['nested_scoping_enabled']),
        source: 'server',
        b2c,
      });
    }
  }

  if (isolation === undefined) return null;
  return new ScopeLadder({
    levels: b2c ? B2C_LEVELS : STRICT_LEVELS,
    is_default: true,
    ladder_ratified: false,
    nested_scoping_enabled: false,
    source: 'inferred',
    b2c,
  });
}
