/** Credits namespace. Mirrors `CreditsInterface` in the Python SDK. */

import { InvalidInputError } from '../errors.js';
import type { HttpTransport } from '../transport/http.js';
import type { Json } from '../context/types.js';

export interface EstimateOptions {
  metric_type?: string;
  metricType?: string;
  units?: number;
  item_count?: number;
  itemCount?: number;
  endpoint?: string;
  mode?: string;
}

export interface CreditsNamespace {
  get_balance(): Promise<CreditBalance>;
  get_ledger(options?: LedgerOptions): Promise<CreditLedgerPage>;
  estimate(options: EstimateOptions): Promise<CreditEstimate>;
  redeem(code: string): Promise<RedeemResult>;
}

/**
 * Return types for the credits namespace, mirroring Python's dataclasses in
 * `maximem_synap/credits.py`. These were all `Json` (`Record<string, unknown>`)
 * before, so every field read was `unknown` and no documented example compiled.
 */
export interface CreditBucket {
  source_type: string;
  balance: number;
  /** ISO-8601, or null when the bucket does not expire. */
  expires_at: string | null;
}

export interface CreditBalance {
  client_id: string;
  balance_credits: number;
  warning_low: boolean;
  buckets: CreditBucket[];
  [key: string]: unknown;
}

export interface CreditLedgerEntry {
  ledger_id: string;
  entry_type: string;
  delta: number;
  metric_type: string | null;
  category: string | null;
  /** ISO-8601. */
  created_at: string;
}

export interface CreditLedgerPage {
  entries: CreditLedgerEntry[];
  total: number;
  limit: number;
  offset: number;
  [key: string]: unknown;
}

export interface CreditEstimate {
  credits_estimate: number;
  [key: string]: unknown;
}

export interface RedeemResult {
  redemption_id: string;
  credits_granted: number;
  new_balance_credits: number;
  /** ISO-8601, or null when the granted credits do not expire. */
  expires_at: string | null;
  [key: string]: unknown;
}

/** Options for `credits.get_ledger`. Mirrors Python's keyword arguments. */
export interface LedgerOptions {
  /** Filter to one ledger entry type. */
  entry_type?: string;
  entryType?: string;
  /** Inclusive lower bound. ISO-8601 string or Date. Sent as `from`. */
  from_time?: string | Date;
  fromTime?: string | Date;
  /** Inclusive upper bound. ISO-8601 string or Date. Sent as `to`. */
  to_time?: string | Date;
  toTime?: string | Date;
  /** Defaults to 100, as in Python. */
  limit?: number;
  offset?: number;
}

function iso(v: string | Date): string {
  return v instanceof Date ? v.toISOString() : v;
}

export function createCreditsNamespace(transport: HttpTransport): CreditsNamespace {
  return {
    async get_balance() { return transport.request<CreditBalance>('credits_balance'); },

    async get_ledger(options: LedgerOptions = {}) {
      // Mirrors Python's CreditsInterface.get_ledger: the three filters were
      // missing here entirely, and the default limit was 50 against Python's
      // 100, so the same call returned different pages in the two SDKs.
      const query: Record<string, string | number> = {
        limit: options.limit ?? 100,
        offset: options.offset ?? 0,
      };
      const entryType = options.entry_type ?? options.entryType;
      if (entryType !== undefined) query['entry_type'] = entryType;
      const from = options.from_time ?? options.fromTime;
      if (from !== undefined) query['from'] = iso(from);
      const to = options.to_time ?? options.toTime;
      if (to !== undefined) query['to'] = iso(to);
      return transport.request<CreditLedgerPage>('credits_ledger', { query });
    },

    async estimate(options) {
      return transport.request<CreditEstimate>('credits_estimate', {
        body: {
          metric_type: options.metric_type ?? options.metricType ?? null,
          units: options.units ?? null,
          item_count: options.item_count ?? options.itemCount ?? null,
          endpoint: options.endpoint ?? null,
          mode: options.mode ?? null,
        },
      });
    },

    async redeem(code) {
      if (!code) throw new InvalidInputError('code is required');
      // Non-idempotent by contract: redeeming twice consumes two codes.
      return transport.request<RedeemResult>('credits_redeem', { body: { code } });
    },
  };
}
