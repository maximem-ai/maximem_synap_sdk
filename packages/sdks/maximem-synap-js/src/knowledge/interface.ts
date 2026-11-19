/**
 * Knowledge namespace. Mirrors `maximem_synap.knowledge.KnowledgeInterface`.
 *
 * `describe` orients: sources, document counts by type, and the instance
 * ontology's entity and fact types with their typed properties and relations.
 * `query` returns complete, filtered sets, a count, or counts grouped by a
 * property, never a ranked top-k. Callers send a structured request, never query
 * text (D165, D167). Every response carries `total`, `complete` and `coverage`.
 *
 * Responses are returned as the server sends them. The routes are behind the
 * `SYNAP_KNOWLEDGE_QUERY` flag (404 while off) and need a Pro plan or above.
 */

import { InvalidInputError } from '../errors.js';
import type { HttpTransport } from '../transport/http.js';
import type { Json } from '../context/types.js';

export const MAX_QUERY_LIMIT = 500;

export interface KnowledgeScope {
  user_id?: string | null;
  userId?: string | null;
  customer_id?: string | null;
  customerId?: string | null;
  /** The scope path, e.g. `{ customer: 'acme', team: 't' }`. */
  scope?: Record<string, string> | null;
}

export interface KnowledgeQueryOptions extends KnowledgeScope {
  /** An entity or fact type from `describe`, e.g. `'Obligation'`. */
  target: string;
  /** Property to a value, or `{ eq|ne|in|before|after|between|gt|lt|contains|exists: ... }`. */
  where?: Record<string, unknown> | null;
  /** Relation to a filter on the node at the other end; nesting is a further hop (at most 2). */
  related?: Record<string, unknown> | null;
  /** An ISO date or datetime; current versions only when unset. */
  as_of?: string | null;
  asOf?: string | null;
  /** Only facts from these documents, as `context.fetch`. */
  document_ids?: string[] | null;
  documentIds?: string[] | null;
  /** `{ count: true }` or `{ group_by: '<property>' }`. */
  aggregate?: Record<string, unknown> | null;
  sort?: Array<{ property: string; order?: 'asc' | 'desc' }> | null;
  /** Rows per page, at most 500 (the server's default is 100). */
  limit?: number | null;
  /** The previous response's `next_cursor`. */
  cursor?: string | null;
}

export interface KnowledgeInterface {
  /** Sources, document counts by type, and the ontology's types, properties and relations. */
  describe(options?: KnowledgeScope): Promise<Json>;
  /** A structured query: filter, traverse (at most 2 hops) or aggregate. */
  query(options: KnowledgeQueryOptions): Promise<Json>;
}

function pick<T>(o: Record<string, unknown>, snake: string, camel: string): T | undefined {
  const a = o[snake];
  if (a !== undefined && a !== null) return a as T;
  const b = o[camel];
  return b === undefined || b === null ? undefined : (b as T);
}

function checkScope(scope: unknown): Record<string, string> | undefined {
  if (scope === undefined || scope === null) return undefined;
  if (typeof scope !== 'object' || Array.isArray(scope)
    || !Object.values(scope as Record<string, unknown>).every((v) => typeof v === 'string')) {
    throw new InvalidInputError('scope must be a flat dict of level key to id, both strings.');
  }
  return scope as Record<string, string>;
}

export function createKnowledgeInterface(transport: HttpTransport): KnowledgeInterface {
  async function checkCustomer(o: Record<string, unknown>, where: string): Promise<void> {
    const customerId = pick<string>(o, 'customer_id', 'customerId');
    if (customerId !== undefined && customerId !== '') await transport.checkCustomerId(customerId, where);
  }

  return {
    async describe(options = {}) {
      const o = options as unknown as Record<string, unknown>;
      await checkCustomer(o, 'knowledge.describe');
      const query: Record<string, string> = {};
      const userId = pick<string>(o, 'user_id', 'userId');
      const customerId = pick<string>(o, 'customer_id', 'customerId');
      if (userId !== undefined) query['user_id'] = userId;
      if (customerId !== undefined) query['customer_id'] = customerId;
      const scope = checkScope(o['scope']);
      // A query string cannot carry a map, so it travels as JSON (Python: `_scope_params`).
      if (scope !== undefined) query['scope'] = JSON.stringify(scope);
      return transport.request<Json>('knowledge_describe', { query });
    },

    async query(options) {
      const o = options as unknown as Record<string, unknown>;
      await checkCustomer(o, 'knowledge.query');
      const target = o['target'];
      if (typeof target !== 'string' || target === '') {
        throw new InvalidInputError('target must be a non-empty string');
      }
      const limit = o['limit'];
      if (limit !== undefined && limit !== null
        && (typeof limit !== 'number' || !Number.isInteger(limit) || limit < 1 || limit > MAX_QUERY_LIMIT)) {
        throw new InvalidInputError(`limit must be a whole number from 1 to ${MAX_QUERY_LIMIT}`);
      }
      const body: Record<string, unknown> = { target };
      const fields: Array<[string, unknown]> = [
        ['where', o['where']], ['related', o['related']],
        ['as_of', pick(o, 'as_of', 'asOf')],
        ['document_ids', pick(o, 'document_ids', 'documentIds')],
        ['aggregate', o['aggregate']], ['sort', o['sort']], ['limit', limit], ['cursor', o['cursor']],
      ];
      for (const [key, value] of fields) {
        if (value !== undefined && value !== null) body[key] = value;
      }
      const userId = pick<string>(o, 'user_id', 'userId');
      const customerId = pick<string>(o, 'customer_id', 'customerId');
      if (userId !== undefined) body['user_id'] = userId;
      if (customerId !== undefined) body['customer_id'] = customerId;
      const scope = checkScope(o['scope']);
      if (scope !== undefined) body['scope'] = scope;
      return transport.request<Json>('knowledge_query', { body });
    },
  };
}
