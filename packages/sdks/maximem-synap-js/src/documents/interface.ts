/**
 * Documents namespace. Mirrors `maximem_synap.documents.DocumentsInterface`.
 *
 * A document is addressed by its `document_key`, the server's stable address
 * returned by `memories.create` and carried in every citation. A caller who
 * kept only its own `document_id` passes it with the same `user_id`,
 * `customer_id` or `scope` it was created with; the SDK then resolves it
 * through `documents_lookup` first, because a `document_id` is unique only
 * within a scope (T12). When it names more than one document the lookup
 * answers 409, thrown as `AmbiguousDocumentError` listing each match's scope.
 *
 * Responses are returned as the server sends them, except `original`, which
 * returns the file's bytes.
 */

import { InvalidInputError } from '../errors.js';
import type { HttpTransport } from '../transport/http.js';
import type { Json } from '../context/types.js';

/** One document: by its key, or by the caller's id and the scope it was created at. */
export interface DocumentAddress {
  document_key?: string | null;
  documentKey?: string | null;
  document_id?: string | null;
  documentId?: string | null;
  user_id?: string | null;
  userId?: string | null;
  customer_id?: string | null;
  customerId?: string | null;
  /** The scope path it was created with, e.g. `{ customer: 'acme', team: 't' }`. */
  scope?: Record<string, string> | null;
}

export interface ListDocumentsOptions {
  document_type?: string | null;
  documentType?: string | null;
  source_system?: string | null;
  sourceSystem?: string | null;
  user_id?: string | null;
  userId?: string | null;
  customer_id?: string | null;
  customerId?: string | null;
  scope?: Record<string, string> | null;
  cursor?: string | null;
}

export interface OriginalOptions extends DocumentAddress {
  version: number;
}

export interface DeleteDocumentOptions extends DocumentAddress {
  /**
   * Delete at once, with no undo. Needs a key with management access (D178).
   * Otherwise the delete is soft: hidden at once, restorable until the
   * instance's purge date (D177).
   */
  permanent?: boolean;
}

export interface IndexOptions extends DocumentAddress {
  /** A version other than the current one. */
  version?: number | null;
}

export interface ReadOptions extends DocumentAddress {
  /** A section of the index; exactly one of `sectionKey` and `pages`. */
  section_key?: string | null;
  sectionKey?: string | null;
  /** `'3-5'` or `[3, 5]`; at most 20 pages or 8,000 tokens per call. */
  pages?: string | [number, number] | null;
}

export interface DocumentsInterface {
  /** Resolve a scope-qualified `document_id` to its document and `document_key`. */
  lookup(options: DocumentAddress): Promise<Json>;
  /** One page of documents; only the filters given are sent. */
  list(options?: ListDocumentsOptions): Promise<Json>;
  /** The document's current version, status and `processing_stage`. */
  get(address: string | DocumentAddress): Promise<Json>;
  /** Every version of the document (D91). */
  versions(address: string | DocumentAddress): Promise<Json>;
  /** One version's original file, as bytes. 404 when the document is not visible (D93). */
  original(options: OriginalOptions): Promise<Uint8Array>;
  /** Delete the document and everything made from it (D63). */
  delete(address: string | DeleteDocumentOptions): Promise<Json>;
  /** Undo a soft delete before the purge date (D178). */
  restore(address: string | DocumentAddress): Promise<Json>;
  /**
   * Title, summary and section tree of the current version, or of `version` (D166).
   * The scope ids are the ones the document is read at, as in `context.fetch`, also
   * with a `document_key` (D175). 404 when the document is not visible (D93) or
   * `SYNAP_DOCS_NAVIGATION` is off; Pro plan and above.
   */
  index(address: string | IndexOptions): Promise<Json>;
  /**
   * One section or one page range of the current version, with its `citation`
   * (D166). 422 `read_too_large` past 20 pages or 8,000 tokens; 404
   * `document_not_ready` while the document is processing.
   */
  read(options: ReadOptions): Promise<Json>;
}

const PAGES = /^(\d+)-(\d+)$/;

function pick<T>(o: Record<string, unknown>, snake: string, camel: string): T | undefined {
  const a = o[snake];
  if (a !== undefined && a !== null) return a as T;
  const b = o[camel];
  return b === undefined || b === null ? undefined : (b as T);
}

/** The scope a document was created at, as query parameters. Python: `_scope_params`. */
function scopeParams(o: Record<string, unknown>): Record<string, string> {
  const params: Record<string, string> = {};
  const userId = pick<string>(o, 'user_id', 'userId');
  const customerId = pick<string>(o, 'customer_id', 'customerId');
  if (userId !== undefined) params['user_id'] = userId;
  if (customerId !== undefined) params['customer_id'] = customerId;
  const scope = o['scope'];
  if (scope !== undefined && scope !== null) {
    if (typeof scope !== 'object' || Array.isArray(scope)
      || !Object.values(scope as Record<string, unknown>).every((v) => typeof v === 'string')) {
      throw new InvalidInputError('scope must be a flat dict of level key to id, both strings.');
    }
    // A query string cannot carry a map, so it travels as JSON, as
    // create_from_file sends it on its multipart route.
    params['scope'] = JSON.stringify(scope);
  }
  return params;
}

export function createDocumentsInterface(transport: HttpTransport): DocumentsInterface {
  async function checkCustomer(o: Record<string, unknown>, where: string): Promise<void> {
    const customerId = pick<string>(o, 'customer_id', 'customerId');
    if (customerId !== undefined && customerId !== '') await transport.checkCustomerId(customerId, where);
  }

  async function lookup(options: DocumentAddress): Promise<Json> {
    const o = options as unknown as Record<string, unknown>;
    const documentId = pick<string>(o, 'document_id', 'documentId');
    if (typeof documentId !== 'string' || documentId === '') {
      throw new InvalidInputError('document_id must be a non-empty string');
    }
    // On B2C a customer_id is refused before any request, as Python does.
    await checkCustomer(o, 'documents.lookup');
    return transport.request<Json>('documents_lookup', {
      query: { document_id: documentId, ...scopeParams(o) },
    });
  }

  /** The `document_key` to address, looking a caller id up when needed. Python: `_key`. */
  async function keyOf(address: string | DocumentAddress): Promise<string> {
    const o = (typeof address === 'string' ? { document_key: address } : address ?? {}) as
      Record<string, unknown>;
    const key = pick<string>(o, 'document_key', 'documentKey');
    const documentId = pick<string>(o, 'document_id', 'documentId');
    if ((key === undefined) === (documentId === undefined)) {
      throw new InvalidInputError('Pass exactly one of document_key or document_id');
    }
    if (key !== undefined) {
      if (typeof key !== 'string' || key === '') {
        throw new InvalidInputError('document_key must be a non-empty string');
      }
      if (Object.keys(scopeParams(o)).length > 0) {
        throw new InvalidInputError(
          'user_id, customer_id and scope resolve a document_id; ' +
            'a document_key already names one document',
        );
      }
      return key;
    }
    const found = await lookup(o as DocumentAddress);
    const resolved = (found as Record<string, unknown> | undefined)?.['document_key'];
    if (typeof resolved !== 'string' || resolved === '') {
      throw new InvalidInputError('documents_lookup returned no document_key');
    }
    return resolved;
  }

  /** The key for `index` and `read`, which send the scope with a key too (D175). */
  async function navKeyOf(o: Record<string, unknown>): Promise<string> {
    const key = pick<string>(o, 'document_key', 'documentKey');
    const documentId = pick<string>(o, 'document_id', 'documentId');
    if ((key === undefined) === (documentId === undefined)) {
      throw new InvalidInputError('Pass exactly one of document_key or document_id');
    }
    if (key !== undefined) {
      if (typeof key !== 'string' || key === '') {
        throw new InvalidInputError('document_key must be a non-empty string');
      }
      return key;
    }
    return keyOf(o as DocumentAddress);
  }

  return {
    lookup,

    async list(options = {}) {
      const o = options as unknown as Record<string, unknown>;
      await checkCustomer(o, 'documents.list');
      const query: Record<string, string> = {};
      const documentType = pick<string>(o, 'document_type', 'documentType');
      const sourceSystem = pick<string>(o, 'source_system', 'sourceSystem');
      if (documentType !== undefined) query['document_type'] = documentType;
      if (sourceSystem !== undefined) query['source_system'] = sourceSystem;
      Object.assign(query, scopeParams(o));
      if (options.cursor !== undefined && options.cursor !== null) query['cursor'] = options.cursor;
      return transport.request<Json>('documents_list', { query });
    },

    async get(address) {
      const key = await keyOf(address);
      return transport.request<Json>('documents_get', { pathParams: { document_key: key } });
    },

    async versions(address) {
      const key = await keyOf(address);
      return transport.request<Json>('documents_versions', { pathParams: { document_key: key } });
    },

    async original(options) {
      const version = options?.version;
      if (typeof version !== 'number' || !Number.isInteger(version) || version < 1) {
        throw new InvalidInputError('version must be a positive integer');
      }
      const { version: _v, ...address } = options;
      const key = await keyOf(address);
      return transport.request<Uint8Array>('documents_original', {
        pathParams: { document_key: key, version },
        responseType: 'bytes',
      });
    },

    async delete(address) {
      const o = (typeof address === 'string' ? { document_key: address } : address ?? {}) as
        DeleteDocumentOptions;
      const permanent = o.permanent ?? false;
      if (typeof permanent !== 'boolean') throw new InvalidInputError('permanent must be True or False');
      const { permanent: _p, ...rest } = o;
      const key = await keyOf(rest);
      return transport.request<Json>('documents_delete', {
        pathParams: { document_key: key },
        ...(permanent ? { query: { permanent: 'true' } } : {}),
      });
    },

    async restore(address) {
      const key = await keyOf(address);
      return transport.request<Json>('documents_restore', { pathParams: { document_key: key } });
    },

    async index(address) {
      const o = (typeof address === 'string' ? { document_key: address } : address ?? {}) as
        Record<string, unknown>;
      await checkCustomer(o, 'documents.index');
      const version = o['version'];
      if (version !== undefined && version !== null
        && (typeof version !== 'number' || !Number.isInteger(version) || version < 1)) {
        throw new InvalidInputError('version must be a positive integer');
      }
      const query: Record<string, string> = {};
      if (version !== undefined && version !== null) query['version'] = String(version);
      Object.assign(query, scopeParams(o));
      const key = await navKeyOf(o);
      return transport.request<Json>('documents_index', {
        pathParams: { document_key: key }, query,
      });
    },

    async read(options) {
      const o = (options ?? {}) as unknown as Record<string, unknown>;
      await checkCustomer(o, 'documents.read');
      const sectionKey = pick<string>(o, 'section_key', 'sectionKey');
      let pages = o['pages'] as string | [number, number] | null | undefined;
      if (pages === null) pages = undefined;
      if ((sectionKey === undefined) === (pages === undefined)) {
        throw new InvalidInputError('Pass exactly one of section_key or pages');
      }
      const query: Record<string, string> = {};
      if (sectionKey !== undefined) {
        if (typeof sectionKey !== 'string' || sectionKey === '') {
          throw new InvalidInputError('section_key must be a non-empty string');
        }
        query['section_key'] = sectionKey;
      } else {
        if (Array.isArray(pages)) {
          if (pages.length !== 2 || !pages.every((n) => Number.isInteger(n))) {
            throw new InvalidInputError('pages is (from, to), two whole page numbers');
          }
          pages = `${pages[0]}-${pages[1]}`;
        }
        if (typeof pages !== 'string') {
          throw new InvalidInputError("pages is 'from-to' or a (from, to) pair");
        }
        const m = PAGES.exec(pages.trim());
        if (m === null || Number(m[1]) < 1 || Number(m[2]) < Number(m[1])) {
          throw new InvalidInputError('pages is from-to, whole page numbers from 1, e.g. 3-5');
        }
        query['pages'] = pages.trim();
      }
      Object.assign(query, scopeParams(o));
      const key = await navKeyOf(o);
      return transport.request<Json>('documents_read', {
        pathParams: { document_key: key }, query,
      });
    },
  };
}
