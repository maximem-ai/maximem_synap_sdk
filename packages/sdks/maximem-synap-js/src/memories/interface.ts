/**
 * Memories namespace. Mirrors `maximem_synap.memories.interface.MemoriesInterface`.
 *
 * ## The request body is built the way Python builds it, on purpose
 *
 * Python constructs a pydantic `CreateMemoryRequest` and posts
 * `model_dump(mode="json")`, which emits **every** field, nulls included, with
 * defaults applied. It is tempting to omit unset keys instead and send a
 * tidier body. Do not: the server sees a different request, and the difference
 * that matters most is `document_type`.
 *
 * `document_type` defaults to `"ai-chat-conversation"`. Leaving it off is not
 * neutral, because the extraction path is type-dependent and the
 * `"document"` type extracts dramatically less from the same text. A body that
 * merely looks cleaner is how you get "the JS SDK stores fewer memories than
 * the Python one from identical input", which is slow and miserable to
 * diagnose.
 */

import { InvalidInputError } from '../errors.js';
import type { HttpTransport } from '../transport/http.js';
import type { Json } from '../context/types.js';

/**
 * What `new Blob([...])` accepts here.
 *
 * Derived from the Blob constructor rather than written out, because `BlobPart`
 * lives in the DOM lib, which this package deliberately does not include, and
 * Node's own Blob accepts a slightly narrower set than the DOM's. Deriving it
 * means this keeps compiling if either definition shifts.
 */
type BlobSource = NonNullable<ConstructorParameters<typeof Blob>[0]>[number];

/** Mirrors the Python DocumentType enum. */
export const DOCUMENT_TYPES = [
  'ai-chat-conversation', 'document', 'email', 'pdf', 'image', 'audio', 'meeting-transcript',
] as const;
export type DocumentType = (typeof DOCUMENT_TYPES)[number];

/** Mirrors the Python IngestMode enum. */
export const INGEST_MODES = ['fast', 'long-range'] as const;
export type IngestMode = (typeof INGEST_MODES)[number];

export const DEFAULT_DOCUMENT_TYPE: DocumentType = 'ai-chat-conversation';
export const DEFAULT_INGEST_MODE: IngestMode = 'long-range';

export interface CreateMemoryOptions {
  document: string;
  user_id?: string | null;
  userId?: string | null;
  customer_id?: string | null;
  customerId?: string | null;
  document_type?: DocumentType | string;
  documentType?: DocumentType | string;
  document_id?: string | null;
  documentId?: string | null;
  /** ISO-8601 string or Date. Serialised the way Python serialises a datetime. */
  document_created_at?: string | Date | null;
  documentCreatedAt?: string | Date | null;
  mode?: IngestMode | string;
  metadata?: Json;
  /**
   * The level this memory belongs to, named in full, one entry per level down
   * to it: `{ customer: "acme", team: "payments", user: "dana" }`.
   *
   * Only needed by an account whose scope ladder has a level between its
   * customer and its user levels. `user_id` and `customer_id` cannot say which
   * one a write belongs to, so on such a ladder the server refuses the write
   * rather than guessing: a wrong guess at or above the customer level crosses
   * a tenant boundary.
   *
   * Omit it and nothing changes. Sending one to an account without nested
   * scoping is an error rather than ignored, because a caller who asks to be
   * narrowed and is ignored gets widened instead.
   */
  scope?: Record<string, string> | null;
}

/**
 * Mirrors Python's `CreateMemoryResponse`.
 *
 * The first four fields are required there (Pydantic would raise if the server
 * omitted one), so they are required here too. Declaring `ingestion_id` optional
 * made the commonest two-line pattern in the docs fail to compile under strict
 * TypeScript: `wait_for_completion(result.ingestion_id)` cannot take
 * `string | undefined`.
 */
export interface CreateMemoryResult {
  ingestion_id: string;
  document_id: string;
  status: string;
  /** ISO-8601. Python types this as a datetime. */
  queued_at: string;
  error_message?: string | null;
  memories_created?: number;
  [key: string]: unknown;
}

function either<T>(o: Record<string, unknown>, snake: string, camel: string): T | undefined {
  const a = o[snake];
  if (a !== undefined) return a as T;
  const b = o[camel];
  return b === undefined ? undefined : (b as T);
}

function isoOrNull(v: string | Date | null | undefined): string | null {
  if (v === undefined || v === null) return null;
  return v instanceof Date ? v.toISOString() : v;
}

/** Byte-for-byte equivalent of Python's `CreateMemoryRequest.model_dump(mode="json")`. */
export function buildCreateBody(options: CreateMemoryOptions): Json {
  if (!options.document) throw new InvalidInputError('document is required');

  const o = options as unknown as Record<string, unknown>;
  const documentType = either<string>(o, 'document_type', 'documentType') ?? DEFAULT_DOCUMENT_TYPE;
  const mode = (options.mode as string) ?? DEFAULT_INGEST_MODE;

  // Validated client-side because Python's enum coercion raises before any
  // request is sent. A silent pass-through would turn a typo into a server
  // error, or worse, a silently different extraction path.
  if (!(DOCUMENT_TYPES as readonly string[]).includes(documentType)) {
    throw new InvalidInputError(
      `Invalid document_type '${documentType}'. Expected one of: ${DOCUMENT_TYPES.join(', ')}`,
    );
  }
  if (!(INGEST_MODES as readonly string[]).includes(mode)) {
    throw new InvalidInputError(
      `Invalid mode '${mode}'. Expected one of: ${INGEST_MODES.join(', ')}`,
    );
  }

  // Key order matches the pydantic model's field order. Not semantically
  // required, but it makes a captured-body diff against Python readable.
  return {
    document: options.document,
    document_type: documentType,
    document_id: either<string>(o, 'document_id', 'documentId') ?? null,
    document_created_at: isoOrNull(
      either<string | Date>(o, 'document_created_at', 'documentCreatedAt'),
    ),
    user_id: either<string>(o, 'user_id', 'userId') ?? null,
    customer_id: either<string>(o, 'customer_id', 'customerId') ?? null,
    // null, never `{}`. An empty path is not absence: the server treats one as
    // an error, because a caller asking to be narrowed and being ignored gets
    // widened instead.
    scope: options.scope ?? null,
    mode,
    metadata: options.metadata ?? {},
  };
}

/**
 * Result of `memories.batch_create`, mirroring Python's `BatchCreateResponse`.
 * It was `Json`, so `results` came back as `unknown` and could not be iterated.
 */
export interface BatchCreateResult {
  batch_id: string;
  total: number;
  succeeded: number;
  failed: number;
  results: CreateMemoryResult[];
  [key: string]: unknown;
}

export interface BatchCreateOptions {
  documents: CreateMemoryOptions[];
  fail_fast?: boolean;
  failFast?: boolean;
}

export interface MemoriesInterface {
  create(options: CreateMemoryOptions): Promise<CreateMemoryResult>;
  batch_create(options: BatchCreateOptions): Promise<BatchCreateResult>;
  create_from_file(options: CreateFromFileOptions): Promise<CreateMemoryResult>;
  get(memoryId: string): Promise<Json>;
  update(options: UpdateMemoryOptions): Promise<Json>;
  delete(memoryId: string): Promise<Json>;
  status(ingestionId: string): Promise<Json>;
  wait_for_completion(ingestionId: string, options?: WaitOptions): Promise<Json>;
}

/**
 * Options for `create_from_file`.
 *
 * Exactly one of `file_path`, `file`, or `text` must be supplied, matching
 * Python. `file_path` reads from disk and so is Node-only; `file` takes a
 * Blob/File/Uint8Array and works anywhere, which is the portable route for
 * Edge and Workers.
 */
export interface CreateFromFileOptions {
  user_id?: string;
  userId?: string;
  customer_id?: string;
  customerId?: string;
  /** "b2b" or "b2c". Defaults to "b2c", as in Python. */
  relationship_type?: string;
  relationshipType?: string;
  /** Node-only: read this path from disk. Uses a lazy `node:fs` import. */
  file_path?: string;
  filePath?: string;
  /** Portable alternative to `file_path`. Pass `filename` alongside a raw buffer. */
  file?: Blob | Uint8Array | ArrayBuffer;
  filename?: string;
  /** Ingest raw text instead of a file. */
  text?: string;
  document_type?: string;
  documentType?: string;
  mode?: IngestMode | string;
  metadata?: Json;
  /**
   * The level this memory belongs to, named in full, one entry per level down
   * to it: `{ customer: "acme", team: "payments", user: "dana" }`.
   *
   * Only needed by an account whose scope ladder has a level between its
   * customer and its user levels. `user_id` and `customer_id` cannot say which
   * one a write belongs to, so on such a ladder the server refuses the write
   * rather than guessing: a wrong guess at or above the customer level crosses
   * a tenant boundary.
   *
   * Omit it and nothing changes. Sending one to an account without nested
   * scoping is an error rather than ignored, because a caller who asks to be
   * narrowed and is ignored gets widened instead.
   */
  scope?: Record<string, string> | null;
}

export interface UpdateMemoryOptions {
  memory_id?: string;
  memoryId?: string;
  document: string;
  merge_strategy?: string;
  mergeStrategy?: string;
  document_type?: string;
  documentType?: string;
  metadata?: Json;
}

export interface WaitOptions {
  /** Seconds. Matches Python's default. */
  timeout_seconds?: number;
  timeoutSeconds?: number;
  poll_interval_seconds?: number;
  pollIntervalSeconds?: number;
  signal?: AbortSignal;
}

const TERMINAL_STATUSES = new Set(['completed', 'failed', 'error', 'cancelled']);

export function createMemoriesInterface(transport: HttpTransport): MemoriesInterface {
  return {
    async create(options) {
      const o = options as unknown as Record<string, unknown>;
      transport.checkCustomerId(o['customer_id'] ?? o['customerId'], 'memories.create');
      return transport.request<CreateMemoryResult>('memories_create', { body: buildCreateBody(options) });
    },

    async batch_create(options) {
      const docs = options.documents;
      if (!Array.isArray(docs) || docs.length === 0) {
        throw new InvalidInputError('documents must be a non-empty array');
      }
      const failFast = options.fail_fast ?? options.failFast ?? false;
      return transport.request<BatchCreateResult>('memories_batch', {
        body: { documents: docs.map(buildCreateBody), fail_fast: failFast },
      });
    },

    async create_from_file(options) {
      const userId = options.user_id ?? options.userId;
      const customerId = options.customer_id ?? options.customerId;
      if (!userId) throw new InvalidInputError('user_id is required');
      // Not required: rejected on B2C, so requiring it made the documented
      // default combination impossible to call.
      transport.checkCustomerId(customerId, 'memories.create_from_file');

      const filePath = options.file_path ?? options.filePath;
      const sources = [filePath, options.file, options.text].filter((v) => v !== undefined);
      if (sources.length === 0) {
        // Python raises ValueError here. There is no JS equivalent, and every
        // other validation in this SDK raises InvalidInputError, so the
        // taxonomy wins over a literal transcription of the exception type.
        throw new InvalidInputError('One of file_path, file, or text must be provided.');
      }
      if (sources.length > 1) {
        throw new InvalidInputError(
          'Provide exactly one of file_path, file, or text, not several.',
        );
      }

      const form = new FormData();
      form.set('user_id', userId);
      form.set('customer_id', customerId);
      form.set('relationship_type', options.relationship_type ?? options.relationshipType ?? 'b2c');
      form.set('mode', String(options.mode ?? DEFAULT_INGEST_MODE));

      // Python appends these only when truthy, so an empty document_type or an
      // empty metadata object is absent from the form rather than present-empty.
      const documentType = options.document_type ?? options.documentType;
      if (documentType) form.set('document_type', String(documentType));
      if (options.metadata !== undefined && Object.keys(options.metadata).length > 0) {
        form.set('metadata', JSON.stringify(options.metadata));
      }

      if (filePath !== undefined) {
        // Lazy, so importing the SDK on Edge or in a Worker stays safe: a
        // top-level `node:fs` import fails the BUILD there, not the call.
        const { readFile } = await import('node:fs/promises');
        const { basename } = await import('node:path');
        const bytes = await readFile(filePath);
        form.set('file', new Blob([bytes as unknown as BlobSource]), basename(filePath));
      } else if (options.file !== undefined) {
        const blob = options.file instanceof Blob
          ? options.file
          : new Blob([options.file as unknown as BlobSource]);
        // Python defaults the name to "upload" when `filename` is omitted.
        form.set('file', blob, options.filename ?? 'upload');
      } else {
        form.set('text', String(options.text));
      }

      return transport.request<CreateMemoryResult>('memories_upload', {
        body: form,
        rawBody: true,
      });
    },

    async get(memoryId) {
      if (!memoryId) throw new InvalidInputError('memory_id is required');
      return transport.request<Json>('memories_get', { pathParams: { memory_id: memoryId } });
    },

    async update(options) {
      const memoryId = options.memory_id ?? options.memoryId;
      if (!memoryId) throw new InvalidInputError('memory_id is required');
      if (!options.document) throw new InvalidInputError('document is required');
      const body: Json = {
        document: options.document,
        merge_strategy: options.merge_strategy ?? options.mergeStrategy ?? 'smart-merge',
      };
      const dt = options.document_type ?? options.documentType;
      if (dt !== undefined) body['document_type'] = dt;
      if (options.metadata !== undefined) body['metadata'] = options.metadata;
      // PUT, so idempotent by verb: safe to retry even on an ambiguous failure.
      return transport.request<Json>('memories_update', {
        pathParams: { memory_id: memoryId },
        body,
      });
    },

    async delete(memoryId) {
      // The 0.3.x wrapper let you delete "all of this user's memories" from a
      // process-local id list, which deleted nothing in serverless while
      // reporting success. Deliberately not ported.
      if (!memoryId) {
        throw new InvalidInputError(
          'memory_id is required. Deleting by user is no longer supported: the ' +
            'previous implementation tracked ids in process memory, so it silently ' +
            'deleted nothing in any serverless or multi-process deployment while ' +
            'reporting success.',
        );
      }
      return transport.request<Json>('memories_delete', { pathParams: { memory_id: memoryId } });
    },

    async status(ingestionId) {
      if (!ingestionId) throw new InvalidInputError('ingestion_id is required');
      return transport.request<Json>('memories_status', {
        pathParams: { ingestion_id: ingestionId },
      });
    },

    async wait_for_completion(ingestionId, options = {}) {
      if (!ingestionId) throw new InvalidInputError('ingestion_id is required');
      const timeoutMs =
        (options.timeout_seconds ?? options.timeoutSeconds ?? 300) * 1000;
      const pollMs =
        (options.poll_interval_seconds ?? options.pollIntervalSeconds ?? 2) * 1000;

      const deadline = Date.now() + timeoutMs;
      let last: Json = {};
      for (;;) {
        last = await this.status(ingestionId);
        const status = String(last['status'] ?? '').toLowerCase();
        if (TERMINAL_STATUSES.has(status)) return last;
        if (Date.now() >= deadline) {
          // Python returns the last status rather than raising, so a caller
          // that only checks `.status` behaves identically in both SDKs.
          return last;
        }
        await sleep(Math.min(pollMs, Math.max(0, deadline - Date.now())), options.signal);
      }
    },
  };
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    // NOT unref'd. This is the poll interval inside wait_for_completion, which
    // a caller is awaiting: foreground work. Unref'ing it let a process whose
    // only pending work was this sleep exit mid-poll, so an awaited
    // wait_for_completion never settled. Node calls it "Detected unsettled
    // top-level await". The unref'd timers in this SDK are the two heartbeats
    // and the request watchdog, all of which are genuinely background.
    function onAbort() {
      clearTimeout(timer);
      reject(signal?.reason ?? new Error('Aborted'));
    }
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}
