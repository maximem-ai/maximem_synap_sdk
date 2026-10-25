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
import {
  isFilePart, normaliseContent, resolveFile,
  type AttachmentResult, type FilePart, type Objective, type TextPart, type WireFilePart,
  type WireTextPart,
} from './parts.js';

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

/**
 * What `memories.create` takes as `document`. Mirrors Python's
 * `Union[str, Dict[str, Any], TextPart, FilePart, List[Union[str, TextPart, FilePart]]]`.
 *
 * A string, a dict, a `TextPart` or a list with no file in it is sent as JSON
 * to `/create`, byte for byte as a string always was (a list is joined with
 * `"\n"`). Any `FilePart` sends the call as multipart to `/create-multipart`.
 */
export type CreateDocument =
  | string
  | Record<string, unknown>
  | TextPart
  | FilePart
  | Array<string | TextPart | FilePart>;

export interface CreateMemoryOptions {
  document: CreateDocument;
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
  /**
   * What this memory is for. One of `OBJECTIVES` (`conversation`, `bootstrap`,
   * `knowledge-base`, `record`, `profile`) or free text, at most 500
   * characters. Sent only when set, so a call without it is unchanged.
   */
  objective?: Objective | string | null;
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
  /**
   * One per file part, in order, when the call carried files. Absent
   * otherwise. A file that could not be read is still `failed` here while the
   * rest of the document was ingested.
   */
  attachments?: AttachmentResult[] | null;
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

/**
 * The fields every create carries besides `document`, validated, in the
 * pydantic model's order. `documentType` is null only for the multipart
 * manifest, where an unset type is sent as null and the server picks.
 */
function createFields(options: CreateMemoryOptions, defaultType: string | null): Json {
  const o = options as unknown as Record<string, unknown>;
  const documentType = either<string>(o, 'document_type', 'documentType') ?? defaultType;
  const mode = (options.mode as string) ?? DEFAULT_INGEST_MODE;

  // Validated client-side because Python's enum coercion raises before any
  // request is sent. A silent pass-through would turn a typo into a server
  // error, or worse, a silently different extraction path.
  if (documentType !== null && !(DOCUMENT_TYPES as readonly string[]).includes(documentType)) {
    throw new InvalidInputError(
      `Invalid document_type '${documentType}'. Expected one of: ${DOCUMENT_TYPES.join(', ')}`,
    );
  }
  if (!(INGEST_MODES as readonly string[]).includes(mode)) {
    throw new InvalidInputError(
      `Invalid mode '${mode}'. Expected one of: ${INGEST_MODES.join(', ')}`,
    );
  }
  const userId = either<string>(o, 'user_id', 'userId') ?? null;
  const customerId = either<string>(o, 'customer_id', 'customerId') ?? null;

  // Key order matches the pydantic model's field order. Not semantically
  // required, but it makes a captured-body diff against Python readable.
  const fields: Json = {
    document_type: documentType,
    document_id: either<string>(o, 'document_id', 'documentId') ?? null,
    document_created_at: isoOrNull(
      either<string | Date>(o, 'document_created_at', 'documentCreatedAt'),
    ),
    user_id: userId,
    customer_id: customerId,
    // null, never `{}`. An empty path is not absence: the server treats one as
    // an error, because a caller asking to be narrowed and being ignored gets
    // widened instead.
    scope: normaliseCreateScope(options.scope, userId, customerId),
    mode,
    metadata: options.metadata ?? {},
  };
  // Only when set, so every existing body stays byte-identical.
  if (options.objective !== undefined && options.objective !== null) {
    fields['objective'] = options.objective;
  }
  return fields;
}

/**
 * `scope: { client: "self" }` names the client level, which is what a write
 * with no user_id and no customer_id already is. Mirrors Python's
 * `CreateMemoryRequest` validator.
 *
 * - On its own, it is sent as `scope: null`, so the body is the one a call
 *   without it sends.
 * - With a user_id or customer_id, it is a contradiction, refused.
 * - `"self"` under any key alongside other entries is refused.
 * - Every other use of a `client` key is passed through untouched.
 */
export function normaliseCreateScope(
  scope: Record<string, string> | null | undefined,
  userId: string | null,
  customerId: string | null,
): Record<string, string> | null {
  if (scope === undefined || scope === null) return null;
  const entries = Object.entries(scope);
  if (entries.length > 1 && entries.some(([, v]) => v === 'self')) {
    throw new InvalidInputError(
      `"self" can only be used alone, as scope={"client": "self"}; got ${JSON.stringify(scope)}`,
    );
  }
  if (entries.length === 1 && entries[0]![0] === 'client' && entries[0]![1] === 'self') {
    // Python checks `is not None`, so an empty-string id counts as given.
    if (userId !== null || customerId !== null) {
      throw new InvalidInputError(
        'scope={"client": "self"} names the client level, so it ' +
          'cannot be combined with user_id or customer_id',
      );
    }
    return null;
  }
  return scope;
}

/** What `document` becomes on the JSON route: a string, or a dict untouched. */
function jsonDocument(document: unknown, where: string): unknown {
  const n = normaliseContent(document, 'document');
  if (n === null) return document;
  if (n.kind === 'parts') {
    throw new InvalidInputError(
      `${where} cannot carry a FilePart. memories.create sends files as multipart; ` +
        'call it once per document that has files.',
    );
  }
  return n.text;
}

/**
 * Byte-for-byte equivalent of Python's `CreateMemoryRequest.model_dump(mode="json")`.
 *
 * Takes the JSON-route shapes only: a string, a dict, a `TextPart`, or a list
 * with no file in it (joined with `"\n"`). A `FilePart` throws; it is sent by
 * `memories.create` as multipart, built by `buildCreateManifest`.
 */
export function buildCreateBody(options: CreateMemoryOptions): Json {
  const document = jsonDocument(options.document, 'This request');
  if (!document) throw new InvalidInputError('document is required');
  return { document, ...createFields(options, DEFAULT_DOCUMENT_TYPE) };
}

/**
 * The `manifest` form field of a multipart create, before any file is read.
 *
 * `document` lists the parts in order; file `i` (counting files only) is the
 * form field `file_<i>`. `document_type` is null unless the caller set it.
 */
export function buildCreateManifest(
  options: CreateMemoryOptions,
  parts: Array<TextPart | FilePart>,
  files: Array<{ filename: string; media_type: string }>,
): Json {
  let next = 0;
  const document = parts.map((p): WireTextPart | WireFilePart => {
    if (!isFilePart(p)) return { type: 'text', text: (p as TextPart).text };
    const index = next++;
    const file = files[index]!;
    return { type: 'file', field: `file_${index}`, filename: file.filename, media_type: file.media_type };
  });
  return { document, ...createFields(options, null) };
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

/** Mirrors the Python ForgetMode enum. */
export const FORGET_MODES = ['preview', 'direct'] as const;
export type ForgetMode = (typeof FORGET_MODES)[number];

export const FORGET_DESCRIPTION_MIN = 3;
export const FORGET_DESCRIPTION_MAX = 500;

export interface ForgetOptions {
  /** What to forget, in the user's own words (3 to 500 characters). */
  description: string;
  user_id?: string | null;
  userId?: string | null;
  customer_id?: string | null;
  customerId?: string | null;
  /**
   * The level to search, named in full, one entry per level down to it:
   * `{ customer: "acme", team: "payments", user: "dana" }`. Only needed when
   * your scope ladder has a level between customer and user.
   */
  scope?: Record<string, string> | null;
  /** `"preview"` (default) or `"direct"`. See `MemoriesInterface.forget`. */
  mode?: ForgetMode | string;
}

/** A memory the description matched. Mirrors Python's `ForgetCandidate`. */
export interface ForgetCandidate {
  memory_id: string;
  content: string;
  memory_type?: string | null;
  /** `"memory"`, or where a copy lives: chunk, sentence, entity, triple, profile, candidate. */
  store?: string;
}

export type ForgetStatus = 'nothing_matches' | 'preview' | 'deleted' | 'partial';

/** Result of `memories.forget`. Mirrors Python's `ForgetResponse`. */
export interface ForgetResult {
  status: ForgetStatus;
  request_id: string | null;
  candidates: ForgetCandidate[];
  deleted_ids: string[];
  failed_ids: string[];
  still_present_in: string[];
  may_be_incomplete: boolean;
}

/** The body Python builds for `memories.forget`, key order included. */
export function buildForgetBody(options: ForgetOptions): Json {
  const description = options.description;
  if (
    typeof description !== 'string' ||
    // Code points, as Python's len() and the server count; `.length` counts
    // UTF-16 units, so 251 emoji (502 units) were refused here only.
    [...description].length < FORGET_DESCRIPTION_MIN ||
    [...description].length > FORGET_DESCRIPTION_MAX
  ) {
    throw new InvalidInputError(
      `description must be between ${FORGET_DESCRIPTION_MIN} and ${FORGET_DESCRIPTION_MAX} characters`,
    );
  }
  const mode = (options.mode as string | undefined) ?? 'preview';
  if (!(FORGET_MODES as readonly string[]).includes(mode)) {
    throw new InvalidInputError(
      `Invalid mode '${mode}'. Expected one of: ${FORGET_MODES.join(', ')}`,
    );
  }
  const o = options as unknown as Record<string, unknown>;
  return {
    description,
    user_id: either<string>(o, 'user_id', 'userId') ?? null,
    customer_id: either<string>(o, 'customer_id', 'customerId') ?? null,
    scope: options.scope ?? null,
    mode,
  };
}

export interface MemoriesInterface {
  create(options: CreateMemoryOptions): Promise<CreateMemoryResult>;
  batch_create(options: BatchCreateOptions): Promise<BatchCreateResult>;
  create_from_file(options: CreateFromFileOptions): Promise<CreateMemoryResult>;
  get(memoryId: string): Promise<Json>;
  update(options: UpdateMemoryOptions): Promise<Json>;
  delete(memoryId: string): Promise<Json>;
  /**
   * Forget memories by describing them in the user's own words. Use it when
   * you do not hold memory ids; to delete one memory by id, use `delete()`.
   *
   * The call must be scoped to exactly one user: pass `user_id`, plus
   * `customer_id` on a B2B instance. Otherwise the server answers 400, raised
   * as `InvalidInputError`.
   *
   * Modes:
   * - `"preview"` (default): files a pending request that a person approves in
   *   the dashboard under Deletion requests. Deletes nothing. `candidates`
   *   lists what matched.
   * - `"direct"`: finds and soft-deletes in one call. Not idempotent as a
   *   request: a repeat is harmless and finds nothing new, but it is a new
   *   request.
   *
   * `status` is `"nothing_matches"`, `"preview"`, `"deleted"` or `"partial"`.
   * `"partial"` means the memories are hidden but a store named in
   * `still_present_in` may still hold the text, so do not tell an end user it
   * was removed unless `status === "deleted"`. `failed_ids` lists memories that
   * could not be deleted; `may_be_incomplete` is true when the match may not
   * have covered everything.
   *
   * Errors: `InvalidInputError` for 400/422 or a bad `description`/`mode`
   * caught before sending; `ServiceUnavailableError` for 503 (the matcher was
   * unavailable and nothing was deleted, so trying again is safe; the
   * transport applies its normal retry policy first); `RateLimitError` for 429.
   */
  forget(options: ForgetOptions): Promise<ForgetResult>;
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

// The server's statuses are queued, processing, completed, failed and
// partial_success. Python stops on the last three; 'error' and 'cancelled' are
// never emitted by the server and are kept only so an older value still stops.
const TERMINAL_STATUSES = new Set(['completed', 'failed', 'partial_success', 'error', 'cancelled']);

export function createMemoriesInterface(
  transport: HttpTransport,
  // Called after a forget that deleted something, so the caller can stop serving
  // that user's cached context. Mirrors Python's `_evict_forgotten`.
  onForgotten?: (ids: { userId: string | null; customerId: string | null }) => void,
): MemoriesInterface {
  return {
    async create(options) {
      const o = options as unknown as Record<string, unknown>;
      const customerId = o['customer_id'] ?? o['customerId'];
      if (customerId !== undefined && customerId !== null && customerId !== '') {
        await transport.checkCustomerId(customerId, 'memories.create');
      }
      const n = normaliseContent(options.document, 'document');
      if (n === null || n.kind === 'text') {
        return transport.request<CreateMemoryResult>('memories_create', {
          body: buildCreateBody(options),
        });
      }

      // Everything but the files is validated before any file is read, so a
      // typo in document_type does not cost a 10 MB download first.
      buildCreateManifest(options, n.parts.filter((p) => !isFilePart(p)), []);
      const fetchImpl = ((url: string, init?: RequestInit) =>
        transport.fetchExternal(url, init)) as typeof fetch;
      const files = await Promise.all(
        n.parts.filter(isFilePart).map((p) => resolveFile(p, fetchImpl)),
      );

      const form = new FormData();
      form.set('manifest', JSON.stringify(buildCreateManifest(options, n.parts, files)));
      files.forEach((file, i) => {
        form.set(
          `file_${i}`,
          new Blob([file.bytes as unknown as BlobSource], { type: file.media_type }),
          file.filename,
        );
      });
      // Not retried: the endpoint is declared non-idempotent, as /create is.
      return transport.request<CreateMemoryResult>('memories_create_multipart', {
        body: form,
        rawBody: true,
        // A server older than this route answers 404, or 405 because
        // /api/v1/memories/{memory_id} exists with other methods. Either would
        // otherwise read as "context not found" or a bare HTTP error.
        // InvalidInputError, with Python's message.
        mapError: (status, error) =>
          status === 404 || status === 405
            ? new InvalidInputError(
              'This Synap server does not support file parts in memories.create. ' +
                'Upgrade the Synap server, or send text only.',
              { correlationId: error.correlationId, cause: error },
            )
            : undefined,
      });
    },

    async batch_create(options) {
      const docs = options.documents;
      if (!Array.isArray(docs) || docs.length === 0) {
        throw new InvalidInputError('documents must be a non-empty array');
      }
      const failFast = options.fail_fast ?? options.failFast ?? false;
      return transport.request<BatchCreateResult>('memories_batch', {
        body: {
          documents: docs.map((d) => {
            jsonDocument(d.document, 'memories.batch_create');
            return buildCreateBody(d);
          }),
          fail_fast: failFast,
        },
      });
    },

    async create_from_file(options) {
      const userId = options.user_id ?? options.userId;
      const customerId = options.customer_id ?? options.customerId;
      // Checked first, as Python checks it: a flat map of level key to id.
      // Until 0.5.20 any shape was serialised and sent.
      const scope = options.scope as unknown;
      if (scope !== undefined && scope !== null && (
        typeof scope !== 'object' || Array.isArray(scope)
        || !Object.values(scope as Record<string, unknown>).every((v) => typeof v === 'string')
      )) {
        throw new InvalidInputError('scope must be a flat dict of level key to id, both strings.');
      }
      if (!userId) throw new InvalidInputError('user_id is required');
      // Not required: rejected on B2C, so requiring it made the documented
      // default combination impossible to call.
      await transport.checkCustomerId(customerId, 'memories.create_from_file');

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
      // Only when there is one. `form.set('customer_id', undefined)` sends the
      // string "undefined", which a B2C instance refuses as a customer id.
      if (typeof customerId === 'string' && customerId !== '') {
        form.set('customer_id', customerId);
      }
      form.set('relationship_type', options.relationship_type ?? options.relationshipType ?? 'b2c');
      form.set('mode', String(options.mode ?? DEFAULT_INGEST_MODE));

      // Python appends these only when truthy, so an empty document_type or an
      // empty metadata object is absent from the form rather than present-empty.
      const documentType = options.document_type ?? options.documentType;
      if (documentType) form.set('document_type', String(documentType));
      if (options.metadata !== undefined && Object.keys(options.metadata).length > 0) {
        form.set('metadata', JSON.stringify(options.metadata));
      }
      // The upload route is multipart and cannot carry an object, so the scope
      // map goes as a JSON string; the server parses it with json.loads.
      if (options.scope && Object.keys(options.scope).length > 0) {
        form.set('scope', JSON.stringify(options.scope));
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

    async forget(options) {
      const o = options as unknown as Record<string, unknown>;
      await transport.checkCustomerId(o['customer_id'] ?? o['customerId'], 'memories.forget');
      const body = buildForgetBody(options) as Record<string, unknown>;
      const result = await transport.request<ForgetResult>('memories_forget', { body: body as Json });
      // A preview deletes nothing, so it evicts nothing. Never let a cache
      // problem turn a successful delete into an error.
      // A forget can remove only copies (a profile entry, a chunk) and so return no
      // memory ids; those are served from the same caches.
      const removed = !!result?.deleted_ids?.length
        || result?.status === 'deleted' || result?.status === 'partial';
      if (removed && onForgotten) {
        try {
          onForgotten({
            userId: (body['user_id'] as string | null) ?? null,
            customerId: (body['customer_id'] as string | null) ?? null,
          });
        } catch { /* eviction is best effort */ }
      }
      return result;
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
