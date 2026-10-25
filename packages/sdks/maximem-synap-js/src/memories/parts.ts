/**
 * Content parts: text and files (today, images) carried by `memories.create`,
 * `conversation.record_message` and `instance.send_message`.
 *
 * Mirrors `TextPart`, `FilePart` and `AttachmentResult` in the Python SDK.
 *
 * ## A text-only list is a string
 *
 * Every existing call has to keep sending exactly the bytes it sends today, so
 * a list with no file in it is joined with `"\n"` and sent as that one string,
 * down the same route a string always took. Only a file changes the shape of a
 * request.
 *
 * ## Runs everywhere
 *
 * Nothing here imports a node builtin at the top level. Reading a `path` uses a
 * lazy `import('node:fs/promises')`, the same as `create_from_file`, and base64
 * uses `Buffer` when the runtime has it and `btoa` when it does not, so the main
 * entry stays safe on Edge, Workers and in a browser.
 */

import { InvalidInputError } from '../errors.js';

/**
 * What the caller is asking a memory to do. Stored with the ingestion as
 * metadata; any other string is accepted too. Mirrors Python's `OBJECTIVES`.
 */
export const OBJECTIVES = [
  'conversation', 'bootstrap', 'knowledge-base', 'record', 'profile',
] as const;
export type Objective = (typeof OBJECTIVES)[number];

/** Each file, however it is supplied. Larger is refused before anything is sent. */
export const MAX_FILE_BYTES = 10 * 1024 * 1024;
/**
 * Total raw image bytes one `instance.send_message` event may carry. The gRPC
 * stream's message limit is 4 MB and base64 adds a third, so 2.5 MB of raw
 * bytes is what fits with room for the rest of the event.
 */
export const MAX_STREAM_IMAGE_BYTES = 2.5 * 1024 * 1024;
/** A `url` part is fetched by the SDK, never by the server. */
export const URL_FETCH_TIMEOUT_MS = 30_000;

/** Raw bytes for a `FilePart`. A Node `Buffer` is a `Uint8Array`. */
export type FileData = Uint8Array | ArrayBuffer | Blob;

export interface TextPart {
  type: 'text';
  text: string;
}

/**
 * A file. Exactly one of `data`, `path` or `url`.
 *
 * Build one with `filePart()`, which fills `filename` and `media_type`. A
 * literal `{ type: 'file', ... }` works too; the same defaults are applied when
 * it is sent.
 */
export interface FilePart {
  type: 'file';
  data?: FileData;
  /** Node only: read from disk when the call is made. */
  path?: string;
  /** Fetched by the SDK when the call is made (30 s, at most 10 MB). */
  url?: string;
  /** Defaults to the basename of `path` or `url`, else `"file"`. */
  filename?: string;
  /** Defaults to a guess from `filename`, else `"application/octet-stream"`. */
  media_type?: string;
}

export interface FilePartInit {
  data?: FileData | null;
  path?: string | null;
  url?: string | null;
  filename?: string | null;
  media_type?: string | null;
  mediaType?: string | null;
}

/** One item of a content list. A bare string is a text part. */
export type ContentPart = string | TextPart | FilePart;

/**
 * What the server did with one file of a `memories.create` call, in the order
 * the files were sent. Mirrors Python's `AttachmentResult`.
 */
export interface AttachmentResult {
  filename: string;
  media_type: string;
  status: 'read' | 'failed' | string;
  error?: string | null;
}

/**
 * Marks an object made by `textPart()` or `filePart()`.
 *
 * A plain object passed as `document` has always been a dict document, sent
 * untouched, and one may well carry `type: "text"` of its own. So at the top
 * level only a branded object is read as a part; inside a list (never valid
 * before parts existed) a plain `{ type: 'text' | 'file' }` is accepted too.
 * Non-enumerable, so it never reaches JSON or a deep-equality check, and
 * `Symbol.for` so two copies of the SDK (ESM and CJS) recognise each other's.
 */
const PART_BRAND = Symbol.for('@maximem/synap-js-sdk.ContentPart');

function brand<T extends object>(part: T): T {
  Object.defineProperty(part, PART_BRAND, { value: true, enumerable: false });
  return part;
}

function isBranded(value: unknown): boolean {
  return typeof value === 'object' && value !== null && PART_BRAND in value;
}

/** `{ type: 'text', text }`. Mirrors Python's `TextPart(text)`. */
export function textPart(text: string): TextPart {
  if (typeof text !== 'string') throw new InvalidInputError('TextPart text must be a string');
  return brand({ type: 'text', text });
}

/**
 * A file part with its defaults filled in. Mirrors Python's `FilePart(...)`.
 *
 * Throws `InvalidInputError` unless exactly one of `data`, `path` or `url` is
 * given.
 */
export function filePart(init: FilePartInit): FilePart {
  const source = checkOneSource(init);
  const filename = defaultFilename(init, source);
  const mediaType =
    init.media_type ?? init.mediaType ?? defaultMediaType(filename, init.data ?? undefined);
  const part: FilePart = { type: 'file' };
  if (source === 'data') part.data = init.data as FileData;
  else if (source === 'path') part.path = init.path as string;
  else part.url = init.url as string;
  part.filename = filename;
  part.media_type = mediaType;
  return brand(part);
}

export function isTextPart(value: unknown): value is TextPart {
  return isPlainPart(value) && value['type'] === 'text' && typeof value['text'] === 'string';
}

export function isFilePart(value: unknown): value is FilePart {
  return isPlainPart(value) && value['type'] === 'file';
}

function isPlainPart(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

type Source = 'data' | 'path' | 'url';

function checkOneSource(p: { data?: unknown; path?: unknown; url?: unknown }): Source {
  const given = (['data', 'path', 'url'] as const).filter(
    (k) => p[k] !== undefined && p[k] !== null,
  );
  if (given.length !== 1) {
    // Python's message, word for word.
    throw new InvalidInputError(
      'FilePart takes exactly one of data, path or url' +
        (given.length > 0 ? ` (got ${given.join(', ')})` : ''),
    );
  }
  return given[0]!;
}

function lastSegment(p: string): string {
  const parts = p.split(/[\\/]/);
  return parts[parts.length - 1] ?? '';
}

function defaultFilename(
  p: { data?: unknown; path?: unknown; url?: unknown; filename?: string | null },
  source: Source,
): string {
  if (p.filename) return p.filename;
  if (source === 'path') return lastSegment(String(p.path)) || 'file';
  if (source === 'url') {
    let pathname = '';
    try {
      pathname = new URL(String(p.url)).pathname;
    } catch {
      pathname = String(p.url).split(/[?#]/)[0] ?? '';
    }
    return lastSegment(pathname) || 'file';
  }
  // A browser `File` carries its own name. Python bytes cannot, so this only
  // ever fills a gap Python leaves as "file".
  const named = (p.data as { name?: unknown } | undefined)?.name;
  return typeof named === 'string' && named ? named : 'file';
}

/**
 * Python's table (png, jpg, jpeg, webp, gif, pdf) plus what its
 * `mimetypes.guess_type` answers for other common extensions. Case-insensitive,
 * as Python's is. The server sniffs image bytes regardless, so a wrong or
 * missing guess is corrected there, never trusted.
 */
const MEDIA_TYPES: Record<string, string> = {
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  jpe: 'image/jpeg',
  webp: 'image/webp',
  gif: 'image/gif',
  bmp: 'image/bmp',
  tif: 'image/tiff',
  tiff: 'image/tiff',
  svg: 'image/svg+xml',
  pdf: 'application/pdf',
  txt: 'text/plain',
  csv: 'text/csv',
  htm: 'text/html',
  html: 'text/html',
  json: 'application/json',
  doc: 'application/msword',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  pptx: 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
  mp3: 'audio/mpeg',
  wav: 'audio/x-wav',
  mp4: 'video/mp4',
};

export function guessMediaType(filename: string): string | null {
  const dot = filename.lastIndexOf('.');
  // Python's splitext gives ".png" no extension, so neither does this.
  if (dot <= 0) return null;
  const ext = filename.slice(dot + 1).toLowerCase();
  return MEDIA_TYPES[ext] ?? null;
}

function defaultMediaType(filename: string, data?: unknown): string {
  const guessed = guessMediaType(filename);
  if (guessed !== null) return guessed;
  const blobType = (data as { type?: unknown } | undefined)?.type;
  return typeof blobType === 'string' && blobType ? blobType : DEFAULT_MEDIA_TYPE;
}

/**
 * The outcome of normalising a content value.
 *
 * `text`: send this string exactly as a string always was sent.
 * `parts`: at least one file; strings have become text parts, in order.
 */
export type Normalised =
  | { kind: 'text'; text: string }
  | { kind: 'parts'; parts: Array<TextPart | FilePart> };

/**
 * Normalise a part, or a list of parts, for the field named `field`.
 *
 * Returns null for anything that is neither: a string, or a dict document,
 * which the caller passes through untouched so its bytes do not change. A
 * single part counts only when `textPart()` or `filePart()` made it; a plain
 * object is a dict, as it always was.
 */
export function normaliseContent(value: unknown, field: string): Normalised | null {
  if (isBranded(value) && isTextPart(value)) return { kind: 'text', text: value.text };
  if (isBranded(value) && isFilePart(value)) return { kind: 'parts', parts: [value] };
  if (!Array.isArray(value)) return null;
  if (value.length === 0) {
    // Explicit: an empty array is truthy, so the `!document` check that
    // guards a string never sees it.
    throw new InvalidInputError(`${field} is an empty list; pass at least one part`);
  }
  const parts: Array<TextPart | FilePart> = value.map((item, i) => {
    if (typeof item === 'string') return { type: 'text', text: item };
    if (isTextPart(item) || isFilePart(item)) return item;
    throw new InvalidInputError(
      `${field}[${i}] is a ${typeName(item)}; expected a string, TextPart or FilePart`,
    );
  });
  if (!parts.some(isFilePart)) {
    return { kind: 'text', text: parts.map((p) => (p as TextPart).text).join('\n') };
  }
  return { kind: 'parts', parts };
}

function typeName(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  if (typeof value === 'object') {
    return (value as { constructor?: { name?: string } }).constructor?.name ?? 'object';
  }
  return typeof value;
}

/** A file's resolved name, type and bytes. */
export interface ResolvedFile {
  filename: string;
  media_type: string;
  bytes: Uint8Array;
}

/**
 * Read a file part's bytes: from `data`, from disk, or over HTTP.
 *
 * `fetchImpl` is the transport's fetch, so a test or a proxying caller sees
 * the request. It is called with no Synap credentials: the URL belongs to
 * somebody else.
 */
export async function resolveFile(part: FilePart, fetchImpl: typeof fetch): Promise<ResolvedFile> {
  const source = checkOneSource(part);
  const filename = defaultFilename(part, source);
  let mediaType = part.media_type || defaultMediaType(filename, part.data);

  let bytes: Uint8Array;
  if (source === 'data') {
    bytes = await toBytes(part.data as FileData);
  } else if (source === 'path') {
    // Lazy, so importing the SDK on Edge or in a Worker stays safe: a
    // top-level `node:fs` import fails the BUILD there, not the call.
    const { readFile } = await import('node:fs/promises');
    try {
      bytes = new Uint8Array(await readFile(part.path as string));
    } catch (error) {
      throw new InvalidInputError(
        `Could not read FilePart path '${part.path}': ${(error as Error)?.message ?? String(error)}`,
        { cause: error },
      );
    }
  } else {
    const fetched = await fetchUrl(part.url as string, fetchImpl);
    bytes = fetched.bytes;
    // The filename said nothing useful, so trust what the server said, as
    // Python does.
    if (mediaType === DEFAULT_MEDIA_TYPE && fetched.contentType) mediaType = fetched.contentType;
  }
  if (bytes.byteLength > MAX_FILE_BYTES) {
    throw new InvalidInputError(
      `${filename} is ${bytes.byteLength} bytes; the limit is ${MAX_FILE_BYTES} bytes per file`,
    );
  }
  return { filename, media_type: mediaType, bytes };
}

const DEFAULT_MEDIA_TYPE = 'application/octet-stream';

function urlTooLarge(url: string): string {
  return `FilePart url '${url}' is over the limit of ${MAX_FILE_BYTES} bytes per file`;
}

async function toBytes(data: FileData): Promise<Uint8Array> {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) {
    const view = data as ArrayBufferView;
    return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  }
  if (typeof Blob !== 'undefined' && data instanceof Blob) {
    return new Uint8Array(await data.arrayBuffer());
  }
  throw new InvalidInputError('FilePart data must be a Uint8Array, ArrayBuffer or Blob');
}

/** GET a URL, at most 10 MB, within 30 s. Every failure is the caller's input. */
export async function fetchUrlBytes(url: string, fetchImpl: typeof fetch): Promise<Uint8Array> {
  return (await fetchUrl(url, fetchImpl)).bytes;
}

async function fetchUrl(
  url: string,
  fetchImpl: typeof fetch,
): Promise<{ bytes: Uint8Array; contentType: string | null }> {
  let scheme = '';
  try {
    scheme = new URL(url).protocol;
  } catch {
    throw new InvalidInputError(`Could not fetch FilePart url '${url}': not a valid URL`);
  }
  // What httpx supports, so the same URLs fail in both SDKs.
  if (scheme !== 'http:' && scheme !== 'https:') {
    throw new InvalidInputError(
      `Could not fetch FilePart url '${url}': only http and https are supported`,
    );
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), URL_FETCH_TIMEOUT_MS);
  try {
    const response = await fetchImpl(url, { method: 'GET', signal: controller.signal });
    if (response.status < 200 || response.status >= 300) {
      throw new InvalidInputError(
        `Fetching FilePart url '${url}' returned HTTP ${response.status}`,
      );
    }
    const declared = Number(response.headers.get('content-length') ?? '');
    if (Number.isFinite(declared) && declared > MAX_FILE_BYTES) {
      throw new InvalidInputError(urlTooLarge(url));
    }
    const bytes = await readCapped(response, url);
    const rawType = response.headers.get('content-type');
    const contentType = rawType ? rawType.split(';', 1)[0]!.trim().toLowerCase() || null : null;
    return { bytes, contentType };
  } catch (error) {
    if (error instanceof InvalidInputError) throw error;
    if (controller.signal.aborted) {
      throw new InvalidInputError(
        `Could not fetch FilePart url '${url}': timed out after ${URL_FETCH_TIMEOUT_MS / 1000} s`,
        { cause: error },
      );
    }
    throw new InvalidInputError(
      `Could not fetch FilePart url '${url}': ${(error as Error)?.message ?? String(error)}`,
      { cause: error },
    );
  } finally {
    clearTimeout(timer);
  }
}

/** Stops reading as soon as the body passes the cap, rather than buffering all of it. */
async function readCapped(response: Response, url: string): Promise<Uint8Array> {
  const body = response.body;
  if (body === null || typeof body.getReader !== 'function') {
    return new Uint8Array(await response.arrayBuffer());
  }
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > MAX_FILE_BYTES) {
      await reader.cancel().catch(() => {});
      throw new InvalidInputError(urlTooLarge(url));
    }
    chunks.push(value);
  }
  const out = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.byteLength;
  }
  return out;
}

/** Standard base64. `Buffer` where there is one, chunked `btoa` everywhere else. */
export function toBase64(bytes: Uint8Array): string {
  const B = (globalThis as { Buffer?: { from(b: Uint8Array): { toString(enc: string): string } } })
    .Buffer;
  if (B !== undefined) return B.from(bytes).toString('base64');
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/** `{type: "text", text}` on the wire. */
export interface WireTextPart {
  type: 'text';
  text: string;
}

/** A file in a `memories.create` multipart manifest; its bytes are the form field `field`. */
export interface WireFilePart {
  type: 'file';
  field: string;
  filename: string;
  media_type: string;
}

/** A file carried inline as base64 (record_message JSON, the gRPC metadata key). */
export interface WireImagePart {
  type: 'image';
  filename: string;
  media_type: string;
  data: string;
}

/**
 * What an image becomes in text where it cannot be read: `[image: <filename>]`.
 * Text parts stay as they are and the pieces are joined with `"\n"`.
 */
export function placeholderText(parts: Array<{ type: string; text?: string; filename?: string }>): string {
  return parts
    .map((p) => (p.type === 'text' ? String(p.text ?? '') : `[image: ${p.filename || 'image'}]`))
    .join('\n');
}

export interface InlineParts {
  /** Text and inline-image wire parts, in order. */
  wire: Array<WireTextPart | WireImagePart>;
  /** The placeholder rendering of `wire`. */
  placeholder: string;
  /** Raw bytes across every file, before base64. */
  rawBytes: number;
}

/** Read every file (concurrently, order kept) and inline it as base64. */
export async function toInlineParts(
  parts: Array<TextPart | FilePart>,
  fetchImpl: typeof fetch,
): Promise<InlineParts> {
  const resolved = await Promise.all(
    parts.map((p) => (isFilePart(p) ? resolveFile(p, fetchImpl) : null)),
  );
  let rawBytes = 0;
  const wire = parts.map((p, i): WireTextPart | WireImagePart => {
    const file = resolved[i];
    if (file === null || file === undefined) return { type: 'text', text: (p as TextPart).text };
    rawBytes += file.bytes.byteLength;
    return {
      type: 'image',
      filename: file.filename,
      media_type: file.media_type,
      data: toBase64(file.bytes),
    };
  });
  return { wire, placeholder: placeholderText(wire), rawBytes };
}
