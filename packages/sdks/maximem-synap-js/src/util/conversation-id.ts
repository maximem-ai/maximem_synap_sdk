/**
 * The one id every local cache is keyed by for a conversation.
 *
 * The stream and the server accept any string as a conversation id and turn one
 * that is not a UUID into `uuid5(NAMESPACE_URL, id)` (the server's
 * `cloud/shared/conversation_id_norm.coerce_conversation_id`). The REST reads
 * need that UUID, so a caller who sends a turn on the stream under `pgv2_abc`
 * and reads under the UUID derived from it is talking about ONE conversation.
 * A local store that keys the two differently holds the turn under one name and
 * is asked for it under the other.
 *
 * A string that parses as a UUID, in any spelling Python's `uuid.UUID(...)`
 * accepts, is returned exactly as it came. Empty values pass through.
 * Idempotent. Mirrors Python's `utils.validators.normalize_conversation_id`.
 *
 * SHA-1 is written out here rather than imported from `node:crypto`: that
 * import breaks Edge and Workers builds (see `correlation.ts`), and `SubtleCrypto`
 * is async where every caller of this is not.
 */

/** `uuid.NAMESPACE_URL`. */
const NAMESPACE_URL = [
  0x6b, 0xa7, 0xb8, 0x11, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8,
];

/**
 * What `uuid.UUID(text)` accepts: `urn:` / `uuid:` prefixes dropped, braces and
 * dashes dropped, 32 characters left, and `int(rest, 16)` succeeds (which also
 * takes a `+`, a `0x`, single underscores between digits and surrounding
 * whitespace).
 */
function parsesAsUuid(text: string): boolean {
  const hex = text.replace(/urn:/g, '').replace(/uuid:/g, '').replace(/^[{}]+|[{}]+$/g, '').replace(/-/g, '');
  if (hex.length !== 32) return false;
  return /^\s*\+?(0[xX]_?)?[0-9a-fA-F]+(_[0-9a-fA-F]+)*\s*$/.test(hex);
}

function sha1(bytes: Uint8Array): Uint8Array {
  const rotl = (x: number, n: number): number => (x << n) | (x >>> (32 - n));
  const bitLength = bytes.length * 8;
  const padded = new Uint8Array(((bytes.length + 9 + 63) >> 6) << 6);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 8, Math.floor(bitLength / 0x100000000));
  view.setUint32(padded.length - 4, bitLength >>> 0);
  let h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476, h4 = 0xc3d2e1f0;
  const w = new Array<number>(80);
  for (let off = 0; off < padded.length; off += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(off + i * 4);
    for (let i = 16; i < 80; i++) w[i] = rotl((w[i - 3]!) ^ (w[i - 8]!) ^ (w[i - 14]!) ^ (w[i - 16]!), 1);
    let a = h0, b = h1, c = h2, d = h3, e = h4;
    for (let i = 0; i < 80; i++) {
      let f: number, k: number;
      if (i < 20) { f = (b & c) | (~b & d); k = 0x5a827999; }
      else if (i < 40) { f = b ^ c ^ d; k = 0x6ed9eba1; }
      else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdc; }
      else { f = b ^ c ^ d; k = 0xca62c1d6; }
      const t = (rotl(a, 5) + f + e + k + (w[i]!)) | 0;
      e = d; d = c; c = rotl(b, 30); b = a; a = t;
    }
    h0 = (h0 + a) | 0; h1 = (h1 + b) | 0; h2 = (h2 + c) | 0; h3 = (h3 + d) | 0; h4 = (h4 + e) | 0;
  }
  const out = new Uint8Array(20);
  const ov = new DataView(out.buffer);
  [h0, h1, h2, h3, h4].forEach((h, i) => ov.setUint32(i * 4, h >>> 0));
  return out;
}

function uuid5Url(name: string): string {
  const nameBytes = new TextEncoder().encode(name);
  const input = new Uint8Array(16 + nameBytes.length);
  input.set(NAMESPACE_URL);
  input.set(nameBytes, 16);
  const hash = sha1(input).slice(0, 16);
  hash[6] = ((hash[6] ?? 0) & 0x0f) | 0x50;
  hash[8] = ((hash[8] ?? 0) & 0x3f) | 0x80;
  const h = Array.from(hash, (x) => x.toString(16).padStart(2, '0')).join('');
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

export function normalizeConversationId(conversationId: string): string;
export function normalizeConversationId(conversationId: string | null | undefined): string | null | undefined;
export function normalizeConversationId(conversationId: string | null | undefined): string | null | undefined {
  if (!conversationId) return conversationId;
  return parsesAsUuid(conversationId) ? conversationId : uuid5Url(conversationId);
}
