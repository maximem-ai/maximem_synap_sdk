/**
 * Correlation ID generation.
 *
 * `globalThis.crypto` is unflagged from Node 19 onward and this package
 * declares `engines.node >= 20`, so `randomUUID` is normally present. It is
 * still feature-detected rather than assumed, because `engines` is advisory:
 * npm warns and installs anyway, and the SDK also runs on Workers, Edge and
 * Bun, where the available crypto surface varies.
 *
 * Deliberately does NOT import `node:crypto`. That import is what breaks Edge
 * and Workers builds at bundle time, and a correlation ID is not a security
 * value, so there is nothing to gain from it.
 */
export function newCorrelationId(): string {
  const c = globalThis.crypto;
  if (typeof c?.randomUUID === 'function') return c.randomUUID();

  if (typeof c?.getRandomValues === 'function') {
    const b = c.getRandomValues(new Uint8Array(16));
    // RFC 4122 version (4) and variant bits. The `?? 0` guards satisfy
    // `noUncheckedIndexedAccess`; a 16-byte array always has these slots.
    b[6] = ((b[6] ?? 0) & 0x0f) | 0x40;
    b[8] = ((b[8] ?? 0) & 0x3f) | 0x80;
    const h = Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
    return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
  }

  // Last resort. Correlation IDs are for tracing only, never for security, so
  // Math.random is acceptable here and only here.
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (ch) => {
    const r = (Math.random() * 16) | 0;
    return (ch === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}
