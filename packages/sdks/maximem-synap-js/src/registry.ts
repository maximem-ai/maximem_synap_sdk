/**
 * Client registry: one shared client per identity, per process.
 *
 * Mirrors Python's `registry.py`. Without it, every `new SynapClient()` gets
 * its own anticipation cache, so a framework integration that constructs a
 * client per request or per agent never gets a cache hit and pays for a fresh
 * retrieval every time. Python has shared state here since 0.2; matching it is
 * the difference between a warm cache and a metered fetch on every turn.
 *
 * JavaScript reaches the same end differently. Python does
 * `self.__dict__ = existing.__dict__`, aliasing state between distinct
 * objects. A JS constructor can simply RETURN the existing instance, so the
 * two callers hold the same object rather than two objects sharing a state
 * bag. That is stricter than Python and avoids its `__dict__`-identity
 * comparisons entirely.
 */

/** Kept structural so the registry does not depend on the client's type. */
export interface RegistrableClient {
  shutdown(): Promise<void>;
}

const instances = new Map<string, RegistrableClient>();

/**
 * Which registry slot an identity belongs in.
 *
 * `instanceId` keys the singleton when present, but it is usually empty at
 * construction and only resolved from the API key during `initialize()`.
 * Keying on it directly would put every `new SynapClient({ apiKey })` in one
 * process onto the single `''` slot, where the second would adopt the first's
 * credentials. So fall back to the credential that WILL resolve the identity,
 * stored as a truncated SHA-256 digest rather than plaintext, so a registry
 * dump or a log line cannot leak a key.
 *
 * `SYNAP_INSTANCE_ID` is deliberately not consulted: the client applies that
 * fallback after the lookup, and honouring it here would put two different
 * credentials back onto one slot whenever the variable happens to be set.
 */
export function buildRegistryKey(instanceId: string, apiKey: string | undefined): string {
  if (instanceId) return instanceId;
  const key = apiKey ?? '';
  // Nothing to key on. The client cannot initialise without a credential
  // anyway, so preserve the empty slot rather than inventing one.
  if (!key) return '';
  return `apikey:${digest(key)}`;
}

/**
 * 64-bit FNV-1a over the credential.
 *
 * Python uses SHA-256 here. This cannot: the digest is needed synchronously
 * inside a constructor, `node:crypto` is unavailable on Edge and Workers (and
 * a static import of it would fail those builds outright), and Web Crypto's
 * `subtle.digest` is async. So this is a non-cryptographic digest, chosen for
 * the one property that actually matters here: the raw key never becomes a Map
 * key that something could enumerate or log.
 *
 * It is not collision-resistant against an adversary, and it does not need to
 * be. A collision would make two clients share state, so the bound that
 * matters is accidental collision between the handful of distinct API keys
 * present in a single process, which at 64 bits is negligible. It is never
 * persisted, never sent anywhere, and never compared against anything outside
 * this module.
 */
function digest(value: string): string {
  // FNV-1a, 64-bit. BigInt keeps the arithmetic exact; Number would lose
  // precision past 2^53 and collapse the space.
  const PRIME = 1099511628211n;
  const MASK = 0xffffffffffffffffn;
  let hash = 14695981039346656037n;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= BigInt(value.charCodeAt(i));
    hash = (hash * PRIME) & MASK;
  }
  return hash.toString(16).padStart(16, '0');
}

export function get(key: string): RegistrableClient | undefined {
  return instances.get(key);
}

/**
 * Claim `key` for `client`, or hand back whoever already holds it.
 *
 * A separate `get` then `set` would let two callers constructing the first
 * client for one identity both miss the lookup, with the second overwriting
 * the first. JavaScript's single-threaded execution makes that impossible
 * between two synchronous statements, so this is a single operation for
 * clarity rather than for locking: there is no equivalent of Python's mutex
 * to hold, and none is needed.
 *
 * Returns `undefined` when `client` won the slot, or the incumbent when not.
 */
export function registerIfAbsent(
  key: string,
  client: RegistrableClient,
): RegistrableClient | undefined {
  const existing = instances.get(key);
  if (existing !== undefined) return existing;
  instances.set(key, client);
  return undefined;
}

/**
 * Point `key` at `client` as well, unless the slot is taken.
 *
 * Used once `initialize()` resolves the real instance id. The client was keyed
 * on its credential because the id did not exist yet, so a later
 * `new SynapClient({ instanceId })` for that same instance would miss and
 * build a second client: two anticipation caches and two Listen streams for
 * one instance. The credential slot is KEPT rather than moved, so constructing
 * by API key keeps returning the same client too.
 *
 * An occupied target is never clobbered. Evicting a live client is worse than
 * the duplication it would fix.
 */
export function aliasIfAbsent(key: string, client: RegistrableClient): boolean {
  if (!key || instances.has(key)) return false;
  instances.set(key, client);
  return true;
}

/**
 * Drop `key`, but only if it still points at `client`.
 *
 * A shutting-down client must not evict whoever currently holds its old slot.
 */
export function unregisterIfOwner(key: string, client: RegistrableClient): void {
  if (instances.get(key) === client) instances.delete(key);
}

/** Test seam. Never call this from library code. */
export function clear(): void {
  instances.clear();
}

/** Diagnostics only. */
export function size(): number {
  return instances.size;
}
