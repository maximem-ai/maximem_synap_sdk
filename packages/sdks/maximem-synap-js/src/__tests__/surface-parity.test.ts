import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SynapClient } from '../client.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const surfacePath = path.resolve(here, '../../../CONTRACT/conformance/python_surface.json');

/**
 * Surface parity with the pip SDK.
 *
 * The point of this test is that a gap must be *declared*, not discovered. If
 * a method exists in Python and not here, either implement it or add it to
 * KNOWN_GAPS with a reason. Anything else fails, so parity cannot rot quietly
 * into a customer-facing TypeError.
 */
const KNOWN_GAPS: Record<string, string> = {};

/**
 * JS members with no Python counterpart. Every one must be listed with a
 * reason, or the reverse-direction test fails.
 *
 * This half of the check did not exist for the ROOT namespace, and that is how
 * `transport` and `anticipationCache` (both leaking internals, one of them the
 * API key) plus an `init` that Python spells `initialize` all went unnoticed
 * while the suite stayed green.
 */
const ALLOWED_EXTRAS: Record<string, string> = {
  // The 0.3.x wrapper's flat camelCase surface. Deprecated but supported; see
  // the class docstring on why the two surfaces return different shapes.
  'fetchUserContext': '0.3.x wrapper compatibility',
  'fetchCustomerContext': '0.3.x wrapper compatibility',
  'fetchClientContext': '0.3.x wrapper compatibility',
  'getContextForPrompt': '0.3.x wrapper compatibility',
  'searchMemory': '0.3.x wrapper compatibility',
  'getMemories': '0.3.x wrapper compatibility',
  'addMemory': '0.3.x wrapper compatibility',
  'deleteMemory': '0.3.x wrapper compatibility',
  'init': 'Deprecated alias for initialize(), kept for 0.3.x startup scripts',
  // Python exposes `instance_id` as a plain attribute and has no public
  // client_id; this SDK exposes both as getters so the value resolved by
  // initialize() is readable.
  'instance_id': 'Mirrors Python\'s instance_id attribute',
  'client_id': 'Readable counterpart to instance_id; Python keeps it private',
};

/**
 * Public members as a consumer sees them: own enumerable keys PLUS the
 * prototype chain. `Object.keys` alone misses every class method, which is why
 * the root namespace looked empty and its extras went unchecked.
 */
function publicMembers(obj: object): string[] {
  const out = new Set<string>();
  for (const k of Object.keys(obj)) if (!k.startsWith('_')) out.add(k);
  let proto: object | null = Object.getPrototypeOf(obj);
  while (proto !== null && proto !== Object.prototype) {
    for (const k of Object.getOwnPropertyNames(proto)) {
      if (k !== 'constructor' && !k.startsWith('_')) out.add(k);
    }
    proto = Object.getPrototypeOf(proto);
  }
  return [...out].sort();
}

function resolveNamespace(client: SynapClient, dotted: string): Record<string, unknown> | undefined {
  // The root namespace ("") IS the client.
  if (dotted === '') return client as unknown as Record<string, unknown>;
  let cur: unknown = client;
  for (const part of dotted.split('.')) {
    if (cur === null || typeof cur !== 'object') return undefined;
    cur = (cur as Record<string, unknown>)[part];
  }
  return cur === null || typeof cur !== 'object' ? undefined : (cur as Record<string, unknown>);
}

describe('surface parity with the Python SDK', () => {
  if (!existsSync(surfacePath)) return;
  const golden = JSON.parse(readFileSync(surfacePath, 'utf8')) as {
    python_version: string;
    namespaces: Record<string, string[]>;
  };
  const client = new SynapClient({ _force_new: true, apiKey: 'k', fetchImpl: (async () => new Response('{}')) as unknown as typeof fetch });

  const entries = Object.entries(golden.namespaces).flatMap(([ns, methods]) =>
    methods.map((m) => ({ ns, m, key: ns === '' ? m : `${ns}.${m}` })),
  );

  it.each(entries)('$key exists', ({ ns, m, key }) => {
    if (key in KNOWN_GAPS) {
      // Declared gap. Assert it really is still missing, so the entry gets
      // removed when someone implements it rather than lingering as a lie.
      const target = resolveNamespace(client, ns);
      expect(target?.[m], `${key} is implemented; remove it from KNOWN_GAPS`).toBeUndefined();
      return;
    }
    const target = resolveNamespace(client, ns);
    expect(target, `namespace client.${ns} is missing entirely`).toBeDefined();
    // A member may be a method or a property: Python's
    // InstanceInterface.is_listening is the latter.
    expect(target !== undefined && m in target, `client.${key} is missing`).toBe(true);
  });

  it('covers every Python namespace', () => {
    for (const ns of Object.keys(golden.namespaces)) {
      expect(resolveNamespace(client, ns), `client.${ns} is missing`).toBeDefined();
    }
  });

  it('exposes no extra public methods on a namespace', () => {
    // Guards the reverse direction: an accidentally leaked internal (this
    // caught `client.cache.now`, a private field reachable at runtime) means
    // the JS surface silently differs from Python's.
    const extras: string[] = [];
    for (const [ns, methods] of Object.entries(golden.namespaces)) {
      const target = resolveNamespace(client, ns);
      if (!target) continue;
      for (const key of publicMembers(target)) {
        const v = (target as Record<string, unknown>)[key];
        const isSubNamespace = typeof v === 'object' && v !== null;
        if (isSubNamespace) continue;
        // Properties count too: Python's InstanceInterface.is_listening is a
        // property, so a callable-only comparison would let a naming
        // divergence through.
        const dotted = ns === '' ? key : `${ns}.${key}`;
        if (!methods.includes(key) && ALLOWED_EXTRAS[dotted] === undefined) extras.push(dotted);
      }
    }
    expect(extras).toEqual([]);
  });

  it('leaks no internals from the client root', () => {
    // `transport` exposed `credentials.apiKey`, and `anticipationCache`
    // exposed the raw cache that `client.cache` exists to wrap. Both were
    // public `readonly` fields; they are now `#private`, which -- unlike
    // TypeScript's `private` -- is actually absent at runtime.
    for (const forbidden of ['transport', 'anticipationCache', 'closed']) {
      expect(
        Object.prototype.hasOwnProperty.call(client, forbidden),
        `client.${forbidden} must not be publicly reachable`,
      ).toBe(false);
      expect((client as unknown as Record<string, unknown>)[forbidden]).toBeUndefined();
    }
  });

  it('records which Python version this was generated from', () => {
    expect(golden.python_version).toMatch(/^\d+\.\d+\.\d+/);
  });
});
