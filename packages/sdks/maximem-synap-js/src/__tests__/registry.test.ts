import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SynapClient } from '../client.js';
import * as registry from '../registry.js';

/**
 * One client per identity, per process.
 *
 * The reason this exists is billing, not tidiness. Python has shared state
 * here since 0.2; without it a framework integration that builds a client per
 * request or per agent gets a cold anticipation cache every time and pays for
 * a retrieval that Python would have served locally.
 */

const json = (body: unknown) =>
  new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });
const stub = (async () => json({})) as unknown as typeof fetch;
const INSTANCE_A = 'inst_0123456789abcdef';
const INSTANCE_B = 'inst_fedcba9876543210';

beforeEach(() => { registry.clear(); });
afterEach(() => { registry.clear(); });

describe('client registry', () => {
  it('hands back the same client for the same API key', () => {
    const a = new SynapClient({ apiKey: 'key-one', fetchImpl: stub });
    const b = new SynapClient({ apiKey: 'key-one', fetchImpl: stub });
    expect(b).toBe(a);
  });

  it('keeps different API keys apart', () => {
    const a = new SynapClient({ apiKey: 'key-one', fetchImpl: stub });
    const b = new SynapClient({ apiKey: 'key-two', fetchImpl: stub });
    expect(b).not.toBe(a);
    expect(registry.size()).toBe(2);
  });

  it('hands back the same client for the same explicit instance id', () => {
    const a = new SynapClient({ apiKey: 'k', instanceId: INSTANCE_A, fetchImpl: stub });
    const b = new SynapClient({ apiKey: 'k', instanceId: INSTANCE_A, fetchImpl: stub });
    expect(b).toBe(a);
  });

  it('keeps different instance ids apart', () => {
    const a = new SynapClient({ apiKey: 'k', instanceId: INSTANCE_A, fetchImpl: stub });
    const b = new SynapClient({ apiKey: 'k', instanceId: INSTANCE_B, fetchImpl: stub });
    expect(b).not.toBe(a);
  });

  it('shares the anticipation cache, which is the point', async () => {
    // The billing fix: a second client for the same identity must see what the
    // first cached, or every construction starts cold and every lookup is a
    // metered fetch.
    const a = new SynapClient({ apiKey: 'shared-key', fetchImpl: stub });
    const b = new SynapClient({ apiKey: 'shared-key', fetchImpl: stub });
    a.cache.clear();
    expect(a.anticipation_cache_snapshot().total_entries).toBe(0);
    // Same object, so trivially the same cache. Asserted through the public
    // surface rather than by identity, because that is what a caller observes.
    expect(b.anticipation_cache_snapshot()).toEqual(a.anticipation_cache_snapshot());
    await a.shutdown();
  });

  it('_force_new opts out entirely', () => {
    const a = new SynapClient({ apiKey: 'key-one', fetchImpl: stub });
    const b = new SynapClient({ apiKey: 'key-one', fetchImpl: stub, _force_new: true });
    expect(b).not.toBe(a);
    // The opted-out client never claims a slot, so it cannot be handed to
    // anyone else either.
    const c = new SynapClient({ apiKey: 'key-one', fetchImpl: stub });
    expect(c).toBe(a);
  });

  it('warns when an existing instance is asked for on a different key', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      new SynapClient({ apiKey: 'original', instanceId: INSTANCE_A, fetchImpl: stub });
      const second = new SynapClient({ apiKey: 'different', instanceId: INSTANCE_A, fetchImpl: stub });
      expect(second.instance_id).toBe(INSTANCE_A);
      // The caller has no other way to learn their key was ignored: the object
      // they get back looks exactly like the one they asked for.
      expect(warn).toHaveBeenCalledWith(expect.stringContaining('different'));
      expect(warn).toHaveBeenCalledWith(expect.stringContaining('Rotating a key this way has no effect'));
    } finally {
      warn.mockRestore();
    }
  });

  it('does not warn when the key matches', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      new SynapClient({ apiKey: 'same', instanceId: INSTANCE_A, fetchImpl: stub });
      new SynapClient({ apiKey: 'same', instanceId: INSTANCE_A, fetchImpl: stub });
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it('aliases the resolved instance id after initialize', async () => {
    const fetchImpl = (async () =>
      json({ client_id: 'cli_x', instance_id: INSTANCE_A })) as unknown as typeof fetch;
    const byKey = new SynapClient({ apiKey: 'aliasing-key', fetchImpl });
    await byKey.initialize();
    expect(byKey.instance_id).toBe(INSTANCE_A);

    // Constructing by the now-known instance id must find the same client, or
    // you get two anticipation caches and two Listen streams for one instance.
    const byId = new SynapClient({ apiKey: 'aliasing-key', instanceId: INSTANCE_A, fetchImpl });
    expect(byId).toBe(byKey);

    // The credential slot is kept, not moved.
    const byKeyAgain = new SynapClient({ apiKey: 'aliasing-key', fetchImpl });
    expect(byKeyAgain).toBe(byKey);
    await byKey.shutdown();
  });

  it('never clobbers an occupied alias target', async () => {
    const occupant = new SynapClient({ apiKey: 'occupant', instanceId: INSTANCE_A, fetchImpl: stub });
    const fetchImpl = (async () =>
      json({ client_id: 'cli_x', instance_id: INSTANCE_A })) as unknown as typeof fetch;
    const other = new SynapClient({ apiKey: 'other-key', fetchImpl });
    await other.initialize();
    // Evicting a live client is worse than the duplication it would fix.
    expect(new SynapClient({ apiKey: 'x', instanceId: INSTANCE_A, fetchImpl: stub })).toBe(occupant);
    await occupant.shutdown();
    await other.shutdown();
  });

  it('releases its slots on shutdown', async () => {
    const a = new SynapClient({ apiKey: 'transient', fetchImpl: stub });
    await a.shutdown();
    // A shut-down client holds closed transports; handing it to a new caller
    // would give them something that cannot make a request.
    const b = new SynapClient({ apiKey: 'transient', fetchImpl: stub });
    expect(b).not.toBe(a);
    await b.shutdown();
  });

  it('keeps the raw credential out of the slot name', () => {
    const key = 'synap_super_secret_value';
    const slot = registry.buildRegistryKey('', key);
    expect(slot).not.toContain(key);
    expect(slot).toMatch(/^apikey:[0-9a-f]{16}$/);
    // Stable, or a second construction would miss its own slot.
    expect(registry.buildRegistryKey('', key)).toBe(slot);
    expect(registry.buildRegistryKey('', 'a-different-key')).not.toBe(slot);
  });

  it('prefers an explicit instance id over the credential', () => {
    expect(registry.buildRegistryKey(INSTANCE_A, 'anything')).toBe(INSTANCE_A);
    // Nothing to key on: the client cannot initialise without a credential
    // anyway, so the empty slot is preserved rather than invented.
    expect(registry.buildRegistryKey('', undefined)).toBe('');
  });
});
