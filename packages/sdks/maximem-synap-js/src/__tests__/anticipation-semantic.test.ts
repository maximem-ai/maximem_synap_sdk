/**
 * Matching fetch queries to prefetched items by meaning (21 Oct 2026).
 * Python: synap/tests/sdk/test_anticipation_semantic_match.py, the same cases.
 * The encoder is a fake with fixed vectors, so each test states exactly how
 * close a query and an item are. Measurements: harnesses/anticipation_match.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { AnticipationCache, canonicalQuery } from '../context/anticipation-cache.js';
import { fetchContext } from '../context/fetch.js';
import { SEMANTIC_MATCH } from '../behavior/contract.js';
import { parseModelId, loadOnnxEncoder } from '../cache/onnx-encoder.js';

const unit = (...xs: number[]) => {
  const n = Math.sqrt(xs.reduce((s, x) => s + x * x, 0));
  return xs.map((x) => x / n);
};

const PHARMACY = "User's pharmacy is Maple Pharmacy on 12 Elm Street.";
const REFILL = "User's Lisinopril refill is due on 15 November.";
const SEATS = 'User prefers aisle seats.';
const Q_WHERE = 'where do I usually pick up my pills?';
const Q_CLOSE = 'what time does Maple Pharmacy close tonight?';

const VECTORS: Record<string, number[]> = {
  [PHARMACY]: unit(1, 0.1, 0, 0),
  [REFILL]: unit(0.6, 0.8, 0, 0),
  [SEATS]: unit(0, 0, 1, 0),
  [Q_WHERE]: unit(1, 0.3, 0, 0),
  [Q_CLOSE]: unit(0.5, 0, 0.1, 0.86),
};

function fake(vectors: Record<string, number[]> = VECTORS, fail: string[] = []) {
  return {
    model: 'fake',
    calls: 0,
    encode(texts: string[]) {
      this.calls++;
      if (texts.some((t) => fail.includes(t))) throw new Error('model gone');
      return texts.map((t) => vectors[t] ?? [0, 0, 0, 1]);
    },
  };
}

async function cache(encoder: ReturnType<typeof fake> | null, extra: { askedKind?: boolean } = {}) {
  const c = new AnticipationCache({ encoder, matcher: 'semantic', minCosine: 0.6, serveMinCosine: 0.4, ...extra });
  for (const user of ['alice', 'bob']) {
    c.store({
      bundleId: `${user}-1`, entityId: user,
      itemsByType: {
        facts: [{ item_id: `${user}-p`, content: PHARMACY }, { item_id: `${user}-r`, content: REFILL }],
        preferences: [{ item_id: `${user}-s`, content: SEATS }],
      },
    });
  }
  await c.whenEmbedded();
  return c;
}

const ids = (r: Awaited<ReturnType<AnticipationCache['lookupAsync']>>) =>
  new Set(Object.values(r?.itemsByType ?? {}).flat().map((i) => i.item_id));

afterEach(() => { delete process.env['SYNAP_SDK_CACHE_MATCHER']; });

describe('anticipation cache: matching by meaning', () => {
  it('a paraphrase with no shared word hits', async () => {
    const c = await cache(fake());
    expect(ids(await c.lookupAsync({ searchQuery: [Q_WHERE], entityId: 'alice' })).has('alice-p')).toBe(true);
    expect(new AnticipationCache({ matcher: 'bm25' }).lookup({ searchQuery: [Q_WHERE], entityId: 'alice' })).toBeNull();
  });

  it('a near miss under the bar goes to the server', async () => {
    const c = await cache(fake());
    expect(await c.lookupAsync({ searchQuery: [Q_CLOSE], entityId: 'alice' })).toBeNull();
  });

  it('a hit also serves items above the serve bar but not below', async () => {
    const c = await cache(fake());
    expect(ids(await c.lookupAsync({ searchQuery: [Q_WHERE], entityId: 'alice' }))).toEqual(new Set(['alice-p', 'alice-r']));
  });

  it("another user's items are never served by meaning", async () => {
    const c = await cache(fake());
    const got = ids(await c.lookupAsync({ searchQuery: [Q_WHERE], entityId: 'alice' }));
    expect([...got].some((i) => String(i).startsWith('bob-'))).toBe(false);
    expect(await c.lookupAsync({ searchQuery: [Q_WHERE], entityId: 'carol' })).toBeNull();
  });

  it('items not embedded yet are matched by words', async () => {
    const c = await cache(null);
    expect(c.semantic).toBe(false);
    expect(ids(await c.lookupAsync({ searchQuery: ['Lisinopril refill due'], entityId: 'alice' })).has('alice-r')).toBe(true);
  });

  it('setEncoder embeds what is already cached', async () => {
    const c = await cache(null);
    const enc = fake();
    c.setEncoder(enc);
    await c.whenEmbedded();
    expect(enc.calls).toBe(1);
    expect(ids(await c.lookupAsync({ searchQuery: [Q_WHERE], entityId: 'alice' })).has('alice-p')).toBe(true);
  });

  it('a failing encoder falls back to words', async () => {
    const c = await cache(fake(VECTORS, [Q_WHERE, 'Lisinopril refill due']));
    expect(await c.lookupAsync({ searchQuery: [Q_WHERE], entityId: 'alice' })).toBeNull();
    expect(ids(await c.lookupAsync({ searchQuery: ['Lisinopril refill due'], entityId: 'alice' })).has('alice-r')).toBe(true);
  });

  it('the bm25 setting keeps word matching', () => {
    process.env['SYNAP_SDK_CACHE_MATCHER'] = 'bm25';
    expect(new AnticipationCache({ encoder: fake() }).semantic).toBe(false);
  });

  it('asked detail misses when no served item holds one', async () => {
    // Asked for a time of day; the cached items hold an address and a date.
    const q = 'what time does the pharmacy close';
    const withCheck = await cache(fake({ ...VECTORS, [q]: unit(1, 0.2, 0, 0) }), { askedKind: true });
    expect(await withCheck.lookupAsync({ searchQuery: [q], entityId: 'alice' })).toBeNull();
    const without = await cache(fake({ ...VECTORS, [q]: unit(1, 0.2, 0, 0) }), { askedKind: false });
    expect(await without.lookupAsync({ searchQuery: [q], entityId: 'alice' })).not.toBeNull();
  });
});

describe('the local model', () => {
  it('is named in the contract, pinned to a revision', () => {
    const ref = parseModelId(SEMANTIC_MATCH.model);
    expect(ref).not.toBeNull();
    expect(ref!.revision).toMatch(/^[0-9a-f]{40}$/);
  });

  it('loads nothing for a name of another kind', async () => {
    expect(await loadOnnxEncoder('fastembed:some/model')).toBeNull();
  });
});

// -- HQEs, pronouns, same-bundle serving, the borderline band (21 Oct 2026) --
// Python: the same cases in synap/tests/sdk/test_anticipation_semantic_match.py.

const LANDLORD = "User's landlord is Michael Trent.";
const Q_LANDLORD = 'Who is my landlord?';

async function one(vectors: Record<string, number[]>, item: Record<string, unknown>, extra: Record<string, unknown> = {}) {
  const c = new AnticipationCache({
    encoder: fake(vectors), matcher: 'semantic', minCosine: 0.6, serveMinCosine: 0.4, sureMinCosine: 0.74, ...extra,
  });
  c.store({ bundleId: 'b1', entityId: 'alice', itemsByType: { facts: [item as never] } });
  await c.whenEmbedded();
  return c;
}

describe('anticipation cache: HQEs, pronouns, bundles, bands', () => {
  it("a question matches the memory's stored questions", async () => {
    const v = {
      [LANDLORD]: unit(1, 0, 0, 0), [Q_LANDLORD]: unit(0.1, 1, 0, 0),
      [canonicalQuery(Q_LANDLORD)]: unit(0.3, 0.2, 1, 0), 'Who is my landlord? ': unit(0, 1, 0, 0),
    };
    const item = { item_id: 'l', content: LANDLORD, questions: ['Who is my landlord? '] };
    expect(await (await one(v, item, { itemQuestions: true })).lookupAsync({ searchQuery: [Q_LANDLORD], entityId: 'alice' })).not.toBeNull();
    expect(await (await one(v, item, { itemQuestions: false })).lookupAsync({ searchQuery: [Q_LANDLORD], entityId: 'alice' })).toBeNull();
  });

  it('canonicalQuery puts the asker in the third person', () => {
    expect(canonicalQuery('Who is my landlord?')).toBe("Who is the user's landlord?");
    expect(canonicalQuery("I'm moving and I have two cats")).toBe('the user is moving and the user has two cats');
    expect(canonicalQuery('Where did I park?')).toBe('Where did the user park?');
  });

  it('both forms of the query are compared', async () => {
    const v = { [LANDLORD]: unit(1, 0, 0, 0), [Q_LANDLORD]: unit(0, 1, 0, 0), [canonicalQuery(Q_LANDLORD)]: unit(1, 0.1, 0, 0) };
    const item = { item_id: 'l', content: LANDLORD };
    expect(await (await one(v, item, { canonical: 'both' })).lookupAsync({ searchQuery: [Q_LANDLORD], entityId: 'alice' })).not.toBeNull();
    expect(await (await one(v, item, { canonical: 'off' })).lookupAsync({ searchQuery: [Q_LANDLORD], entityId: 'alice' })).toBeNull();
  });

  it('a hit serves its bundle mates at the lower bar only', async () => {
    const v = { [PHARMACY]: unit(1, 0, 0, 0), [REFILL]: unit(0.35, 0.94, 0, 0), [SEATS]: unit(0.35, 0, 0.94, 0), [Q_WHERE]: unit(1, 0, 0, 0) };
    const c = new AnticipationCache({ encoder: fake(v), matcher: 'semantic', minCosine: 0.6, serveMinCosine: 0.4, bundleServeMinCosine: 0.3 });
    c.store({ bundleId: 'b1', entityId: 'alice', itemsByType: { facts: [{ item_id: 'p', content: PHARMACY }, { item_id: 'r', content: REFILL }] } });
    c.store({ bundleId: 'b2', entityId: 'alice', itemsByType: { facts: [{ item_id: 's', content: SEATS }] } });
    await c.whenEmbedded();
    expect(ids(await c.lookupAsync({ searchQuery: [Q_WHERE], entityId: 'alice' }))).toEqual(new Set(['p', 'r']));
  });

  it('a match below the sure bar is borderline', async () => {
    const v = { [PHARMACY]: unit(1, 0, 0, 0), close: unit(0.7, 0.71, 0, 0), exact: unit(1, 0.05, 0, 0) };
    const item = { item_id: 'p', content: PHARMACY };
    expect((await (await one(v, item, { canonical: 'off' })).lookupAsync({ searchQuery: ['close'], entityId: 'alice' }))?.band).toBe('borderline');
    expect((await (await one(v, item, { canonical: 'off' })).lookupAsync({ searchQuery: ['exact'], entityId: 'alice' }))?.band).toBe('sure');
  });
});

describe('fetch: a borderline match races the server', () => {
  const cached = { facts: [{ content: 'cached' }] };
  const transport = (delayMs: number) => ({
    checkCustomerId: () => undefined,
    request: () => new Promise((resolve) => setTimeout(() => resolve({ context: { facts: [{ content: 'server' }] } }), delayMs)),
  });
  const hooks = (borderline: boolean) => ({
    beforeFetch: async () => ({ response: cached, servedItemIds: [], bundleId: 'b', borderline, raceDeadlineMs: 50 }),
  });
  const first = (r: Record<string, unknown>) => ((r['facts'] as { content: string }[])[0]!).content;

  it("takes the server's answer when it is in time", async () => {
    expect(first(await fetchContext(transport(5) as never, 'user', { user_id: 'u' }, hooks(true) as never))).toBe('server');
  });
  it('serves the cached match when the server is late', async () => {
    expect(first(await fetchContext(transport(300) as never, 'user', { user_id: 'u' }, hooks(true) as never))).toBe('cached');
  });
  it('does not race a sure match', async () => {
    expect(first(await fetchContext(transport(5) as never, 'user', { user_id: 'u' }, hooks(false) as never))).toBe('cached');
  });
});
