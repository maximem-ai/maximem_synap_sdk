import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SynapClient, type SynapSdkLike } from '../index.js';
import { InvalidInputError } from '../errors.js';
import { buildFetchRequest } from '../context/fetch.js';
import { buildCreateBody } from '../memories/interface.js';

const ENV = ['SYNAP_API_KEY', 'SYNAP_CLIENT_ID', 'SYNAP_INSTANCE_ID'];
beforeEach(() => { for (const k of ENV) delete process.env[k]; });
afterEach(() => { for (const k of ENV) delete process.env[k]; });

const json = (body: unknown) =>
  new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } });

// A real-shaped instance id: `inst_` + 16 hex. The constructor validates the
// format now, so the old 'i' placeholder is rejected before any request.
const INSTANCE_ID = 'inst_0123456789abcdef';

function client(fetchImpl: typeof fetch) {
  return new SynapClient({ _force_new: true, apiKey: 'k', clientId: 'c', instanceId: INSTANCE_ID, fetchImpl });
}

describe('SynapClient', () => {
  describe('credentials', () => {
    it('requires an API key', () => {
      expect(() => new SynapClient({ _force_new: true })).toThrow(InvalidInputError);
      expect(() => new SynapClient({ _force_new: true })).toThrow(/SYNAP_API_KEY/);
    });

    it('reads env at construction time, not module load', () => {
      process.env['SYNAP_API_KEY'] = 'from-env';
      process.env['SYNAP_CLIENT_ID'] = 'cid';
      expect(() => new SynapClient({ _force_new: true })).not.toThrow();
    });

    it('prefers explicit options over env', async () => {
      process.env['SYNAP_API_KEY'] = 'from-env';
      const spy = vi.fn(async () => json({ context: {} }));
      await client(spy as unknown as typeof fetch).user.context.fetch({ user_id: 'u' });
      const h = (spy.mock.calls[0] as unknown as [string, RequestInit])[1].headers as Record<string, string>;
      expect(h['Authorization']).toBe('Bearer k');
    });
  });

  describe('namespaced surface returns RAW snake_case', () => {
    it('passes result.context through untouched', async () => {
      const raw = { facts: [{ id: 'f1', content: 'x', extracted_at: '2026-01-01', temporal_category: 'past' }] };
      const c = client((async () => json({ context: raw })) as unknown as typeof fetch);
      const out = await c.user.context.fetch({ user_id: 'u' });
      // snake_case survives; nothing is renamed or defaulted.
      expect(out).toEqual(raw);
      expect(out.facts![0]!['extracted_at']).toBe('2026-01-01');
    });

    it('returns {} when the envelope has no context', async () => {
      const c = client((async () => json({})) as unknown as typeof fetch);
      expect(await c.user.context.fetch({ user_id: 'u' })).toEqual({});
    });
  });

  describe('deprecated camelCase surface returns the NORMALISED shape', () => {
    it('renames and defaults', async () => {
      const c = client((async () => json({
        context: { facts: [{ id: 'f1', content: 'x', extracted_at: '2026-01-01' }] },
      })) as unknown as typeof fetch);
      const out = await c.fetchUserContext({ userId: 'u' });
      expect(out.facts[0]!.extractedAt).toBe('2026-01-01');
      expect(out.facts[0]!.confidence).toBe(0);
      expect(out.metadata.source).toBe('unknown');
      expect(out.conversationContext).toBeNull();
      // The two surfaces genuinely differ. That is deliberate (G-B).
      expect(out).not.toHaveProperty('facts.0.extracted_at');
    });

    it('preserves an explicit null rather than applying the default', async () => {
      const c = client((async () => json({
        context: { temporal_events: [{ id: 't', content: 'x', temporal_category: null }] },
      })) as unknown as typeof fetch);
      const out = await c.fetchUserContext({ userId: 'u' });
      expect(out.temporalEvents[0]!.temporalCategory).toBeNull();
    });

    it('falls back across alternative server field names', async () => {
      const c = client((async () => json({
        context: {
          preferences: [{ id: 'p', content: 'aisle', confidence: 0.9 }],
          episodes: [{ id: 'e', content: 'summary text', confidence: 0.5 }],
          emotions: [{ id: 'm', emotion_type: 'joy', confidence: 0.7 }],
        },
      })) as unknown as typeof fetch);
      const out = await c.fetchUserContext({ userId: 'u' });
      expect(out.preferences[0]!.strength).toBe(0.9);   // strength <- confidence
      expect(out.episodes[0]!.summary).toBe('summary text'); // summary <- content
      expect(out.emotions[0]!.intensity).toBe(0.7);     // intensity <- confidence
      expect(out.emotions[0]!.emotionType).toBe('joy');
    });
  });

  describe('argument spelling', () => {
    it('accepts snake_case and camelCase interchangeably', () => {
      const a = buildFetchRequest('user', { user_id: 'u', customer_id: 'c', max_results: 5 });
      const b = buildFetchRequest('user', { userId: 'u', customerId: 'c', maxResults: 5 });
      expect(a).toEqual(b);
    });

    it('lets snake_case win when both are given', () => {
      const r = buildFetchRequest('user', { user_id: 'snake', userId: 'camel' });
      expect(r.body['user_id']).toBe('snake');
    });
  });

  describe('request building per scope', () => {
    it('picks the right endpoint', () => {
      expect(buildFetchRequest('user', { user_id: 'u' }).endpoint).toBe('context_fetch_user');
      expect(buildFetchRequest('customer', { customer_id: 'c' }).endpoint).toBe('context_fetch_customer');
      expect(buildFetchRequest('client', {}).endpoint).toBe('context_fetch_client');
      expect(buildFetchRequest('conversation', { conversation_id: 'v' }).endpoint).toBe('context_fetch_conversation');
    });

    it('matches the Python payload field for field', () => {
      // The server is tuned against what the Python controllers post, and
      // Python is the reference implementation. A key we omit that Python
      // always sends (or vice versa) shows up as "works in Python, empty in
      // JS", which is expensive to diagnose.
      const body = buildFetchRequest('user', { user_id: 'u' }).body;
      expect(body).toEqual({
        conversation_id: null,   // always present, null when unset
        search_query: [],
        max_results: 10,
        types: ['all'],
        mode: 'fast',            // always defaulted
        user_id: 'u',
      });
      // customer_id is passed through only when supplied, as Python does.
      expect('customer_id' in body).toBe(false);
      // precision_level and include_conversation_context are omitted at their
      // defaults, again matching Python.
      expect('precision_level' in body).toBe(false);
      expect('include_conversation_context' in body).toBe(false);
    });

    it('sends the conditional keys only when they differ from the default', () => {
      const p = buildFetchRequest('user', { user_id: 'u', precision_level: 'medium' }).body;
      expect(p['precision_level']).toBe('medium');

      const i = buildFetchRequest('user', { user_id: 'u', include_conversation_context: false }).body;
      expect(i['include_conversation_context']).toBe(false);

      // Explicitly true is the default, so it stays omitted.
      const t = buildFetchRequest('user', { user_id: 'u', include_conversation_context: true }).body;
      expect('include_conversation_context' in t).toBe(false);
    });

    it('passes customer_id through when supplied', () => {
      expect(buildFetchRequest('user', { user_id: 'u', customer_id: 'c' }).body['customer_id']).toBe('c');
      expect(buildFetchRequest('user', { user_id: 'u', customer_id: '' }).body['customer_id']).toBe('');
    });

    it('enforces the required id per scope', () => {
      expect(() => buildFetchRequest('user', {})).toThrow(/user_id is required/);
      expect(() => buildFetchRequest('customer', {})).toThrow(/customer_id is required/);
      expect(() => buildFetchRequest('conversation', {})).toThrow(/conversation_id is required/);
      expect(() => buildFetchRequest('client', {})).not.toThrow();
    });

    it('rejects non-array search_query and types', () => {
      expect(() => buildFetchRequest('client', { search_query: 'oops' as unknown as string[] }))
        .toThrow(/must be an array/);
      expect(() => buildFetchRequest('client', { types: 'oops' as unknown as string[] }))
        .toThrow(/must be an array/);
    });

    it('applies the documented defaults', () => {
      const b = buildFetchRequest('client', {}).body;
      expect(b['max_results']).toBe(10);
      expect(b['types']).toEqual(['all']);
      expect(b['search_query']).toEqual([]);
    });

    it('treats conversation_id as a narrowing filter, not a scope tier', () => {
      // Scopes are client > customer > user. conversation_id groups turns.
      const b = buildFetchRequest('user', { user_id: 'u', conversation_id: 'v' });
      expect(b.endpoint).toBe('context_fetch_user');
      expect(b.body['conversation_id']).toBe('v');
    });
  });

  describe('memories', () => {
    it('requires a document', () => {
      expect(() => buildCreateBody({ document: '' })).toThrow(/document is required/);
    });

    it('sends every field with Python defaults applied', () => {
      // Python posts CreateMemoryRequest.model_dump(mode="json"), which emits
      // all fields including nulls. Omitting them would send the server a
      // different request; document_type in particular drives which extraction
      // path runs, so a missing one silently degrades what gets stored.
      expect(buildCreateBody({ document: 'x' })).toEqual({
        document: 'x',
        document_type: 'ai-chat-conversation',
        document_id: null,
        document_created_at: null,
        user_id: null,
        customer_id: null,
        mode: 'long-range',
        metadata: {},
      });
    });

    it('maps both spellings to the wire name', () => {
      const b = buildCreateBody({ document: 'x', userId: 'u', documentType: 'meeting-transcript' });
      expect(b['user_id']).toBe('u');
      expect(b['document_type']).toBe('meeting-transcript');
    });

    it('serialises a Date the way Python serialises a datetime', () => {
      const d = new Date('2026-08-12T10:00:00.000Z');
      expect(buildCreateBody({ document: 'x', document_created_at: d })['document_created_at'])
        .toBe('2026-08-12T10:00:00.000Z');
      expect(buildCreateBody({ document: 'x', document_created_at: '2026-08-12T10:00:00+00:00' })['document_created_at'])
        .toBe('2026-08-12T10:00:00+00:00');
    });

    it('rejects an invalid document_type or mode before sending', () => {
      // Python's enum coercion raises before any request goes out. Passing a
      // typo through would either 500 or silently take a different path.
      expect(() => buildCreateBody({ document: 'x', document_type: 'conversation' }))
        .toThrow(/Invalid document_type/);
      expect(() => buildCreateBody({ document: 'x', mode: 'accurate' }))
        .toThrow(/Invalid mode/);
    });

    it('refuses delete-by-user with an explanatory error', async () => {
      const c = client((async () => json({})) as unknown as typeof fetch);
      await expect(c.memories.delete('')).rejects.toThrow(/memory_id is required/);
      await expect(c.memories.delete('')).rejects.toThrow(/reporting success/);
    });

    it('deletes by id', async () => {
      const spy = vi.fn(async () => json({ deleted: true }));
      await client(spy as unknown as typeof fetch).memories.delete('mem-1');
      expect(spy.mock.calls[0]![0]).toContain('/api/v1/memories/mem-1');
      expect((spy.mock.calls[0] as unknown as [string, RequestInit])[1].method).toBe('DELETE');
    });

    it('updates with PATCH and omits unset fields', async () => {
      const spy = vi.fn(async () => json({ ok: true }));
      await client(spy as unknown as typeof fetch).memories.update({ memory_id: 'm1', document: 'new' });
      const [url, init] = spy.mock.calls[0] as unknown as [string, RequestInit];
      expect(url).toContain('/api/v1/memories/m1');
      expect(init.method).toBe('PATCH');
      expect(JSON.parse(init.body as string)).toEqual({ document: 'new', merge_strategy: 'smart-merge' });
    });

    it('batch_create rejects an empty list', async () => {
      const c = client((async () => json({})) as unknown as typeof fetch);
      await expect(c.memories.batch_create({ documents: [] })).rejects.toThrow(/non-empty array/);
    });

    it('surfaces validation failures as rejections, never as sync throws', async () => {
      // These methods return Promises, so a synchronous throw would bypass
      // .catch() and any surrounding try/await. Python raises inside a
      // coroutine, which rejects; this matches that.
      const c = client((async () => json({})) as unknown as typeof fetch);
      const calls: Promise<unknown>[] = [
        c.memories.get(''),
        c.memories.status(''),
        c.memories.delete(''),
        c.memories.update({ document: 'x' }),
        c.credits.redeem(''),
        c.user.get_profile(''),
        c.conversation.record_message({ role: 'user', content: 'x' }),
        c.conversation.ingest_transcript({ transcript: 'x' }),
        c.conversation.context.compact({}),
      ];
      for (const p of calls) {
        expect(p).toBeInstanceOf(Promise);
        await expect(p).rejects.toBeInstanceOf(InvalidInputError);
      }
    });

    it('wait_for_completion polls until terminal', async () => {
      let n = 0;
      const spy = vi.fn(async () => json({ status: ++n < 3 ? 'processing' : 'completed' }));
      const c = client(spy as unknown as typeof fetch);
      const out = await c.memories.wait_for_completion('ing-1', { poll_interval_seconds: 0 });
      expect(out['status']).toBe('completed');
      expect(spy).toHaveBeenCalledTimes(3);
    });

    it('wait_for_completion returns the last status on timeout rather than throwing', async () => {
      // Python returns the last status, so a caller that only reads .status
      // behaves identically in both SDKs.
      const spy = vi.fn(async () => json({ status: 'processing' }));
      const c = client(spy as unknown as typeof fetch);
      const out = await c.memories.wait_for_completion('ing-1', {
        timeout_seconds: 0, poll_interval_seconds: 0,
      });
      expect(out['status']).toBe('processing');
    });
  });

  describe('lifecycle', () => {
    it('init() is a no-op that still resolves', async () => {
      await expect(client((async () => json({})) as unknown as typeof fetch).init()).resolves.toBeUndefined();
    });

    it('shutdown() is idempotent', async () => {
      const c = client((async () => json({})) as unknown as typeof fetch);
      await c.shutdown();
      await expect(c.shutdown()).resolves.toBeUndefined();
    });

    it('does not register process-level exit handlers', () => {
      // The wrapper added beforeExit/SIGINT/SIGTERM listeners per client, which
      // leaked a listener each time, never fired in Lambda, and referenced a
      // `process` that does not exist in Workers (gotcha G-I).
      const before = process.listenerCount('beforeExit') + process.listenerCount('SIGINT') + process.listenerCount('SIGTERM');
      const clients = Array.from({ length: 12 }, () => client((async () => json({})) as unknown as typeof fetch));
      const after = process.listenerCount('beforeExit') + process.listenerCount('SIGINT') + process.listenerCount('SIGTERM');
      expect(after).toBe(before);
      expect(clients).toHaveLength(12);
    });
  });

  it('satisfies the duck-typed integration contract', () => {
    // Compile-enforced: if the client's surface drifts from what
    // synap-mastra / synap-eve / synap-claude-agent-ts rely on, this stops
    // compiling instead of failing at their runtime.
    const c: SynapSdkLike = client((async () => json({})) as unknown as typeof fetch);
    expect(typeof c.user.context.fetch).toBe('function');
    expect(typeof c.customer.context.fetch).toBe('function');
    expect(typeof c.client.context.fetch).toBe('function');
    expect(typeof c.memories.create).toBe('function');
  });
});
