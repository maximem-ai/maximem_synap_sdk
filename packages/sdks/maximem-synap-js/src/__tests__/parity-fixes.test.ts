import { describe, it, expect, vi, afterEach } from 'vitest';
import { SynapClient } from '../client.js';
import { HttpTransport } from '../transport/http.js';
import { AnticipationCache } from '../context/anticipation-cache.js';
import { CompactionSubscribers, createConversationNamespace } from '../conversation/interface.js';
import { validateConversationId, validateInstanceId } from '../util/validators.js';
import {
  InvalidConversationIdError, InvalidInstanceIdError, InvalidInputError,
  ConflictError, TranscriptConflictError,
} from '../errors.js';

const INSTANCE_ID = 'inst_0123456789abcdef';
const UUID = '3f2504e0-4f89-11d3-9a0c-0305e82c3301';
const CREDS = { apiKey: 'k', clientId: 'c', instanceId: INSTANCE_ID };

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

function client(fetchImpl: typeof fetch, options = {}) {
  return new SynapClient({ _force_new: true, apiKey: 'k', instanceId: INSTANCE_ID, fetchImpl, ...options });
}

describe('identifier validation, matching Python', () => {
  it('accepts a UUID conversation id and empty values', () => {
    expect(() => validateConversationId(UUID)).not.toThrow();
    // Python leaves empty/None untouched: callers that require an id say so
    // separately, so only a clearly malformed value is rejected.
    for (const empty of [undefined, null, '']) {
      expect(() => validateConversationId(empty)).not.toThrow();
    }
  });

  it('rejects a malformed conversation id with Python\'s message', () => {
    expect(() => validateConversationId('conv_123')).toThrow(InvalidConversationIdError);
    expect(() => validateConversationId('conv_123')).toThrow('Invalid conversation ID: conv_123');
    // Still an InvalidInputError, so an existing catch keeps working.
    expect(() => validateConversationId('conv_123')).toThrow(InvalidInputError);
  });

  it('accepts inst_ plus 16 hex and rejects everything else', () => {
    expect(() => validateInstanceId(INSTANCE_ID)).not.toThrow();
    expect(() => validateInstanceId('inst_ABCDEF0123456789')).not.toThrow();
    for (const bad of ['i', 'inst_123', 'inst_0123456789abcdeg', 'xxxx_0123456789abcdef']) {
      expect(() => validateInstanceId(bad), bad).toThrow(InvalidInstanceIdError);
    }
    expect(() => validateInstanceId('i')).toThrow('Invalid instance ID: i');
  });

  it('rejects a malformed instance id at construction, before any request', () => {
    const fetchImpl = vi.fn();
    expect(() => new SynapClient({ _force_new: true, apiKey: 'k', instanceId: 'nope', fetchImpl: fetchImpl as never }))
      .toThrow(InvalidInstanceIdError);
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it('rejects a malformed conversation id in the unified fetch without a request', async () => {
    const fetchImpl = vi.fn(async () => json({}));
    await expect(
      client(fetchImpl as never).fetch({ conversation_id: 'conv_123' }),
    ).rejects.toThrow(InvalidConversationIdError);
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it('validates in exactly the places Python does', async () => {
    const fetchImpl = vi.fn(async () => json({}));
    const c = client(fetchImpl as never);
    const bad = 'conv_123';

    // The seven Python call sites, all in the conversation namespace.
    await expect(c.conversation.record_message({
      conversation_id: bad, user_id: 'u', role: 'user', content: 'x',
    })).rejects.toThrow(InvalidConversationIdError);
    await expect(c.conversation.record_messages_batch([
      { conversation_id: bad, user_id: 'u', role: 'user', content: 'x' },
    ])).rejects.toThrow(InvalidConversationIdError);
    await expect(c.conversation.context.fetch({ conversation_id: bad }))
      .rejects.toThrow(InvalidConversationIdError);
    await expect(c.conversation.context.compact({ conversation_id: bad }))
      .rejects.toThrow(InvalidConversationIdError);
    await expect(c.conversation.context.get_compacted({ conversation_id: bad }))
      .rejects.toThrow(InvalidConversationIdError);
    await expect(c.conversation.context.get_compaction_status({ conversation_id: bad }))
      .rejects.toThrow(InvalidConversationIdError);
    await expect(c.conversation.context.get_context_for_prompt({ conversation_id: bad }))
      .rejects.toThrow(InvalidConversationIdError);

    expect(fetchImpl, 'nothing should have reached the network').not.toHaveBeenCalled();
  });

  it('does NOT validate where Python deliberately skips it', async () => {
    // ingest_transcript takes a free-form client string the server coerces
    // (spec 4.1), and user/customer/client context.fetch never validated it.
    // Tightening either would reject ids the pip SDK accepts.
    const fetchImpl = vi.fn(async () => json({ context: {} }));
    const c = client(fetchImpl as never);
    await expect(c.conversation.ingest_transcript({
      conversation_id: 'free-form-id', user_id: 'u', transcript: 'hello',
    })).resolves.toBeDefined();
    await expect(c.user.context.fetch({ user_id: 'u', conversation_id: 'free-form-id' }))
      .resolves.toBeDefined();
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });
});

describe('HTTP 409 discrimination', () => {
  async function errorFrom(body: unknown) {
    const transport = new HttpTransport({
      credentials: CREDS,
      retryPolicy: { maxAttempts: 1 },
      fetchImpl: (async () => json(body, 409)) as never,
    });
    return transport.request('conversations_compact', { body: {} }).catch((e) => e);
  }

  it('maps a transcript_conflict code to TranscriptConflictError', async () => {
    const error = await errorFrom({
      detail: { code: 'transcript_conflict', message: 'Turn 3 already recorded' },
    });
    expect(error).toBeInstanceOf(TranscriptConflictError);
    // Subclass of ConflictError, so an existing broad catch still fires.
    expect(error).toBeInstanceOf(ConflictError);
    expect(error.message).toContain('Turn 3 already recorded');
  });

  it('leaves any other 409 as the generic ConflictError', async () => {
    const error = await errorFrom({ detail: { code: 'already_in_progress', message: 'busy' } });
    expect(error).toBeInstanceOf(ConflictError);
    expect(error).not.toBeInstanceOf(TranscriptConflictError);
  });

  it('renders a structured detail object instead of [object Object]', async () => {
    const error = await errorFrom({ detail: { code: 'x', message: 'readable text' } });
    expect(error.message).toContain('readable text');
    expect(error.message).not.toContain('[object Object]');
  });

  it('still handles a plain string detail', async () => {
    const error = await errorFrom({ detail: 'just a string' });
    expect(error.message).toContain('just a string');
  });
});

describe('anticipation_cache_snapshot', () => {
  it('returns Python\'s empty shape for an untouched cache', () => {
    const snapshot = client((async () => json({})) as never).anticipation_cache_snapshot();
    expect(snapshot).toEqual({
      total_entries: 0,
      total_item_records: 0,
      scope_breakdown_overall: {},
      corpus_vocab_size: 0,
      corpus_vocab_sample: [],
      item_records: [],
      bundles: [],
    });
  });

  it('projects counts, scope breakdown and previews', () => {
    const cache = new AnticipationCache();
    cache.store({
      bundleId: 'b1', entityId: 'u1', conversationId: 'conv-1',
      bundleType: 'anticipation', searchQueries: ['q'],
      itemsByType: {
        facts: [
          { item_id: 'i1', content: 'alpha beta', scope: 'user' },
          { item_id: 'i2', content: 'gamma', scope: 'customer' },
        ],
      },
    });
    const s = cache.snapshot();
    expect(s.total_entries).toBe(1);
    expect(s.total_item_records).toBe(2);
    expect(s.scope_breakdown_overall).toEqual({ user: 1, customer: 1 });
    expect(s.corpus_vocab_size).toBeGreaterThan(0);
    expect(s.bundles[0]?.scope_counts).toEqual({ user: 1, customer: 1 });
    expect(s.bundles[0]?.total_items).toBe(2);
    expect(s.bundles[0]?.search_queries).toEqual(['q']);
    expect(s.item_records[0]?.bundle_id).toBe('b1');
  });

  it('truncates previews at 140 chars, not the telemetry cap of 120', () => {
    const cache = new AnticipationCache();
    cache.store({
      bundleId: 'b1', entityId: 'u1',
      itemsByType: { facts: [{ item_id: 'i', content: 'x'.repeat(200), scope: 'user' }] },
    });
    expect(cache.snapshot().item_records[0]?.content).toHaveLength(140);
    expect(cache.snapshot().bundles[0]?.items[0]?.content).toHaveLength(140);
  });
});

describe('memories.create_from_file', () => {
  function capture() {
    const calls: Array<{ url: string; init: RequestInit }> = [];
    const fetchImpl = (async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return json({ ingestion_id: 'ing_1', status: 'queued' });
    }) as unknown as typeof fetch;
    return { calls, fetchImpl };
  }

  it('requires exactly one source', async () => {
    const c = client((async () => json({})) as never);
    await expect(c.memories.create_from_file({ user_id: 'u', customer_id: 'cu' }))
      .rejects.toThrow(/One of file_path, file, or text/);
    await expect(
      c.memories.create_from_file({ user_id: 'u', customer_id: 'cu', text: 't', file_path: '/x' }),
    ).rejects.toThrow(/exactly one/);
  });

  it('requires user_id, and no longer requires customer_id', async () => {
    const c = client((async () => json({})) as never);
    await expect(c.memories.create_from_file({ text: 't', customer_id: 'cu' } as never))
      .rejects.toThrow(/user_id is required/);
    // customer_id used to be required here too. On a B2C instance
    // (user_context_isolation = equals_customer) the server REJECTS one, so
    // demanding it made the only correct B2C call impossible to express.
    await expect(c.memories.create_from_file({ text: 't', user_id: 'u' } as never))
      .resolves.toBeDefined();
  });

  it('sends multipart with Python\'s field set and defaults', async () => {
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create_from_file({
      user_id: 'u1', customer_id: 'cu1', text: 'raw text',
    });
    const body = calls[0]?.init.body as FormData;
    expect(body).toBeInstanceOf(FormData);
    expect(body.get('user_id')).toBe('u1');
    expect(body.get('customer_id')).toBe('cu1');
    expect(body.get('relationship_type')).toBe('b2c');
    expect(body.get('mode')).toBe('long-range');
    expect(body.get('text')).toBe('raw text');
    // Only when truthy in Python, so absent here.
    expect(body.get('document_type')).toBeNull();
    expect(body.get('metadata')).toBeNull();
  });

  it('omits an empty metadata object and JSON-encodes a populated one', async () => {
    const { calls, fetchImpl } = capture();
    const c = client(fetchImpl);
    await c.memories.create_from_file({ user_id: 'u', customer_id: 'c', text: 't', metadata: {} });
    expect((calls[0]?.init.body as FormData).get('metadata')).toBeNull();
    await c.memories.create_from_file({
      user_id: 'u', customer_id: 'c', text: 't', metadata: { a: 1 },
    });
    expect((calls[1]?.init.body as FormData).get('metadata')).toBe('{"a":1}');
  });

  it('never sets a JSON content-type on the multipart request', async () => {
    // Setting it by hand produces a boundary the server cannot parse.
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create_from_file({
      user_id: 'u', customer_id: 'c', text: 't',
    });
    const headers = calls[0]?.init.headers as Record<string, string>;
    expect(headers['Content-Type']).toBeUndefined();
    expect(headers['Authorization']).toBe('Bearer k');
  });

  it('names an unnamed buffer "upload", like Python', async () => {
    const { calls, fetchImpl } = capture();
    await client(fetchImpl).memories.create_from_file({
      user_id: 'u', customer_id: 'c', file: new Uint8Array([1, 2, 3]),
    });
    const file = (calls[0]?.init.body as FormData).get('file') as File;
    expect(file.name).toBe('upload');
  });
});

describe('compaction subscriptions', () => {
  it('fires subscribers in registration order', () => {
    const subs = new CompactionSubscribers();
    const order: number[] = [];
    subs.add('c1', () => order.push(1));
    subs.add('c1', () => order.push(2));
    subs.dispatch('c1', { x: 1 }, () => {});
    expect(order).toEqual([1, 2]);
  });

  it('returns an idempotent unsubscribe thunk', () => {
    const subs = new CompactionSubscribers();
    const seen: number[] = [];
    const off = subs.add('c1', () => seen.push(1));
    off();
    off();
    subs.dispatch('c1', {}, () => {});
    expect(seen).toEqual([]);
  });

  it('counts what removeAll removed', () => {
    const subs = new CompactionSubscribers();
    subs.add('c1', () => {});
    subs.add('c1', () => {});
    subs.add('c2', () => {});
    expect(subs.removeAll('c1')).toBe(2);
    expect(subs.removeAll('c1')).toBe(0);
    expect(subs.removeAll()).toBe(1);
  });

  it('does not let one throwing listener stop the others', () => {
    const subs = new CompactionSubscribers();
    const errors: unknown[] = [];
    const reached: string[] = [];
    subs.add('c1', () => { throw new Error('bad listener'); });
    subs.add('c1', () => reached.push('second'));
    subs.dispatch('c1', {}, (e) => errors.push(e));
    expect(reached).toEqual(['second']);
    expect(errors).toHaveLength(1);
  });

  it('swallows a rejected async listener', async () => {
    const subs = new CompactionSubscribers();
    const errors: unknown[] = [];
    subs.add('c1', async () => { throw new Error('async boom'); });
    subs.dispatch('c1', {}, (e) => errors.push(e));
    await new Promise((r) => setTimeout(r, 0));
    expect(errors).toHaveLength(1);
  });

  it('survives a listener that unsubscribes itself mid-dispatch', () => {
    const subs = new CompactionSubscribers();
    const reached: string[] = [];
    const off = subs.add('c1', () => { reached.push('first'); off(); });
    subs.add('c1', () => reached.push('second'));
    subs.dispatch('c1', {}, () => {});
    expect(reached).toEqual(['first', 'second']);
  });

  it('is reachable from the client and validates its input', () => {
    const c = client((async () => json({})) as never);
    expect(() => c.conversation.context.subscribe_to_compaction_updates('', () => {}))
      .toThrow(/conversation_id is required/);
    expect(() => c.conversation.context.subscribe_to_compaction_updates('c1', null as never))
      .toThrow(/callback is required/);
    const off = c.conversation.context.subscribe_to_compaction_updates('c1', () => {});
    expect(typeof off).toBe('function');
    expect(c.conversation.context.unsubscribe_all_compaction_updates('c1')).toBe(1);
  });
});

describe('initialize and configure', () => {
  it('resolves identity from whoami and exposes it', async () => {
    const fetchImpl = vi.fn(async () =>
      json({ client_id: 'cli_resolved', instance_id: 'inst_ffffffffffffffff' }));
    const c = new SynapClient({ _force_new: true, apiKey: 'k', fetchImpl: fetchImpl as never });
    expect(c.instance_id).toBe('');
    await c.initialize();
    expect(c.client_id).toBe('cli_resolved');
    expect(c.instance_id).toBe('inst_ffffffffffffffff');
  });

  it('is idempotent and skips whoami when identity is already known', async () => {
    const fetchImpl = vi.fn(async () => json({}));
    const c = new SynapClient({ _force_new: true, apiKey: 'k', clientId: 'c', instanceId: INSTANCE_ID, fetchImpl: fetchImpl as never,
    });
    await c.initialize();
    await c.initialize();
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it('treats a whoami failure as non-fatal', async () => {
    const c = new SynapClient({ _force_new: true, apiKey: 'k', fetchImpl: (async () => json({ detail: 'nope' }, 500)) as never,
      retryPolicy: { maxAttempts: 1 },
    });
    await expect(c.initialize()).resolves.toBeUndefined();
  });

  it('init() forwards to initialize()', async () => {
    const fetchImpl = vi.fn(async () => json({ client_id: 'cli_x' }));
    const c = new SynapClient({ _force_new: true, apiKey: 'k', fetchImpl: fetchImpl as never });
    await c.init();
    expect(c.client_id).toBe('cli_x');
  });

  it('refuses to reconfigure after initialize', async () => {
    const c = new SynapClient({ _force_new: true, apiKey: 'k', clientId: 'c', instanceId: INSTANCE_ID,
      fetchImpl: (async () => json({})) as never,
    });
    c.configure({ timeouts: { read: 5 } });
    await c.initialize();
    expect(() => c.configure({ timeouts: { read: 9 } })).toThrow(/Cannot reconfigure/);
  });

  it('accepts and ignores the Python-only configure keys', () => {
    const c = client((async () => json({})) as never);
    expect(() => c.configure({
      storage_path: '/tmp/x', cache_backend: 'sqlite',
      session_timeout_minutes: 30, log_level: 'DEBUG', logger: console,
    })).not.toThrow();
  });
});

describe('unified fetch scope selection', () => {
  function recorder() {
    const paths: string[] = [];
    const fetchImpl = (async (url: string) => {
      paths.push(new URL(url).pathname);
      return json({ context: {}, facts: [] });
    }) as unknown as typeof fetch;
    return { paths, fetchImpl };
  }

  it('queries only the scopes with an identifier', async () => {
    const { paths, fetchImpl } = recorder();
    await client(fetchImpl).fetch({ user_id: 'u1' });
    expect(paths.filter((p) => p.includes('/context/'))).toHaveLength(1);
    expect(paths.some((p) => p.includes('user'))).toBe(true);
  });

  it('does not query client scope unless it is named explicitly', async () => {
    // Client scope needs no id, so including it by default would silently add
    // a billed fetch to every unified call.
    const { paths, fetchImpl } = recorder();
    await client(fetchImpl).fetch({ user_id: 'u1' });
    expect(paths.some((p) => p.includes('client'))).toBe(false);

    const second = recorder();
    await client(second.fetchImpl).fetch({ user_id: 'u1', scopes: ['client'] });
    expect(second.paths.some((p) => p.includes('client'))).toBe(true);
  });

  it('returns an empty result without a request when no scope applies', async () => {
    const { paths, fetchImpl } = recorder();
    const result = await client(fetchImpl).fetch({});
    expect(paths).toHaveLength(0);
    expect(result.total_items).toBe(0);
    expect(result.formatted_context).toBe('');
    expect(result.scopes_queried).toEqual([]);
  });

  it('drops a failed scope but surfaces an InvalidInputError', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const failing = client((async (url: string) =>
        String(url).includes('customer') ? json({ detail: 'boom' }, 500) : json({ facts: [] })
      ) as never, { retryPolicy: { maxAttempts: 1 } });
      const result = await failing.fetch({ user_id: 'u1', customer_id: 'c1' });
      expect(result.scopes_queried).toEqual(['user']);

      const invalid = client((async (url: string) =>
        String(url).includes('customer') ? json({ detail: 'bad' }, 400) : json({ facts: [] })
      ) as never, { retryPolicy: { maxAttempts: 1 } });
      await expect(invalid.fetch({ user_id: 'u1', customer_id: 'c1' }))
        .rejects.toThrow(InvalidInputError);
    } finally {
      warn.mockRestore();
    }
  });
});

describe('polling loops hold the event loop', () => {
  it('wait_for_completion settles when its poll interval is the only pending work', async () => {
    // Regression: the poll `sleep` was unref'd, so a process with no other
    // work exited mid-poll and the awaited call never settled. Only reachable
    // when ingestion is not already terminal on the first status check, which
    // is why it survived every earlier test.
    let calls = 0;
    const fetchImpl = (async () => {
      calls += 1;
      return json({ status: calls < 3 ? 'processing' : 'completed', ingestion_id: 'ing_1' });
    }) as unknown as typeof fetch;

    const result = await client(fetchImpl).memories.wait_for_completion('ing_1', {
      timeout_seconds: 5,
      poll_interval_seconds: 0.05,
    });
    expect(calls).toBe(3);
    expect(result['status']).toBe('completed');
  });

  it('returns the last status at timeout rather than throwing, as Python does', async () => {
    const fetchImpl = (async () => json({ status: 'processing' })) as unknown as typeof fetch;
    const result = await client(fetchImpl).memories.wait_for_completion('ing_1', {
      timeout_seconds: 0.1,
      poll_interval_seconds: 0.05,
    });
    expect(result['status']).toBe('processing');
  });
});

describe('invalidate-on-write (SYNAP_SDK_CACHE_INVALIDATE_ON_WRITE)', () => {
  const FLAG = 'SYNAP_SDK_CACHE_INVALIDATE_ON_WRITE';
  afterEach(() => { delete process.env[FLAG]; });

  it('is a no-op when the flag is unset, matching Python\'s default', () => {
    const cache = new AnticipationCache();
    cache.store({ bundleId: 'b1', entityId: 'u1', itemsByType: { facts: [] } });
    expect(cache.invalidateEntity('u1')).toBe(0);
    expect(cache.snapshot().total_entries).toBe(1);
  });

  it('drops the writing entity\'s bundles when the flag is set', () => {
    process.env[FLAG] = 'true';
    const cache = new AnticipationCache();
    cache.store({ bundleId: 'b1', entityId: 'u1', itemsByType: { facts: [] } });
    cache.store({ bundleId: 'b2', entityId: 'u2', itemsByType: { facts: [] } });
    expect(cache.invalidateEntity('u1')).toBe(1);
    // Only the writing entity, never everyone.
    expect(cache.snapshot().bundles.map((b) => b.bundle_id)).toEqual(['b2']);
  });

  it('reaches the cache from all three write paths', async () => {
    // Regression: the flag and its gate both existed, but nothing called
    // invalidateEntity from a write path, so turning the flag on did nothing.
    // Python invalidates from record_message, record_messages_batch and
    // ingest_transcript.
    process.env[FLAG] = 'true';
    const seen: string[] = [];
    const ns = createConversationNamespace(
      new HttpTransport({
        credentials: CREDS,
        fetchImpl: (async () => json({ ok: true })) as never,
      }),
      new CompactionSubscribers(),
      ({ userId, customerId }) => {
        for (const id of new Set([userId, customerId])) if (id) seen.push(id);
      },
    );

    await ns.record_message({
      conversation_id: UUID, user_id: 'u1', customer_id: 'c1', role: 'user', content: 'x',
    });
    expect(seen).toEqual(['u1', 'c1']);

    seen.length = 0;
    await ns.record_messages_batch([
      { conversation_id: UUID, user_id: 'u1', customer_id: 'c1', role: 'user', content: 'a' },
      { conversation_id: UUID, user_id: 'u2', customer_id: 'c1', role: 'user', content: 'b' },
    ]);
    // Every distinct entity in the batch, not just the first message's.
    expect(seen).toContain('u1');
    expect(seen).toContain('u2');

    seen.length = 0;
    await ns.ingest_transcript({
      conversation_id: 'free-form', user_id: 'u3', customer_id: 'c1', transcript: 'hello',
    });
    expect(seen).toEqual(['u3', 'c1']);
  });
});

describe('environment compatibility with the 0.3.x wrapper', () => {
  const VARS = ['SYNAP_BASE_URL', 'SYNAP_GRPC_TLS', 'SYNAP_GRPC_USE_TLS'];
  afterEach(() => { for (const v of VARS) delete process.env[v]; });

  it('honours SYNAP_BASE_URL, as Python and the old wrapper do', async () => {
    // Ignoring it was a wrong-DESTINATION bug: anyone pointing at a self-hosted
    // or staging deployment through the environment would silently have started
    // sending their data to production on upgrade.
    process.env['SYNAP_BASE_URL'] = 'https://synap.internal.example.com';
    let seen = '';
    const c = new SynapClient({
      apiKey: 'k',
      _force_new: true,
      fetchImpl: (async (url: string) => { seen = String(url); return json({}); }) as never,
    });
    await c.credits.get_balance().catch(() => {});
    expect(seen.startsWith('https://synap.internal.example.com')).toBe(true);
    await c.shutdown();
  });

  it('lets an explicit baseUrl beat the environment', async () => {
    process.env['SYNAP_BASE_URL'] = 'https://from-env.example.com';
    let seen = '';
    const c = new SynapClient({
      apiKey: 'k',
      _force_new: true,
      baseUrl: 'https://explicit.example.com',
      fetchImpl: (async (url: string) => { seen = String(url); return json({}); }) as never,
    });
    await c.credits.get_balance().catch(() => {});
    expect(seen.startsWith('https://explicit.example.com')).toBe(true);
    await c.shutdown();
  });

  it('ignores a blank SYNAP_BASE_URL rather than building a bad URL', async () => {
    process.env['SYNAP_BASE_URL'] = '   ';
    let seen = '';
    const c = new SynapClient({
      apiKey: 'k',
      _force_new: true,
      fetchImpl: (async (url: string) => { seen = String(url); return json({}); }) as never,
    });
    await c.credits.get_balance().catch(() => {});
    expect(seen.startsWith('https://')).toBe(true);
    await c.shutdown();
  });
});
