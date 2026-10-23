/**
 * The package used the way the README shows: createSynap(), wrap(), then the
 * AI SDK's own generateText/streamText over its mock model. No listen().
 * Against a real local gRPC `Listen` server; HTTP is a stubbed fetch.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { generateText, streamText, tool, jsonSchema } from 'ai';
import { MockLanguageModelV1, simulateReadableStream } from 'ai/test';
import { createSynap } from '../provider.js';
import { startGrpcServer, type TestServer } from './helpers/grpc-server.js';

const BASE = 'https://synap.test';

interface Call { method: string; url: string; body: Record<string, unknown> | null }

/** Routes the Synap HTTP surface; `compaction: null` is a conversation not yet compacted. */
function stubHttp(compaction: Record<string, unknown> | null): Call[] {
  const calls: Call[] = [];
  globalThis.fetch = vi.fn(async (url: unknown, init?: RequestInit) => {
    const u = String(url).replace(BASE, '');
    calls.push({ method: init?.method ?? 'GET', url: u, body: init?.body ? JSON.parse(String(init.body)) : null });
    const json = (o: unknown) => ({ ok: true, status: 200, statusText: 'OK', json: async () => o }) as Response;
    if (u.includes('/context-for-prompt')) return json(compaction ?? { available: false, recent_messages: [{ role: 'user', content: 'x' }] });
    if (u.includes('/v1/context/')) {
      return json({
        items_by_type: { facts: { items: [{ item_id: 'f1', content: 'User likes tea', context_type: 'fact' }] } },
        conversation_context: { summary: 'CTX-SUMMARY', recent_turns: [{ role: 'user', content: 'CTX-RECENT-TURN', timestamp: '' }] },
      });
    }
    return json({});
  }) as never;
  return calls;
}

function mockModel(seen: Array<unknown[]> = []) {
  return new MockLanguageModelV1({
    modelId: 'mock-1',
    doGenerate: async (o) => {
      seen.push(o.prompt);
      return { text: 'the reply', finishReason: 'stop', usage: { promptTokens: 1, completionTokens: 1 }, rawCall: { rawPrompt: null, rawSettings: {} } };
    },
    doStream: async (o) => {
      seen.push(o.prompt);
      return {
        stream: simulateReadableStream({
          chunks: [
            { type: 'text-delta', textDelta: 'streamed ' }, { type: 'text-delta', textDelta: 'reply' },
            { type: 'finish', finishReason: 'stop', usage: { promptTokens: 1, completionTokens: 1 } },
          ],
        }),
        rawCall: { rawPrompt: null, rawSettings: {} },
      };
    },
  });
}

const history = (n: number) => Array.from({ length: n }, (_, i) => [
  { role: 'user' as const, content: `earlier user ${i + 1}` },
  { role: 'assistant' as const, content: `earlier assistant ${i + 1}` },
]).flat();

async function until(cond: () => boolean, ms = 2000): Promise<void> {
  const end = Date.now() + ms;
  while (!cond() && Date.now() < end) await new Promise((r) => setTimeout(r, 10));
}

const COMPACTION = {
  available: true, compaction_id: 'cmp_1',
  context_block: '## Conversation so far (compacted)\nLisbon trip, prefers trains.',
  recent_messages: [{ role: 'user', content: 'earlier user 4' }, { role: 'assistant', content: 'earlier assistant 4' }],
};

describe('the default path (no listen())', () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = process.env;
  let server: TestServer;
  const providers: Array<{ stopListening(): Promise<void> }> = [];

  const make = async (extra: Record<string, unknown> = {}) => {
    const p = await createSynap({
      apiKey: 'sk-test', baseUrl: BASE, grpcHost: '127.0.0.1', grpcPort: server.port, grpcUseTls: false, ...extra,
    });
    providers.push(p);
    return p;
  };

  beforeEach(async () => {
    process.env = { ...originalEnv };
    delete process.env['SYNAP_AUTO_LISTEN'];
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    server = await startGrpcServer();
  });
  afterEach(async () => {
    for (const p of providers.splice(0)) await p.stopListening();
    await server.close();
    globalThis.fetch = originalFetch;
    process.env = originalEnv;
    vi.restoreAllMocks();
  });

  it('opens the stream on the first call and sends the turn on it, not over REST', async () => {
    const calls = stubHttp(null);
    const synap = await make();
    expect(synap.isListening).toBe(false);

    const model = synap.wrap(mockModel(), { userId: 'u1', customerId: 'c1', conversationId: 'conv-1' });
    await generateText({ model, messages: [{ role: 'user', content: 'hello there' }] });
    await until(() => server.events.length >= 2);

    expect(synap.isListening).toBe(true);
    expect(server.streams()).toBe(1);
    expect(server.events.map((e) => [e['event_type'], e['role'], e['content']])).toEqual([
      ['user_message', 'user', 'hello there'],
      ['assistant_message', 'assistant', 'the reply'],
    ]);
    expect(server.events[0]).toMatchObject({ conversation_id: 'conv-1', user_id: 'u1', customer_id: 'c1' });
    // Never both: no REST turn write, and no transcript re-ingest.
    expect(calls.filter((c) => c.method === 'POST' && /conversations\/messages|memories/.test(c.url))).toHaveLength(0);
  });

  it('streamText takes the same path and records the accumulated reply', async () => {
    const calls = stubHttp(null);
    const synap = await make();
    const model = synap.wrap(mockModel(), { userId: 'u1', conversationId: 'conv-1' });
    const result = streamText({ model, messages: [{ role: 'user', content: 'hi' }] });
    let text = '';
    for await (const c of result.textStream) text += c;
    await until(() => server.events.length >= 2);

    expect(text).toBe('streamed reply');
    expect(server.events.map((e) => [e['event_type'], e['content']])).toEqual([
      ['user_message', 'hi'], ['assistant_message', 'streamed reply'],
    ]);
    expect(calls.filter((c) => /conversations\/messages|memories/.test(c.url))).toHaveLength(0);
  });

  it('records only the new turn, not the history the caller resends', async () => {
    stubHttp(null);
    const synap = await make();
    const model = synap.wrap(mockModel(), { userId: 'u1', conversationId: 'conv-1' });
    await generateText({ model, messages: [...history(5), { role: 'user', content: 'newest' }] });
    await until(() => server.events.length >= 2);
    await new Promise((r) => setTimeout(r, 50));

    expect(server.events.map((e) => e['content'])).toEqual(['newest', 'the reply']);
  });

  it('a model wrapped BEFORE the stream existed still uses it, and again after stop + listen', async () => {
    stubHttp(null);
    const synap = await make({ autoListen: false });
    const model = synap.wrap(mockModel(), { userId: 'u1', conversationId: 'conv-1' });

    await synap.listen();
    await generateText({ model, messages: [{ role: 'user', content: 'one' }] });
    await until(() => server.events.length >= 2);
    expect(server.events.map((e) => e['content'])).toEqual(['one', 'the reply']);

    await synap.stopListening();
    await synap.listen();
    await generateText({ model, messages: [{ role: 'user', content: 'two' }] });
    await until(() => server.events.length >= 4);
    expect(server.events.slice(2).map((e) => e['content'])).toEqual(['two', 'the reply']);
    expect(server.streams()).toBe(2);
  });

  it('concurrent first calls share one stream', async () => {
    stubHttp(null);
    const synap = await make();
    const a = synap.wrap(mockModel(), { userId: 'u1', conversationId: 'conv-a' });
    const b = synap.wrap(mockModel(), { userId: 'u2', conversationId: 'conv-b' });
    await Promise.all([
      generateText({ model: a, messages: [{ role: 'user', content: 'a' }] }),
      generateText({ model: b, messages: [{ role: 'user', content: 'b' }] }),
    ]);
    await until(() => server.events.length >= 4);
    expect(server.streams()).toBe(1);
    expect(server.events).toHaveLength(4);
  });

  it('SYNAP_AUTO_LISTEN=false: nothing opens, turns go over REST', async () => {
    process.env['SYNAP_AUTO_LISTEN'] = 'false';
    const calls = stubHttp(null);
    const synap = await make();
    const model = synap.wrap(mockModel(), { userId: 'u1', conversationId: 'conv-1' });
    await generateText({ model, messages: [{ role: 'user', content: 'hello' }] });
    await until(() => calls.filter((c) => c.url === '/v1/conversations/messages').length >= 2);

    expect(synap.isListening).toBe(false);
    expect(server.streams()).toBe(0);
    const writes = calls.filter((c) => c.url === '/v1/conversations/messages');
    expect(writes.map((c) => [c.body?.['role'], c.body?.['content']])).toEqual([['user', 'hello'], ['assistant', 'the reply']]);
    expect(writes[0]!.body).toMatchObject({ conversation_id: 'conv-1', user_id: 'u1' });
    expect(calls.some((c) => c.url.includes('memories'))).toBe(false);
  });

  it('the autoListen: false option does the same', async () => {
    const calls = stubHttp(null);
    const synap = await make({ autoListen: false });
    const model = synap.wrap(mockModel(), { userId: 'u1', conversationId: 'conv-1' });
    await generateText({ model, messages: [{ role: 'user', content: 'hello' }] });
    await until(() => calls.filter((c) => c.url === '/v1/conversations/messages').length >= 2);
    expect(server.streams()).toBe(0);
  });

  it('an unreachable gRPC host falls back to REST once, without failing the call, and is not retried at once', async () => {
    const calls = stubHttp(null);
    const dead = await startGrpcServer();
    const deadPort = dead.port;
    await dead.close();
    const synap = await make({ grpcPort: deadPort });
    const model = synap.wrap(mockModel(), { userId: 'u1', conversationId: 'conv-1' });

    const r1 = await generateText({ model, messages: [{ role: 'user', content: 'one' }] });
    await until(() => calls.filter((c) => c.url === '/v1/conversations/messages').length >= 2);
    expect(r1.text).toBe('the reply');
    expect(synap.isListening).toBe(false);

    const warns = (console.warn as unknown as ReturnType<typeof vi.fn>).mock.calls.filter((c) => String(c[0]).includes('gRPC stream'));
    await generateText({ model, messages: [{ role: 'user', content: 'two' }] });
    await until(() => calls.filter((c) => c.url === '/v1/conversations/messages').length >= 4);
    const warns2 = (console.warn as unknown as ReturnType<typeof vi.fn>).mock.calls.filter((c) => String(c[0]).includes('gRPC stream'));
    expect(warns).toHaveLength(1);
    expect(warns2).toHaveLength(1); // cooldown: the second call did not try again
    expect(calls.filter((c) => c.url === '/v1/conversations/messages')).toHaveLength(4);
  });

  it('degrades to REST without a warning where gRPC cannot load (no Node)', async () => {
    const calls = stubHttp(null);
    const synap = await make();
    const model = synap.wrap(mockModel(), { userId: 'u1', conversationId: 'conv-1' });
    const desc = Object.getOwnPropertyDescriptor(process.versions, 'node')!;
    Object.defineProperty(process.versions, 'node', { value: undefined, configurable: true });
    try {
      await generateText({ model, messages: [{ role: 'user', content: 'edge' }] });
      await until(() => calls.filter((c) => c.url === '/v1/conversations/messages').length >= 2);
    } finally {
      Object.defineProperty(process.versions, 'node', desc);
    }
    expect(server.streams()).toBe(0);
    expect(calls.filter((c) => c.url === '/v1/conversations/messages')).toHaveLength(2);
    expect((console.warn as unknown as ReturnType<typeof vi.fn>).mock.calls.filter((c) => String(c[0]).includes('gRPC'))).toHaveLength(0);
  });

  it('tool calls and tool results go on the stream, each once, over a multi-step turn', async () => {
    stubHttp(null);
    const synap = await make();
    let step = 0;
    const model = synap.wrap(new MockLanguageModelV1({
      modelId: 'mock-1',
      doGenerate: async () => (step++ === 0
        ? { toolCalls: [{ toolCallType: 'function' as const, toolCallId: 'call_1', toolName: 'lookup', args: '{"id":"A-1"}' }], finishReason: 'tool-calls' as const, usage: { promptTokens: 1, completionTokens: 1 }, rawCall: { rawPrompt: null, rawSettings: {} } }
        : { text: 'Shipped.', finishReason: 'stop' as const, usage: { promptTokens: 1, completionTokens: 1 }, rawCall: { rawPrompt: null, rawSettings: {} } }),
    }), { userId: 'u1', conversationId: 'conv-1' });

    await generateText({
      model, maxSteps: 3,
      tools: { lookup: tool({ description: 'x', parameters: jsonSchema<{ id: string }>({ type: 'object', properties: { id: { type: 'string' } }, required: ['id'] }), execute: async () => ({ status: 'shipped' }) }) },
      messages: [{ role: 'user', content: 'where is my order?' }],
    });
    await until(() => server.events.some((e) => e['event_type'] === 'assistant_message'));
    await new Promise((r) => setTimeout(r, 50));

    const types = server.events.map((e) => e['event_type']);
    // The user message is recorded once although the model was called twice.
    expect(types.filter((t) => t === 'user_message')).toHaveLength(1);
    expect(types.filter((t) => t === 'tool_call')).toHaveLength(1);
    const result = server.events.find((e) => e['event_type'] === 'tool_result')!;
    expect(result).toMatchObject({ role: 'tool', tool_call_id: 'call_1', tool_name: 'lookup', tool_result_json: '{"status":"shipped"}' });
    expect(types.filter((t) => t === 'tool_result')).toHaveLength(1);
  });
});

describe('Synap\'s compaction replaces the caller\'s earlier turns (default)', () => {
  const originalFetch = globalThis.fetch;
  const originalEnv = process.env;
  beforeEach(() => {
    process.env = { ...originalEnv, SYNAP_AUTO_LISTEN: 'false' };
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });
  afterEach(() => { globalThis.fetch = originalFetch; process.env = originalEnv; vi.restoreAllMocks(); });

  const render = (prompt: unknown[]) => (prompt as Array<{ role: string; content: unknown }>).map((m) =>
    `${m.role}:${typeof m.content === 'string' ? m.content : (m.content as Array<{ text?: string }>).map((p) => p.text ?? '').join('')}`);

  it('the model sees the preamble and Synap\'s turns, not the replaced ones', async () => {
    stubHttp(COMPACTION);
    const synap = await createSynap({ apiKey: 'sk-test', baseUrl: BASE });
    const seen: unknown[][] = [];
    const model = synap.wrap(mockModel(seen), { userId: 'u1', conversationId: 'conv-1' });
    await generateText({ model, system: 'Be brief.', messages: [...history(4), { role: 'user', content: 'what should I pack?' }] });

    const lines = render(seen[0]!);
    const joined = lines.join('\n');
    expect(joined).toContain('<synap_conversation_history>');
    expect(joined).toContain('Lisbon trip, prefers trains.');
    for (const gone of ['earlier user 1', 'earlier assistant 1', 'earlier user 3', 'earlier assistant 3']) {
      expect(joined).not.toContain(gone);
    }
    // Synap's post-compaction turns stand in for them, then the current turn.
    expect(lines.slice(-3)).toEqual(['user:earlier user 4', 'assistant:earlier assistant 4', 'user:what should I pack?']);
    // The caller's system prompt and the context block survive.
    expect(joined).toContain('Be brief.');
    expect(joined).toContain('User likes tea');
    // The block no longer repeats turns: not its recent turns, not its summary.
    expect(joined).not.toContain('CTX-RECENT-TURN');
    expect(joined).not.toContain('CTX-SUMMARY');
  });

  it('with no compaction yet the caller\'s messages are untouched, and the block holds no duplicate turns', async () => {
    stubHttp(null);
    const synap = await createSynap({ apiKey: 'sk-test', baseUrl: BASE });
    const seen: unknown[][] = [];
    const model = synap.wrap(mockModel(seen), { userId: 'u1', conversationId: 'conv-1' });
    const messages = [...history(4), { role: 'user' as const, content: 'what should I pack?' }];
    await generateText({ model, messages });

    const lines = render(seen[0]!);
    expect(lines.filter((l) => !l.startsWith('system:'))).toEqual(
      messages.map((m) => `${m.role}:${m.content}`));
    expect(lines.join('\n')).not.toContain('synap_conversation_history');
    expect(lines.join('\n')).not.toContain('CTX-RECENT-TURN');
    // Not compacted: the context summary has nothing to duplicate and stays.
    expect(lines.join('\n')).toContain('CTX-SUMMARY');
  });

  it('a failing compaction fetch leaves the messages untouched', async () => {
    const calls = stubHttp(null);
    const real = globalThis.fetch;
    globalThis.fetch = vi.fn(async (url: unknown, init?: RequestInit) => {
      if (String(url).includes('/context-for-prompt')) throw new Error('boom');
      return real(url as never, init);
    }) as never;
    void calls;
    const synap = await createSynap({ apiKey: 'sk-test', baseUrl: BASE });
    const seen: unknown[][] = [];
    const model = synap.wrap(mockModel(seen), { userId: 'u1', conversationId: 'conv-1' });
    await generateText({ model, messages: [...history(2), { role: 'user', content: 'now' }] });
    expect(render(seen[0]!).filter((l) => l.startsWith('user:'))).toEqual(
      ['user:earlier user 1', 'user:earlier user 2', 'user:now']);
  });

  it('is not applied without a conversationId', async () => {
    const calls = stubHttp(COMPACTION);
    const synap = await createSynap({ apiKey: 'sk-test', baseUrl: BASE });
    const seen: unknown[][] = [];
    const model = synap.wrap(mockModel(seen), { userId: 'u1' });
    await generateText({ model, messages: [...history(2), { role: 'user', content: 'now' }] });
    expect(calls.some((c) => c.url.includes('context-for-prompt'))).toBe(false);
    expect(render(seen[0]!).join('\n')).toContain('earlier user 1');
  });

  it('the spliced preamble is never written back to Synap as a turn', async () => {
    const calls = stubHttp(COMPACTION);
    const synap = await createSynap({ apiKey: 'sk-test', baseUrl: BASE });
    const model = synap.wrap(mockModel(), { userId: 'u1', conversationId: 'conv-1' });
    await generateText({ model, messages: [...history(4), { role: 'user', content: 'what should I pack?' }] });
    await until(() => calls.filter((c) => c.url === '/v1/conversations/messages').length >= 2);
    const written = calls.filter((c) => c.url === '/v1/conversations/messages').map((c) => String(c.body?.['content']));
    expect(written).toEqual(['what should I pack?', 'the reply']);
  });
});
