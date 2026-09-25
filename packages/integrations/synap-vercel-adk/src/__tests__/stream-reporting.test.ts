/**
 * What a wrapped model reports to Synap.
 *
 * The middleware sat on every model call and read one thing out of it:
 * `text-delta`. Tool calls and reasoning passed straight through, so
 * anticipation got the answer and none of the work that produced it. And the
 * memory write posted to a route that does not exist, with nobody checking the
 * response, so it 404'd for as long as it existed.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createSynapMiddleware } from '../middleware.js';
import { AnticipationCache } from '../context/anticipation-cache.js';

interface Sent { [k: string]: unknown }

function harness() {
  const sent: Sent[] = [];
  const grpcClient = {
    isConnected: true,
    async sendConversationEvent(event: Sent) { sent.push(event); },
    async sendContextUsed() {},
    async sendContextAssembled() {},
  };
  const middleware = createSynapMiddleware({
    credentials: { api_key: 'k', client_id: 'c', instance_id: 'i' },
    anticipationCache: new AnticipationCache(),
    grpcClient: grpcClient as never,
    userId: 'u1',
    customerId: 'cus1',
    conversationId: 'c1',
    injectContext: false,
    writeMemory: false,
  });
  return { middleware, sent };
}

function eventsOfType(sent: Sent[], type: string): Sent[] {
  return sent.filter((e) => e['event_type'] === type);
}

/** Let the fire-and-forget emitters settle. */
const settle = () => new Promise((r) => setTimeout(r, 20));

describe('a non-streaming call', () => {
  it('reports the tool the model asked for', async () => {
    const { middleware, sent } = harness();
    await middleware.wrapGenerate?.({
      doGenerate: async () => ({
        text: 'Your order shipped.',
        toolCalls: [{ toolName: 'lookup_order', toolCallId: 'call_1', args: { id: 'A-1' } }],
      }),
      params: { prompt: [] },
    } as never);
    await settle();

    const [call] = eventsOfType(sent, 'tool_call');
    expect(call?.['tool_name']).toBe('lookup_order');
    expect(call?.['tool_call_id']).toBe('call_1');
    expect(call?.['tool_args_json']).toBe('{"id":"A-1"}');
    expect(call?.['role']).toBe('assistant');
  });

  it('reports reasoning with no role', async () => {
    // A thought is not the assistant's reply, and role 'assistant' is what
    // made the server file every reasoning step as the final answer.
    const { middleware, sent } = harness();
    await middleware.wrapGenerate?.({
      doGenerate: async () => ({ text: 'done', reasoning: 'Check the order first.' }),
      params: { prompt: [] },
    } as never);
    await settle();

    const [thought] = eventsOfType(sent, 'agent_thinking');
    expect(thought?.['content']).toBe('Check the order first.');
    expect(thought?.['role']).toBe('');
    expect((thought?.['metadata'] as Record<string, string>)?.['thought_type'])
      .toBe('model_reasoning');
  });

  it('sends no reasoning event when the provider returned none', async () => {
    const { middleware, sent } = harness();
    await middleware.wrapGenerate?.({
      doGenerate: async () => ({ text: 'done' }),
      params: { prompt: [] },
    } as never);
    await settle();
    expect(eventsOfType(sent, 'agent_thinking')).toHaveLength(0);
  });
});

describe('a streaming call', () => {
  function streamOf(parts: Array<Record<string, unknown>>) {
    return {
      doStream: async () => ({
        stream: new ReadableStream({
          start(controller) {
            for (const p of parts) controller.enqueue(p);
            controller.close();
          },
        }),
      }),
      params: { prompt: [] },
    };
  }

  async function drain(stream: ReadableStream) {
    const reader = stream.getReader();
    for (;;) {
      const { done } = await reader.read();
      if (done) break;
    }
  }

  it('reports a tool call mid-stream', async () => {
    const { middleware, sent } = harness();
    const out = await middleware.wrapStream?.(streamOf([
      { type: 'tool-call', toolName: 'lookup_order', toolCallId: 'call_1', args: { id: 1 } },
      { type: 'text-delta', textDelta: 'shipped' },
    ]) as never);
    await drain((out as { stream: ReadableStream }).stream);
    await settle();

    expect(eventsOfType(sent, 'tool_call')[0]?.['tool_name']).toBe('lookup_order');
  });

  it('accumulates reasoning deltas into one event', async () => {
    // One thought is one step. A stream event per token would be noise.
    const { middleware, sent } = harness();
    const out = await middleware.wrapStream?.(streamOf([
      { type: 'reasoning', textDelta: 'Check the ' },
      { type: 'reasoning', textDelta: 'order first.' },
      { type: 'text-delta', textDelta: 'shipped' },
    ]) as never);
    await drain((out as { stream: ReadableStream }).stream);
    await settle();

    const thoughts = eventsOfType(sent, 'agent_thinking');
    expect(thoughts).toHaveLength(1);
    expect(thoughts[0]?.['content']).toBe('Check the order first.');
  });

  it('passes every part through untouched', async () => {
    // Reporting must not change what the caller receives.
    const { middleware } = harness();
    const parts = [
      { type: 'tool-call', toolName: 't', toolCallId: '1', args: {} },
      { type: 'reasoning', textDelta: 'hm' },
      { type: 'text-delta', textDelta: 'hi' },
    ];
    const out = await middleware.wrapStream?.(streamOf(parts) as never);
    const seen: unknown[] = [];
    const reader = (out as { stream: ReadableStream }).stream.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      seen.push(value);
    }
    expect(seen).toHaveLength(parts.length);
  });
});

describe('the memory write', () => {
  const originalFetch = globalThis.fetch;
  beforeEach(() => { vi.restoreAllMocks(); });
  afterEach(() => { globalThis.fetch = originalFetch; });

  it('posts to a route that exists', async () => {
    // It posted to /v1/memories/ingest, which is not a route, and never looked
    // at the response — so every write 404'd silently.
    const calls: string[] = [];
    globalThis.fetch = vi.fn(async (url: unknown) => {
      calls.push(String(url));
      return { ok: true, status: 200, statusText: 'OK' } as Response;
    }) as never;

    const { writeMemory } = await import('../memory/writer.js');
    await writeMemory({
      credentials: { api_key: 'k', client_id: 'c', instance_id: 'i' },
      modelOptions: { userId: 'u1', conversationId: 'c1' },
      messages: [{ role: 'user', content: 'hi' }],
      assistantResponse: 'hello',
      baseUrl: 'https://example.test',
    });

    expect(calls[0]).toBe('https://example.test/api/v1/memories/create');
  });

  it('warns when the write is refused instead of swallowing it', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    globalThis.fetch = vi.fn(async () => (
      { ok: false, status: 404, statusText: 'Not Found' } as Response
    )) as never;

    const { writeMemory } = await import('../memory/writer.js');
    await writeMemory({
      credentials: { api_key: 'k', client_id: 'c', instance_id: 'i' },
      modelOptions: { userId: 'u1' },
      messages: [{ role: 'user', content: 'hi' }],
      assistantResponse: 'hello',
      baseUrl: 'https://example.test',
    });

    expect(warn).toHaveBeenCalled();
  });

  it('omits customer_id rather than sending an empty one', async () => {
    // A B2C instance refuses a call that carries a customer_id at all.
    let body: Record<string, unknown> = {};
    globalThis.fetch = vi.fn(async (_url: unknown, init: unknown) => {
      body = JSON.parse((init as { body: string }).body);
      return { ok: true, status: 200, statusText: 'OK' } as Response;
    }) as never;

    const { writeMemory } = await import('../memory/writer.js');
    await writeMemory({
      credentials: { api_key: 'k', client_id: 'c', instance_id: 'i' },
      modelOptions: { userId: 'u1' },
      messages: [{ role: 'user', content: 'hi' }],
      assistantResponse: 'hello',
      baseUrl: 'https://example.test',
    });

    expect('customer_id' in body).toBe(false);
  });
});
