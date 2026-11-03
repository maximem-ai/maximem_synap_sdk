// scopePath: the level of a deeper scope ladder, carried to every Synap call
// this package makes. Every assertion is on what reached the fake SDK.
//
// The names differ by call: `scope_path` on fetch and conversation writes,
// `scope` on memories.create. A call made without a path is pinned to exactly
// the keys it had before.

import { Agent } from '@mastra/core/agent';
import { createMockModel } from '@mastra/core/test-utils/llm-mock';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { SynapMemory } from '../memory.js';
import { setAutoListen } from '../stream-events.js';
import { synapSearchTool, synapStoreTool } from '../tools.js';
import { makeSdk, msg } from './helpers.js';

const PATH = { customer: 'acme', team: 'eng', user: 'alice' };
const EVIL = { team: 'finance', user: 'bob' };

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  return () => vi.restoreAllMocks();
});

const callOf = (fn: unknown, i = 0): Record<string, unknown> =>
  (fn as ReturnType<typeof vi.fn>).mock.calls[i]![0] as Record<string, unknown>;

const run = (tool: { execute?: unknown }, input: unknown) =>
  (tool.execute as (i: unknown) => Promise<unknown>)(input);

describe('tools', () => {
  it('search reads with scope_path', async () => {
    const sdk = makeSdk();
    await run(synapSearchTool({ sdk, userId: 'alice', scopePath: PATH }), { query: 'x' });
    expect(callOf(sdk.fetch)['scope_path']).toEqual(PATH);
  });

  it('store writes with scope (not scope_path)', async () => {
    const sdk = makeSdk();
    await run(synapStoreTool({ sdk, userId: 'alice', scopePath: PATH }), { content: 'note' });
    const sent = callOf(sdk.memories.create);
    expect(sent['scope']).toEqual(PATH);
    expect(Object.keys(sent)).not.toContain('scope_path');
  });

  it('without a path (or with an empty one) both calls carry exactly the keys they always did', async () => {
    for (const scopePath of [undefined, {}]) {
      const sdk = makeSdk();
      await run(synapSearchTool({ sdk, userId: 'alice', scopePath }), { query: 'x' });
      await run(synapStoreTool({ sdk, userId: 'alice', scopePath }), { content: 'note' });
      expect(callOf(sdk.fetch)).toEqual({
        conversation_id: null, user_id: 'alice', customer_id: null, search_query: ['x'],
        max_results: 10, mode: 'accurate', include_conversation_context: false,
      });
      expect(callOf(sdk.memories.create)).toEqual({
        document: 'note', user_id: 'alice', customer_id: null, metadata: { source: 'mastra' },
      });
    }
  });

  it('is closed over: in neither input schema, and a model-supplied one is ignored', async () => {
    const sdk = makeSdk();
    const search = synapSearchTool({ sdk, userId: 'alice', scopePath: PATH });
    const store = synapStoreTool({ sdk, userId: 'alice', scopePath: PATH });
    for (const tool of [search, store]) {
      const shape = Object.keys((tool as unknown as { inputSchema: { shape: object } }).inputSchema.shape);
      expect(shape.filter((k) => /scope/i.test(k))).toEqual([]);
    }
    await run(search, { query: 'x', scope_path: EVIL, scopePath: EVIL, scope: EVIL });
    await run(store, { content: 'n', scope_path: EVIL, scopePath: EVIL, scope: EVIL });
    expect(callOf(sdk.fetch)['scope_path']).toEqual(PATH);
    expect(callOf(sdk.memories.create)['scope']).toEqual(PATH);
  });

  it('a tool with no path configured ignores a model-supplied one', async () => {
    const sdk = makeSdk();
    await run(synapSearchTool({ sdk, userId: 'alice' }), { query: 'x', scope_path: EVIL });
    await run(synapStoreTool({ sdk, userId: 'alice' }), { content: 'n', scope: EVIL });
    expect(Object.keys(callOf(sdk.fetch))).not.toContain('scope_path');
    expect(Object.keys(callOf(sdk.memories.create))).not.toContain('scope');
  });
});

describe('SynapMemory', () => {
  it('getSystemMessage reads with scope_path', async () => {
    const sdk = makeSdk();
    await new SynapMemory({ sdk, userId: 'alice', scopePath: PATH }).getSystemMessage({ threadId: 't-1' });
    expect(callOf(sdk.fetch)['scope_path']).toEqual(PATH);
  });

  it('getSystemMessage without a path sends exactly the keys it always did', async () => {
    const sdk = makeSdk();
    await new SynapMemory({ sdk, userId: 'alice' }).getSystemMessage({ threadId: 't-1' });
    expect(callOf(sdk.fetch)).toEqual({
      conversation_id: 't-1', user_id: 'alice', customer_id: null, max_results: 20,
      mode: 'accurate', include_conversation_context: false,
    });
  });

  it('saveMessages: user, assistant and system turns each reach record_message with scope_path', async () => {
    const sdk = makeSdk();
    setAutoListen(sdk, false);
    const mem = new SynapMemory({ sdk, userId: 'alice', scopePath: PATH });
    await mem.saveMessages({
      messages: [msg('user', 'hello', 't-1'), msg('assistant', 'hi', 't-1'), msg('system', 'be nice', 't-1')],
    });
    const record = sdk.conversation.record_message as ReturnType<typeof vi.fn>;
    expect(record.mock.calls.map((c) => [c[0].role, c[0].scope_path])).toEqual([
      ['user', PATH], ['assistant', PATH], ['system', PATH],
    ]);
  });

  it('saveMessages without a path writes exactly the payloads it always did', async () => {
    const sdk = makeSdk();
    setAutoListen(sdk, false);
    await new SynapMemory({ sdk, userId: 'alice' }).saveMessages({
      messages: [msg('user', 'hello', 't-1'), msg('system', 'be nice', 't-1')],
    });
    const record = sdk.conversation.record_message as ReturnType<typeof vi.fn>;
    expect(record.mock.calls.map((c) => c[0])).toEqual([
      { conversation_id: 't-1', role: 'user', content: 'hello', user_id: 'alice' },
      { conversation_id: 't-1', role: 'system', content: 'be nice', user_id: 'alice' },
    ]);
  });

  it('recall (a conversation-only read) still works and asks for the conversation id alone', async () => {
    const sdk = makeSdk();
    const mem = new SynapMemory({ sdk, userId: 'alice', scopePath: PATH });
    const out = await mem.recall({ threadId: 't-1' } as never);
    expect(out.messages.length).toBeGreaterThan(0);
    expect(callOf(sdk.conversation.context.get_context_for_prompt)).toEqual({ conversation_id: 't-1' });
  });
});

// The default path through a real Agent: Mastra's input processor does the
// fetch, the output processor writes the turns.
describe('through a real Agent', () => {
  interface Call { method: string; options: Record<string, unknown> }

  function agentWith(scopePath: Record<string, string> | undefined, listen: boolean) {
    const stream: Call[] = [];
    const state = { listening: false };
    const push = (method: string) => async (options: Record<string, unknown>) => {
      stream.push({ method, options });
    };
    const base = makeSdk();
    const sdk = {
      ...base,
      instance: {
        get is_listening() { return state.listening; },
        async listen() { state.listening = true; },
        send_message: push('send_message'),
        record_tool_call: push('record_tool_call'),
        record_tool_result: push('record_tool_result'),
        record_thinking: push('record_thinking'),
        end_session: push('end_session'),
      },
    };
    if (!listen) setAutoListen(sdk, false);
    const agent = new Agent({
      id: 'a', name: 'a', instructions: 'be nice',
      model: createMockModel({ mockText: 'MODEL_REPLY' }) as never,
      memory: new SynapMemory({ sdk: sdk as never, userId: 'alice', scopePath }),
    });
    return { agent, sdk, stream };
  }
  const MEM = { memory: { thread: 't1', resource: 'u1' } };

  it('the input processor fetches with scope_path; both turns go on the stream with it', async () => {
    const { agent, sdk, stream } = agentWith(PATH, true);
    await agent.generate('hello one', MEM);
    expect(callOf(sdk.fetch)['scope_path']).toEqual(PATH);
    const turns = stream.filter((c) => c.method === 'send_message');
    expect(turns.map((c) => [c.options['event_type'], c.options['scope_path']])).toEqual([
      ['user_message', PATH], ['assistant_message', PATH],
    ]);
  });

  it('with the stream off, both turns go over REST with scope_path', async () => {
    const { agent, sdk, stream } = agentWith(PATH, false);
    await agent.generate('hello one', MEM);
    expect(stream).toEqual([]);
    const record = sdk.conversation.record_message as ReturnType<typeof vi.fn>;
    expect(record.mock.calls.map((c) => [c[0].role, c[0].scope_path])).toEqual([
      ['user', PATH], ['assistant', PATH],
    ]);
  });

  it('without a path, nothing on the fetch, the stream or REST carries one', async () => {
    const live = agentWith(undefined, true);
    await live.agent.generate('hello one', MEM);
    expect(Object.keys(callOf(live.sdk.fetch))).not.toContain('scope_path');
    for (const c of live.stream) expect(Object.keys(c.options)).not.toContain('scope_path');

    const rest = agentWith(undefined, false);
    await rest.agent.generate('hello one', MEM);
    for (const c of (rest.sdk.conversation.record_message as ReturnType<typeof vi.fn>).mock.calls) {
      expect(Object.keys(c[0])).not.toContain('scope_path');
    }
  });
});
