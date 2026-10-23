// A real Mastra Agent, a mock model, and SynapMemory on the Agent's DEFAULT path.
//
// The unit tests call `saveMessages` and `recall` by hand, which is how a memory
// that Mastra never calls its methods on looked healthy: against @mastra/core
// 1.26 the default `agent.generate` / `agent.stream` path reads history through
// the MessageHistory input processor and writes through its output hook, and
// asks the memory for neither `recall`, `getSystemMessage` nor `saveMessages`.
// Everything here goes through `Agent`, and asserts on what the MODEL received
// and on what reached the fake SDK.

import { Agent } from '@mastra/core/agent';
import { Mastra } from '@mastra/core';
import { InMemoryStore } from '@mastra/core/storage';
import { createMockModel } from '@mastra/core/test-utils/llm-mock';
import { describe, expect, it, vi } from 'vitest';

import { SynapMemory } from '../memory.js';
import { setAutoListen } from '../stream-events.js';

const COMPACTION = '## Conversation so far (compacted at 6)\nSYNAP_COMPACTION_TEXT';

interface Call { method: string; options: Record<string, unknown> }

/**
 * A Synap SDK whose stream is closed until `listen()` is called, the way the
 * real one is. `compaction` is what `get_context_for_prompt` answers with.
 */
function makeSdk(opts: {
  compaction?: boolean;
  recent?: Array<{ role: string; content: string }>;
} = {}) {
  const stream: Call[] = [];
  const rest: Array<Record<string, unknown>> = [];
  const state = { listening: false, listens: 0, fetches: 0 };
  const push = (method: string) => async (options: Record<string, unknown>) => {
    stream.push({ method, options });
  };
  const sdk = {
    fetch: vi.fn(async () => {
      state.fetches += 1;
      return { formatted_context: 'SYNAP_LONG_TERM_FACT' };
    }),
    conversation: {
      record_message: vi.fn(async (o: Record<string, unknown>) => { rest.push(o); return {}; }),
      context: {
        get_context_for_prompt: vi.fn(async () => ({
          formatted_context: '',
          available: true,
          recent_messages: opts.recent ?? [],
          recent_message_count: 0,
          total_message_count: 0,
          context_block: opts.compaction === true ? COMPACTION : null,
          history_block: null,
        })),
      },
    },
    memories: { create: vi.fn(async () => ({ ingestion_id: 'i' })) },
    instance: {
      get is_listening() { return state.listening; },
      async listen() { state.listens += 1; state.listening = true; },
      send_message: push('send_message'),
      record_tool_call: push('record_tool_call'),
      record_tool_result: push('record_tool_result'),
      record_thinking: push('record_thinking'),
      end_session: push('end_session'),
    },
  };
  return { sdk, stream, rest, state };
}

function makeAgent(sdk: unknown, opts: { storage?: boolean } = {}) {
  const prompts: string[] = [];
  const model = createMockModel({
    mockText: 'MODEL_REPLY',
    spyGenerate: (p: { prompt: unknown }) => prompts.push(JSON.stringify(p.prompt)),
    spyStream: (p: { prompt: unknown }) => prompts.push(JSON.stringify(p.prompt)),
  });
  const agent = new Agent({
    id: 'a', name: 'a', instructions: 'be nice', model: model as never,
    memory: new SynapMemory({ sdk: sdk as never, userId: 'u1' }),
  });
  if (opts.storage === true) new Mastra({ agents: { a: agent }, storage: new InMemoryStore() });
  return { agent, prompts };
}

const MEM = { memory: { thread: 't1', resource: 'u1' } };

describe('no Mastra storage attached', () => {
  it('generate works (it used to throw "Memory requires a storage provider")', async () => {
    const { sdk } = makeSdk();
    const { agent } = makeAgent(sdk);
    const result = await agent.generate('hello one', MEM);
    expect(result.text).toBe('MODEL_REPLY');
  });

  it('the model receives Synap\'s long-term context', async () => {
    const { sdk } = makeSdk();
    const { agent, prompts } = makeAgent(sdk);
    await agent.generate('hello one', MEM);
    expect(prompts[0]).toContain('SYNAP_LONG_TERM_FACT');
    expect(sdk.fetch).toHaveBeenCalledOnce();
  });

  it('opens the stream on the first turn and records both turns on it, not over REST', async () => {
    const { sdk, stream, rest, state } = makeSdk();
    const { agent } = makeAgent(sdk);
    await agent.generate('hello one', MEM);
    expect(state.listens).toBe(1);
    const turns = stream.filter((c) => c.method === 'send_message').map((c) => [c.options['event_type'], c.options['content']]);
    expect(turns).toEqual([['user_message', 'hello one'], ['assistant_message', 'MODEL_REPLY']]);
    expect(rest).toEqual([]);
    expect(stream[0]?.options).toMatchObject({ conversation_id: 't1', user_id: 'u1' });
  });

  it('keeps using that stream on later turns', async () => {
    const { sdk, stream, state } = makeSdk();
    const { agent } = makeAgent(sdk);
    await agent.generate('hello one', MEM);
    await agent.generate('hello two', MEM);
    expect(state.listens).toBe(1);
    expect(stream.filter((c) => c.method === 'send_message')).toHaveLength(4);
  });

  it('SYNAP_AUTO_LISTEN off (setAutoListen) writes the turns over REST instead, once each', async () => {
    const { sdk, stream, rest, state } = makeSdk();
    setAutoListen(sdk, false);
    const { agent } = makeAgent(sdk);
    await agent.generate('hello one', MEM);
    expect(state.listens).toBe(0);
    expect(stream).toEqual([]);
    expect(rest.map((r) => [r['role'], r['content']])).toEqual([['user', 'hello one'], ['assistant', 'MODEL_REPLY']]);
    // B2C: no empty customer_id key.
    expect(rest[0]).not.toHaveProperty('customer_id');
  });

  it('with no compaction, Synap\'s own turns are the history', async () => {
    const { sdk } = makeSdk({ recent: [
      { role: 'user', content: 'SYNAP_TURN_Q' }, { role: 'assistant', content: 'SYNAP_TURN_A' },
    ] });
    const { agent, prompts } = makeAgent(sdk);
    await agent.generate('hello two', MEM);
    expect(prompts[0]).toContain('SYNAP_TURN_Q');
    expect(prompts[0]).toContain('SYNAP_TURN_A');
    expect(prompts[0]).toContain('hello two');
  });

  it('a turn Synap already recorded is not sent twice', async () => {
    const { sdk } = makeSdk({ recent: [{ role: 'user', content: 'hello two' }] });
    const { agent, prompts } = makeAgent(sdk);
    await agent.generate('hello two', MEM);
    expect(prompts[0]!.split('hello two').length - 1).toBe(1);
  });

  it('with a compaction, the model sees it, then the current message', async () => {
    const { sdk } = makeSdk({ compaction: true, recent: [
      { role: 'user', content: 'POST_COMPACTION_Q' }, { role: 'assistant', content: 'POST_COMPACTION_A' },
    ] });
    const { agent, prompts } = makeAgent(sdk);
    await agent.generate('hello three', MEM);
    const p = prompts[0]!;
    expect(p).toContain('<synap_conversation_history>');
    expect(p).toContain('SYNAP_COMPACTION_TEXT');
    expect(p.indexOf('SYNAP_COMPACTION_TEXT')).toBeLessThan(p.indexOf('POST_COMPACTION_Q'));
    expect(p.indexOf('POST_COMPACTION_A')).toBeLessThan(p.indexOf('hello three'));
  });

  it('stream() takes the same path', async () => {
    const { sdk, stream, state } = makeSdk({ compaction: true });
    const { agent, prompts } = makeAgent(sdk);
    const out = await agent.stream('hello one', MEM);
    await out.consumeStream?.();
    await out.text;
    expect(prompts[0]).toContain('SYNAP_COMPACTION_TEXT');
    expect(prompts[0]).toContain('SYNAP_LONG_TERM_FACT');
    expect(state.listens).toBe(1);
    expect(stream.filter((c) => c.method === 'send_message')).toHaveLength(2);
  });

  it('the compaction is never written back as something somebody said', async () => {
    const { sdk, stream } = makeSdk({ compaction: true });
    const { agent } = makeAgent(sdk);
    await agent.generate('hello one', MEM);
    await agent.generate('hello two', MEM);
    for (const c of stream) {
      expect(JSON.stringify(c.options)).not.toContain('SYNAP_COMPACTION_TEXT');
    }
  });

  it('injectSystemContext: false skips the long-term fetch', async () => {
    const { sdk } = makeSdk();
    const prompts: string[] = [];
    const model = createMockModel({ mockText: 'ok', spyGenerate: (p: { prompt: unknown }) => prompts.push(JSON.stringify(p.prompt)) });
    const agent = new Agent({
      id: 'a', name: 'a', instructions: 'i', model: model as never,
      memory: new SynapMemory({ sdk: sdk as never, userId: 'u1', injectSystemContext: false }),
    });
    await agent.generate('hi', MEM);
    expect(sdk.fetch).not.toHaveBeenCalled();
    expect(prompts[0]).not.toContain('SYNAP_LONG_TERM_FACT');
  });

  it('a Synap read that fails does not end the run', async () => {
    const { sdk } = makeSdk();
    sdk.conversation.context.get_context_for_prompt.mockRejectedValue(new Error('500'));
    sdk.fetch.mockRejectedValue(new Error('500'));
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const { agent } = makeAgent(sdk);
    const result = await agent.generate('hello', MEM);
    expect(result.text).toBe('MODEL_REPLY');
    vi.restoreAllMocks();
  });
});

describe('a Mastra storage is attached', () => {
  it('Mastra\'s own history reaches the model when Synap has no compaction', async () => {
    const { sdk } = makeSdk();
    const { agent, prompts } = makeAgent(sdk, { storage: true });
    await agent.generate('hello one', MEM);
    await agent.generate('hello two', MEM);
    expect(prompts[1]).toContain('hello one');
    expect(prompts[1]).not.toContain('<synap_conversation_history>');
  });

  it('Synap\'s compaction replaces Mastra\'s earlier turns once it exists', async () => {
    const { sdk } = makeSdk({ compaction: true, recent: [{ role: 'user', content: 'POST_COMPACTION_Q' }] });
    const { agent, prompts } = makeAgent(sdk, { storage: true });
    await agent.generate('hello one', MEM);
    await agent.generate('hello two', MEM);
    const p = prompts[1]!;
    expect(p).toContain('SYNAP_COMPACTION_TEXT');
    expect(p).toContain('POST_COMPACTION_Q');
    expect(p).not.toContain('hello one');
    expect(p).not.toContain('MODEL_REPLY');
    expect(p).toContain('hello two');
  });

  it('records the turns on the stream, once each, with Mastra persisting its own copy', async () => {
    const { sdk, stream, rest } = makeSdk();
    const { agent } = makeAgent(sdk, { storage: true });
    await agent.generate('hello one', MEM);
    expect(stream.filter((c) => c.method === 'send_message')).toHaveLength(2);
    expect(rest).toEqual([]);
  });

  it('savePerStep does not record a turn twice', async () => {
    const { sdk, stream } = makeSdk();
    const { agent } = makeAgent(sdk, { storage: true });
    await agent.generate('hello one', { ...MEM, savePerStep: true } as never);
    const contents = stream.filter((c) => c.method === 'send_message').map((c) => c.options['content']);
    expect(contents.filter((c) => c === 'hello one')).toHaveLength(1);
    expect(contents.filter((c) => c === 'MODEL_REPLY')).toHaveLength(1);
  });
});
