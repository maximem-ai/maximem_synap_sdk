/**
 * Gaps found by translating the Python docs into JavaScript.
 *
 * Each one is a place where the JS surface silently accepted less than Python's,
 * so a translated snippet either did not compile or dropped data on the wire.
 * The docs are the only thing that exercised these paths, hence the file.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createCreditsNamespace } from '../credits/interface.js';
import type { HttpTransport } from '../transport/http.js';

function recordingTransport() {
  const calls: { name: string; opts: unknown }[] = [];
  const t = {
    request: async (name: string, opts?: unknown) => {
      calls.push({ name, opts });
      return {};
    },
  } as unknown as HttpTransport;
  return { transport: t, calls };
}

beforeEach(() => { vi.resetModules(); });

describe('credits.get_ledger mirrors Python', () => {
  it('forwards entry_type, from and to, and defaults limit to 100', async () => {
    const { transport, calls } = recordingTransport();
    const credits = createCreditsNamespace(transport);

    await credits.get_ledger({
      entry_type: 'debit',
      from_time: new Date('2026-01-01T00:00:00.000Z'),
      to_time: '2026-02-01T00:00:00Z',
    });

    expect(calls[0]?.name).toBe('credits_ledger');
    expect((calls[0]?.opts as { query: Record<string, unknown> }).query).toEqual({
      limit: 100,                       // Python's default, not the 50 we had
      offset: 0,
      entry_type: 'debit',
      from: '2026-01-01T00:00:00.000Z', // Python sends from_time.isoformat()
      to: '2026-02-01T00:00:00Z',
    });
  });

  it('omits the filters entirely when they are not given', async () => {
    const { transport, calls } = recordingTransport();
    await createCreditsNamespace(transport).get_ledger();
    expect((calls[0]?.opts as { query: Record<string, unknown> }).query)
      .toEqual({ limit: 100, offset: 0 });
  });
});

describe('instance.send_message mirrors Python', () => {
  it('forwards tool_name and JSON-encodes tool_args into tool_args_json', async () => {
    const events: Record<string, unknown>[] = [];

    // The real client is behind a dynamic import so grpc-js never loads on
    // Edge; mock the module rather than standing up a server.
    vi.doMock('../grpc/stream-client.js', () => ({
      GrpcStreamClient: class {
        async connect() {}
        async disconnect() {}
        get isConnected() { return true; }
        sendConversationEvent(e: Record<string, unknown>) { events.push(e); }
        sendSessionControl() {}
        sendContextUsed() {}
        sendContextAssembled() {}
      },
    }));

    const { createInstanceNamespace } = await import('../instance/interface.js');
    const { AnticipationCache } = await import('../context/anticipation-cache.js');

    const instance = createInstanceNamespace(
      () => ({ apiKey: 'synap_x', clientId: 'cli_x', instanceId: 'inst_0123456789abcdef' }),
      new AnticipationCache(),
    );

    await instance.listen();
    await instance.send_message({
      content: 'looking that up',
      role: 'assistant',
      conversation_id: '3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c',
      user_id: 'user_1',
      tool_name: 'get_weather',
      tool_args: { city: 'SF' },
    });

    expect(events).toHaveLength(1);
    expect(events[0]?.['tool_name']).toBe('get_weather');
    expect(events[0]?.['tool_args_json']).toBe('{"city":"SF"}');
  });

  it('sends empty strings when no tool is involved, as Python does', async () => {
    const events: Record<string, unknown>[] = [];
    vi.doMock('../grpc/stream-client.js', () => ({
      GrpcStreamClient: class {
        async connect() {}
        async disconnect() {}
        get isConnected() { return true; }
        sendConversationEvent(e: Record<string, unknown>) { events.push(e); }
        sendSessionControl() {}
        sendContextUsed() {}
        sendContextAssembled() {}
      },
    }));

    const { createInstanceNamespace } = await import('../instance/interface.js');
    const { AnticipationCache } = await import('../context/anticipation-cache.js');
    const instance = createInstanceNamespace(
      () => ({ apiKey: 'synap_x', clientId: 'cli_x', instanceId: 'inst_0123456789abcdef' }),
      new AnticipationCache(),
    );

    await instance.listen();
    await instance.send_message({ content: 'hello', user_id: 'user_1' });

    expect(events[0]?.['tool_name']).toBe('');
    expect(events[0]?.['tool_args_json']).toBe('');
  });
});

describe('retryPolicy: null disables retries, as in Python', () => {
  it('makes exactly one attempt', async () => {
    const { HttpTransport } = await import('../transport/http.js');
    let calls = 0;
    const t = new HttpTransport({
      credentials: { apiKey: 'synap_x', clientId: 'cli_x', instanceId: 'inst_0123456789abcdef' },
      retryPolicy: null,
      fetchImpl: (async () => {
        calls += 1;
        return new Response('{"detail":"boom"}', { status: 503 });
      }) as unknown as typeof fetch,
    });

    await t.request('whoami').catch(() => {});
    // Python: SDKConfig(retry_policy=None) leaves max_attempts at 1.
    expect(calls).toBe(1);
  });

  it('still retries three times with the default policy', async () => {
    const { HttpTransport } = await import('../transport/http.js');
    let calls = 0;
    const t = new HttpTransport({
      credentials: { apiKey: 'synap_x', clientId: 'cli_x', instanceId: 'inst_0123456789abcdef' },
      retryPolicy: { backoffBase: 0, backoffMax: 0, backoffJitter: false },
      fetchImpl: (async () => {
        calls += 1;
        return new Response('{"detail":"boom"}', { status: 503 });
      }) as unknown as typeof fetch,
    });

    await t.request('whoami').catch(() => {});
    expect(calls).toBe(3);
  });
});
