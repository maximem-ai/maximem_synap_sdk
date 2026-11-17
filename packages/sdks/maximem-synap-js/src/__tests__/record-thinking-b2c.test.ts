/**
 * A6-03: instance.record_thinking refuses a customer_id on a B2C instance before
 * writing, as the Python SDK's record_thinking does (`_check_customer_id`).
 * It used to write straight to the stream, so the server refused the event
 * where nothing surfaced it to the caller.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

type Event = Record<string, unknown>;
let written: Event[] = [];

beforeEach(() => {
  vi.resetModules();
  written = [];
  vi.doMock('../grpc/stream-client.js', () => ({
    GrpcStreamClient: class {
      async connect() {}
      async disconnect() {}
      get isConnected() { return true; }
      sendConversationEvent(e: Event) { written.push(e); }
      sendSessionControl() {}
      sendContextUsed() {}
      sendContextAssembled() {}
    },
  }));
});

async function client(isolation: string) {
  const { SynapClient } = await import('../client.js');
  const fetchImpl = (async (url: string) => {
    const p = new URL(url).pathname;
    const body = p.endsWith('/whoami')
      ? { client_id: 'c', instance_id: 'inst_0123456789abcdef', user_context_isolation: isolation }
      : {};
    return new Response(JSON.stringify(body), { status: 200 });
  }) as unknown as typeof fetch;
  const c = new SynapClient({ _force_new: true, apiKey: 'synap_k', fetchImpl });
  await c.initialize();
  await c.instance.listen();
  return c;
}

const THOUGHT = { content: 'plan: check the order', conversation_id: 'conv-1', user_id: 'u1' };

describe('instance.record_thinking and the B2C customer_id rule', () => {
  it('refuses a customer_id on a B2C instance before writing', async () => {
    const c = await client('equals_customer');
    const { CustomerIdNotAcceptedError } = await import('../errors.js');
    const err = await c.instance
      .record_thinking({ ...THOUGHT, customer_id: 'acme' } as never)
      .then(() => null, (e: unknown) => e);
    expect(err).toBeInstanceOf(CustomerIdNotAcceptedError);
    expect(written).toEqual([]);
    await c.shutdown();
  });

  it('writes the thought on a B2C instance when no customer_id is sent', async () => {
    const c = await client('equals_customer');
    await c.instance.record_thinking(THOUGHT as never);
    expect(written).toHaveLength(1);
    expect(written[0]!['event_type']).toBe('agent_thinking');
    await c.shutdown();
  });

  it('accepts a customer_id on a B2B instance', async () => {
    const c = await client('strict');
    await c.instance.record_thinking({ ...THOUGHT, customer_id: 'acme' } as never);
    expect(written).toHaveLength(1);
    await c.shutdown();
  });
});
