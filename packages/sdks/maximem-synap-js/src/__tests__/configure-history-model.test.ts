/**
 * configure({ history_search_model }) chooses the local model the SDK loads,
 * as Python's SDKConfig.history_search_model does. Until 0.5.19 only the
 * constructor took it, though the configure docs listed it for both SDKs.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

let loaded: string[] = [];

beforeEach(() => {
  vi.resetModules();
  loaded = [];
  delete process.env['SYNAP_SDK_LOCAL_MODEL'];
  vi.doMock('../cache/onnx-encoder.js', () => ({
    canRunLocalModel: () => true,
    loadOnnxEncoder: async (name: string) => { loaded.push(name); return null; },
  }));
  vi.doMock('../grpc/stream-client.js', () => ({
    GrpcStreamClient: class {
      async connect() {}
      async disconnect() {}
      get isConnected() { return true; }
      sendConversationEvent() {}
      sendSessionControl() {}
      sendContextUsed() {}
      sendContextAssembled() {}
    },
  }));
});

async function listenedClient(configure?: Record<string, unknown>) {
  const { SynapClient } = await import('../client.js');
  const c = new SynapClient({
    _force_new: true, apiKey: 'k', clientId: 'c', instanceId: 'inst_0123456789abcdef',
    fetchImpl: (async () => new Response('{}')) as never,
  });
  if (configure) c.configure(configure as never);
  await c.instance.listen();
  await new Promise((r) => setTimeout(r, 0));
  return c;
}

describe('configure({ history_search_model })', () => {
  it('loads the model it names', async () => {
    const c = await listenedClient({ history_search_model: 'onnx:org/custom@rev/model.onnx' });
    expect(loaded).toEqual(['onnx:org/custom@rev/model.onnx']);
    await c.shutdown();
  });

  it('accepts the camelCase spelling', async () => {
    const c = await listenedClient({ historySearchModel: 'onnx:org/other@rev/model.onnx' });
    expect(loaded).toEqual(['onnx:org/other@rev/model.onnx']);
    await c.shutdown();
  });

  it('null loads no model', async () => {
    const c = await listenedClient({ history_search_model: null });
    expect(loaded).toEqual([]);
    await c.shutdown();
  });

  it('without it, the default model loads (control)', async () => {
    const { SEMANTIC_MATCH } = await import('../behavior/contract.js');
    const c = await listenedClient();
    expect(loaded).toEqual([SEMANTIC_MATCH.model]);
    await c.shutdown();
  });
});
