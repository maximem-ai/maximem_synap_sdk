/**
 * Surface Python had and this SDK did not.
 *
 * Each of these was documented as "Python only" in the customer docs, which is
 * the wrong place to fix a missing feature. The tests assert the JS behaviour
 * matches Python's, so the docs can simply stop mentioning a gap.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { SynapClient, DEFAULT_TIMEOUTS } from '../index.js';
import { verbatimOverlayEnabled } from '../context/overlay.js';
import { renderForPrompt, renderCompacted } from '../context/local-prompt.js';
import type { ShortTermEntry } from '../cache/short-term-store.js';

function entry(over: Partial<ShortTermEntry> = {}): ShortTermEntry {
  return {
    conversationId: '3f6b1a2c-4d5e-6f7a-8b9c-0d1e2f3a4b5c',
    summary: null,
    factualParagraph: null,
    conversationalParagraph: null,
    currentState: {},
    keyExtractions: {},
    compactionId: null,
    compactedAt: null,
    endTimestamp: null,
    recentTurns: [],
    lastActivityAt: 0,
    ...over,
  };
}

describe('TimeoutConfig.stream_idle', () => {
  it('exists, and defaults to 60s as in Python', () => {
    expect(DEFAULT_TIMEOUTS.streamIdle).toBe(60.0);
  });
});

describe('st_verbatim_overlay as a config field', () => {
  const saved = process.env['SYNAP_ST_VERBATIM_OVERLAY'];
  beforeEach(() => { delete process.env['SYNAP_ST_VERBATIM_OVERLAY']; });
  afterEach(() => {
    if (saved === undefined) delete process.env['SYNAP_ST_VERBATIM_OVERLAY'];
    else process.env['SYNAP_ST_VERBATIM_OVERLAY'] = saved;
  });

  it('defaults on, and the option can turn it off', () => {
    expect(verbatimOverlayEnabled(undefined)).toBe(true);
    expect(verbatimOverlayEnabled(false)).toBe(false);
  });

  it('the env var still wins over the option, either way', () => {
    process.env['SYNAP_ST_VERBATIM_OVERLAY'] = '0';
    expect(verbatimOverlayEnabled(true)).toBe(false);
    process.env['SYNAP_ST_VERBATIM_OVERLAY'] = '1';
    expect(verbatimOverlayEnabled(false)).toBe(true);
  });
});

describe('record_thinking carries step_index and thought_type', () => {
  beforeEach(() => { vi.resetModules(); });

  it('folds both into metadata, exactly as Python does', async () => {
    const events: Record<string, unknown>[] = [];
    vi.doMock('../grpc/stream-client.js', () => ({
      GrpcStreamClient: class {
        async connect() {} async disconnect() {}
        get isConnected() { return true; }
        sendConversationEvent(e: Record<string, unknown>) { events.push(e); }
        sendSessionControl() {} sendContextUsed() {} sendContextAssembled() {}
      },
    }));
    const { createInstanceNamespace } = await import('../instance/interface.js');
    const { AnticipationCache } = await import('../context/anticipation-cache.js');
    const instance = createInstanceNamespace(
      () => ({ apiKey: 'synap_x', clientId: 'cli_x', instanceId: 'inst_0123456789abcdef' }),
      new AnticipationCache(),
    );
    await instance.listen();
    await instance.record_thinking({
      content: 'checking the calendar',
      step_index: 2,
      thought_type: 'tool_selection',
      metadata: { trace: 'abc' },
    });

    const md = events[0]?.['metadata'] as Record<string, string>;
    expect(md['step_index']).toBe('2');
    expect(md['thought_type']).toBe('tool_selection');
    expect(md['trace']).toBe('abc');       // caller metadata preserved
  });
});

describe('logger option', () => {
  it('receives diagnostics instead of the console', async () => {
    const seen: [string, string][] = [];
    const sink = (level: string, message: string): void => { seen.push([level, message]); };

    // `a` must be REGISTERED (no _force_new) for `b` to collide with it.
    // A second client on the same instance with a different key is the one
    // thing the registry warns about.
    const a = new SynapClient({
      apiKey: 'synap_a', instanceId: 'inst_00000000000000aa', logger: sink,
    });
    const b = new SynapClient({
      apiKey: 'synap_b', instanceId: 'inst_00000000000000aa', logger: sink,
    });

    expect(seen.length).toBeGreaterThan(0);
    expect(seen[0]?.[0]).toBe('warn');
    expect(seen[0]?.[1]).toContain('already running');

    void b;
    await a.shutdown();   // leave the process registry as we found it
  });
});

describe('SDK-authoritative short-term rendering', () => {
  it('renders the structured style with Python\'s exact headings', () => {
    const out = renderForPrompt(entry({
      summary: 'User is booking a flight.',
      currentState: { destination: 'Tokyo' },
      keyExtractions: { preferences: [{ content: 'window seat' }] },
      recentTurns: [{ role: 'user', content: 'any window seats?', timestamp: '2026-01-01T00:00:00Z' }],
    }));
    expect(out.available).toBe(true);
    expect(out.formatted_context).toContain('## Summary');
    expect(out.formatted_context).toContain('## Current State');
    expect(out.formatted_context).toContain('- destination: Tokyo');
    expect(out.formatted_context).toContain('### preferences');
    expect(out.formatted_context).toContain('- window seat');
    expect(out.formatted_context).toContain('## Recent Turns (1)');
    expect(out.formatted_context).toContain('**user**: any window seats?');
    expect(out.recent_message_count).toBe(1);
  });

  it('supports narrative and bullet_points, and falls back on an unknown style', () => {
    const e = entry({ summary: 'S', recentTurns: [{ role: 'user', content: 'hi', timestamp: '2026-01-01T00:00:00Z' }] });
    expect(renderForPrompt(e, 'narrative').formatted_context).toContain('Recent exchanges:');
    expect(renderForPrompt(e, 'bullet_points').formatted_context).toContain('- Summary: S');
    // Unknown style falls back to structured, as Python does.
    expect(renderForPrompt(e, 'nonsense').formatted_context).toContain('## Summary');
  });

  it('reports unavailable for a cold entry', () => {
    const out = renderForPrompt(entry());
    expect(out.available).toBe(false);
    expect(out.formatted_context).toBeNull();
  });

  it('computes compaction age from compactedAt', () => {
    const out = renderForPrompt(
      entry({ compactionId: 'c1', compactedAt: '2026-01-01T00:00:00Z' }),
      'structured',
      () => Date.parse('2026-01-01T00:01:30Z'),
    );
    expect(out.compaction_age_seconds).toBe(90);
  });

  it('renderCompacted returns null until a compaction has landed', () => {
    expect(renderCompacted(entry({ recentTurns: [{ role: 'user', content: 'x', timestamp: '2026-01-01T00:00:00Z' }] }))).toBeNull();
    const out = renderCompacted(entry({ compactionId: 'c1', summary: 'S' }));
    expect(out?.['compaction_id']).toBe('c1');
    expect(String(out?.['compacted_context'])).toContain('## Summary');
  });
});

describe('api_base_url, the name Python uses', () => {
  it('is accepted alongside baseUrl and reaches the same host', async () => {
    const urls: string[] = [];
    const capture = (async (input: unknown) => {
      urls.push(String(input));
      return new Response('{}', { status: 200 });
    }) as unknown as typeof fetch;

    const viaPythonName = new SynapClient({
      apiKey: 'synap_x', instanceId: 'inst_00000000000000ab', _force_new: true,
      api_base_url: 'https://example.invalid', fetchImpl: capture,
    });
    const viaJsName = new SynapClient({
      apiKey: 'synap_x', instanceId: 'inst_00000000000000ac', _force_new: true,
      baseUrl: 'https://example.invalid', fetchImpl: capture,
    });

    await viaPythonName.credits.get_balance();
    await viaJsName.credits.get_balance();

    expect(urls).toHaveLength(2);
    expect(urls[0]).toContain('https://example.invalid');
    expect(urls[0]).toBe(urls[1]);

    await viaPythonName.shutdown();
    await viaJsName.shutdown();
  });
});
