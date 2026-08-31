import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { finalItemIds, resolveSource, buildAssembledEvent } from '../context/assembled.js';
import { SynapClient } from '../client.js';
import * as registry from '../registry.js';
import type { RawContext } from '../context/types.js';

beforeEach(() => registry.clear());
afterEach(() => registry.clear());

describe('context_assembled payload', () => {
  it('collects ids across every collection', () => {
    const response: RawContext = {
      facts: [{ id: 'f1' }, { id: 'f2' }],
      preferences: [{ item_id: 'p1' }],
      episodes: [{ id: 'e1' }],
      emotions: [],
      temporal_events: [{ id: 't1' }],
    };
    expect(finalItemIds(response).sort()).toEqual(['e1', 'f1', 'f2', 'p1', 't1']);
  });

  it('skips items with no usable id rather than emitting blanks', () => {
    expect(finalItemIds({ facts: [{ content: 'no id' }, { id: '' }] })).toEqual([]);
  });

  describe('assembly source', () => {
    it('defaults to cloud', () => {
      expect(resolveSource({}).source).toBe('cloud');
    });

    it('renames the local HTTP cache to its proto name', () => {
      // ResponseMetadata says `cache`; the proto wants `http_cache`.
      expect(resolveSource({ metadata: { source: 'cache' } }).source).toBe('http_cache');
    });

    it('infers cache_hit from sources that skipped the server', () => {
      for (const source of ['anticipation', 'anticipation_cache', 'http_cache', 'sdk_authoritative']) {
        expect(resolveSource({ metadata: { source } }).cacheHit, source).toBe(true);
      }
    });

    it('otherwise trusts the metadata flag', () => {
      expect(resolveSource({ metadata: { source: 'cloud', cache_hit: true } }).cacheHit).toBe(true);
      expect(resolveSource({ metadata: { source: 'cloud' } }).cacheHit).toBe(false);
    });

    it('lets an override win, for shapes with no metadata.source', () => {
      expect(resolveSource({ metadata: { source: 'cloud' } }, 'sdk_authoritative').source)
        .toBe('sdk_authoritative');
    });
  });

  it('carries short-term metadata from the conversation context', () => {
    const event = buildAssembledEvent({
      correlationId: 'corr-1',
      response: {
        facts: [{ id: 'f1' }],
        conversation_context: {
          compaction_id: 'comp-9',
          end_timestamp: '2026-01-01T00:00:00Z',
          recent_turns: [{ role: 'user' }, { role: 'assistant' }],
        },
        metadata: { source: 'anticipation', total_tokens: 42 },
      },
      conversationId: 'conv-1',
      userId: 'u1',
      customerId: 'c1',
      startedAt: Date.now() - 5,
      sdkVersion: '0.4.0',
    });
    expect(event.compaction_id).toBe('comp-9');
    expect(event.recent_turn_count).toBe(2);
    expect(event.compaction_end_timestamp).toBe('2026-01-01T00:00:00Z');
    expect(event.final_total_tokens).toBe(42);
    expect(event.cache_hit).toBe(true);
    expect(event.assembly_duration_ms).toBeGreaterThanOrEqual(0);
    expect(event.final_item_ids).toEqual(['f1']);
  });

  it('carries ids and counts only, never content', () => {
    // Privacy: the proto forbids prompt or item content on this event.
    const event = buildAssembledEvent({
      correlationId: 'c',
      response: { facts: [{ id: 'f1', content: 'SECRET USER DETAIL' }] },
      startedAt: Date.now(),
      sdkVersion: '0.4.0',
    });
    expect(JSON.stringify(event)).not.toContain('SECRET USER DETAIL');
  });
});

describe('emission is never load-bearing', () => {
  it('a fetch succeeds while not listening, so nothing is emitted', async () => {
    const c = new SynapClient({
      apiKey: 'k',
      _force_new: true,
      fetchImpl: (async () =>
        new Response(JSON.stringify({ context: { facts: [{ id: 'f1', content: 'x' }] } }), {
          status: 200, headers: { 'content-type': 'application/json' },
        })) as unknown as typeof fetch,
    });
    expect(c.instance.is_listening).toBe(false);
    const ctx = await c.user.context.fetch({ user_id: 'u1' });
    // The events go over the Listen stream; with no stream there is nowhere to
    // put them, and that must not affect the fetch.
    expect(ctx.facts?.[0]?.content).toBe('x');
    await c.shutdown();
  });
});
