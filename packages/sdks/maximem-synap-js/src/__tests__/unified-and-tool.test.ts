import { describe, it, expect, vi } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { mergeScopeResults, formatForPrompt } from '../context/unified.js';
import { buildTool, defaultToolDescription, toolInputSchema, invokeScopeFetch } from '../tool/as-tool.js';
import { InvalidInputError } from '../errors.js';
import type { Json, RawContext } from '../context/types.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const corpusPath = path.resolve(here, '../../../CONTRACT/conformance/unified_and_tool.json');

interface Corpus {
  fixtures: Record<string, RawContext>;
  merge_cases: Array<{
    name: string;
    scopes: Array<[string, string]>;
    include_scope: boolean;
    expected: {
      scopes_queried: string[];
      total_items: number;
      scope_map: Record<string, string>;
      fact_ids: string[];
      preference_ids: string[];
      formatted_context: string;
    };
  }>;
  tool_cases: Array<{
    scope: string;
    has_conversation_id: boolean;
    description: string;
    schema: Json;
  }>;
}

describe('unified fetch and as_tool parity with Python', () => {
  if (!existsSync(corpusPath)) return;
  const corpus = JSON.parse(readFileSync(corpusPath, 'utf8')) as Corpus;

  describe('merge + format_for_prompt', () => {
    it.each(corpus.merge_cases)('$name matches Python byte for byte', (testCase) => {
      const scopeResults = testCase.scopes.map(
        ([name, key]) => [name, corpus.fixtures[key] as RawContext] as const,
      );
      const merged = mergeScopeResults(scopeResults);
      const formatted = formatForPrompt(merged, {
        includeScope: testCase.include_scope,
        includeConversationContext: true,
      });

      expect(merged.scopes_queried).toEqual(testCase.expected.scopes_queried);
      expect(merged.total_items).toBe(testCase.expected.total_items);
      expect(merged.scope_map).toEqual(testCase.expected.scope_map);
      expect(merged.facts.map((f) => f.id)).toEqual(testCase.expected.fact_ids);
      expect(merged.preferences.map((p) => p.id)).toEqual(testCase.expected.preference_ids);
      // The whole point: this string goes straight into a system prompt.
      expect(formatted).toBe(testCase.expected.formatted_context);
    });

    it('keeps the first scope on a duplicate id', () => {
      const merged = mergeScopeResults([
        ['user', corpus.fixtures['scope_a'] as RawContext],
        ['customer', corpus.fixtures['scope_b'] as RawContext],
      ]);
      const f1 = merged.facts.find((f) => f.id === 'f1');
      expect(f1?.content).toBe('Prefers window seats');
      expect(merged.scope_map['f1']).toBe('user');
    });

    it('appends conversation history only when it carries formatted text', () => {
      const base = mergeScopeResults([['user', corpus.fixtures['scope_a'] as RawContext]]);
      base.conversation_context = { formatted_context: '' };
      expect(formatForPrompt(base)).not.toContain('### Conversation History');
      base.conversation_context = { formatted_context: 'turn 1' };
      expect(formatForPrompt(base)).toContain('### Conversation History\nturn 1');
    });

    it('drops conversation history when includeConversationContext is false', () => {
      const base = mergeScopeResults([['user', corpus.fixtures['scope_a'] as RawContext]]);
      base.conversation_context = { formatted_context: 'turn 1' };
      expect(formatForPrompt(base, { includeConversationContext: false }))
        .not.toContain('### Conversation History');
    });

    it('returns an empty string when there is nothing at all', () => {
      expect(formatForPrompt(mergeScopeResults([]))).toBe('');
    });

    it('renders the profile and previous-conversation blocks', () => {
      const merged = mergeScopeResults([]);
      merged.profile = {
        attributes: { name: { value: 'Ada' }, langs: { value: ['en', 'de'] }, blank: { value: null } },
        overview: 'Long-haul flyer.',
      };
      merged.conversations = [{
        started_at: '2026-05-04T10:00:00Z',
        conversation_type: 'support_call',
        summary: { narrative_summary: 'Asked about a refund.' },
        classification: { primary_category: 'billing', objective: 'refund' },
        summary_status: 'available',
      }];
      const out = formatForPrompt(merged);
      expect(out).toContain('## Caller Profile');
      expect(out).toContain('- name: Ada');
      expect(out).toContain('- langs: en, de');
      // Python rstrips, so an empty value leaves no trailing space.
      expect(out).toContain('- blank:\n');
      expect(out).toContain('Long-haul flyer.');
      expect(out).toContain('### Call on 2026-05-04 (support_call)');
      expect(out).toContain('Overview: Asked about a refund.');
      expect(out).toContain('Outcome: billing / refund');
    });

    it('falls back to Status when a conversation has neither overview nor outcome', () => {
      const merged = mergeScopeResults([]);
      merged.conversations = [{ started_at: null, summary_status: 'pending' }];
      const out = formatForPrompt(merged);
      expect(out).toContain('### Call on unknown date');
      expect(out).toContain('Status: pending');
    });
  });

  describe('as_tool', () => {
    it.each(corpus.tool_cases)(
      'scope=$scope hasConversationId=$has_conversation_id matches Python',
      (testCase) => {
        expect(defaultToolDescription(testCase.scope as 'user')).toBe(testCase.description);
        expect(
          toolInputSchema(testCase.scope as 'user', {
            hasConversationId: testCase.has_conversation_id,
          }),
        ).toEqual(testCase.schema);
      },
    );

    const target = () => ({
      fetch: vi.fn(async () => ({ formatted_context: 'ctx', scopes_queried: ['user'], total_items: 1 })),
      conversation: { context: { fetch: vi.fn(async () => ({ facts: [{ id: 'f' }] })) } },
      user: { context: { fetch: vi.fn(async () => ({ facts: [{ id: 'f' }] })) } },
      customer: { context: { fetch: vi.fn(async () => ({})) } },
      client: { context: { fetch: vi.fn(async () => ({})) } },
    });

    it('never exposes a scope identifier in the schema', () => {
      // This is the whole reason the helper exists: an LLM that can pass
      // user_id can read another user's memory.
      for (const testCase of corpus.tool_cases) {
        const props = Object.keys(
          (testCase.schema as { properties: Json }).properties,
        );
        expect(props).not.toContain('user_id');
        expect(props).not.toContain('customer_id');
      }
    });

    it('wraps in the OpenAI shape by default', () => {
      const tool = buildTool(target() as never, { scope: 'user', user_id: 'u1' });
      expect(tool).toMatchObject({ type: 'function' });
      expect((tool as { function: { name: string } }).function.name).toBe('synap_fetch_user_context');
      expect(typeof tool.handler).toBe('function');
    });

    it('wraps in the Anthropic shape on request', () => {
      const tool = buildTool(target() as never, { scope: 'unified', user_id: 'u1', style: 'anthropic' });
      expect(tool).toHaveProperty('input_schema');
      expect(tool).not.toHaveProperty('type');
      expect((tool as { name: string }).name).toBe('synap_fetch_unified_context');
    });

    it('rejects an unknown scope and an unknown style', () => {
      expect(() => buildTool(target() as never, { scope: 'nope' })).toThrow(InvalidInputError);
      expect(() => buildTool(target() as never, { scope: 'user', user_id: 'u', style: 'cohere' }))
        .toThrow(/openai/);
    });

    it('requires the identifier its scope depends on', () => {
      expect(() => buildTool(target() as never, { scope: 'user' })).toThrow(/requires user_id/);
      expect(() => buildTool(target() as never, { scope: 'customer' })).toThrow(/requires customer_id/);
    });

    it('warns rather than throwing for conversation scope without user_id', () => {
      const warn = vi.fn();
      const tool = buildTool(target() as never, { scope: 'conversation' }, warn);
      expect(tool).toBeDefined();
      expect(warn).toHaveBeenCalledWith(expect.stringContaining('per-user filtering'));
    });

    it('merges the closed-over ids into the handler call', async () => {
      const t = target();
      const tool = buildTool(t as never, { scope: 'user', user_id: 'u1', customer_id: 'c1' });
      await tool.handler({ search_query: ['q'] });
      expect(t.user.context.fetch).toHaveBeenCalledWith(
        expect.objectContaining({ user_id: 'u1', customer_id: 'c1', search_query: ['q'] }),
      );
    });

    it('defaults max_results to 10 and treats 0 as unset, like Python', async () => {
      const t = target();
      const tool = buildTool(t as never, { scope: 'user', user_id: 'u1' });
      await tool.handler({});
      expect(t.user.context.fetch).toHaveBeenCalledWith(expect.objectContaining({ max_results: 10 }));
      await tool.handler({ max_results: 0 });
      expect(t.user.context.fetch).toHaveBeenLastCalledWith(
        expect.objectContaining({ max_results: 10 }),
      );
    });

    it('returns the missing-conversation-id case as data, not an exception', async () => {
      const t = target();
      const result = await invokeScopeFetch({
        target: t as never, scope: 'conversation',
        userId: 'u', customerId: undefined, conversationId: undefined, callArgs: {},
      });
      expect(result).toEqual({ error: 'conversation_id is required', items: [] });
      expect(t.conversation.context.fetch).not.toHaveBeenCalled();
    });

    it('projects the unified scope down to three keys', async () => {
      const t = target();
      const tool = buildTool(t as never, { scope: 'unified', user_id: 'u1' });
      const out = await tool.handler({});
      expect(Object.keys(out).sort()).toEqual(['formatted_context', 'scopes_queried', 'total_items']);
    });

    it('always returns all five collections for a scoped fetch', async () => {
      const t = target();
      const tool = buildTool(t as never, { scope: 'customer', customer_id: 'c1' });
      const out = await tool.handler({});
      expect(Object.keys(out).sort()).toEqual(
        ['emotions', 'episodes', 'facts', 'preferences', 'temporal_events'],
      );
    });
  });
});
