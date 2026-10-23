/**
 * Compaction v3 in the SDK: the server's framed block is printed once, and
 * turns carry the agent's model. Python twin: python/tests/test_compaction_v3_block.py.
 */
import { describe, expect, it } from 'vitest';

import { withAgentModel } from '../agent-model.js';
import { ShortTermStore } from '../cache/short-term-store.js';
import { renderForPrompt } from '../context/local-prompt.js';

const BLOCK =
  '## Conversation so far (compacted: turns 1–2, 2026-10-17 09:00–09:01 UTC)\n' +
  '[CONTEXT COMPACTION, REFERENCE ONLY] A record of what was said earlier.\n\n' +
  '### Customer and order\nPriya reported order ORD-1 arrived broken.';

const bundle = (cc: Record<string, unknown>) => ({
  _bundle_type: 'compaction_update',
  conversation_context: {
    conversation_id: 'c1', compaction_id: 'k1', compacted_at: '2026-10-17T09:05:00+00:00', ...cc,
  },
});

function storeWithV2ThenV3(): ShortTermStore {
  const store = new ShortTermStore();
  store.applyCompaction(bundle({
    summary: 'old', factual_paragraph: 'OLD FACTS', conversational_paragraph: 'OLD TALK',
    compacted_at: '2026-10-17T09:00:00+00:00',
  }));
  store.applyCompaction(bundle({
    summary: BLOCK, context_block: BLOCK, factual_paragraph: '', conversational_paragraph: '',
    compaction_id: 'k2',
  }));
  return store;
}

describe('compaction v3 block', () => {
  it('a v3 compaction clears cached v2 paragraphs', () => {
    const entry = storeWithV2ThenV3().get('c1')!;
    expect(entry.contextBlock).toBe(BLOCK);
    expect(entry.factualParagraph).toBeNull();
    expect(entry.conversationalParagraph).toBeNull();
  });

  it('every style prints the block once', () => {
    const entry = storeWithV2ThenV3().get('c1')!;
    for (const style of ['structured', 'narrative', 'bullet_points']) {
      const text = String(renderForPrompt(entry, style).formatted_context);
      expect(text.split('## Conversation so far').length - 1).toBe(1);
      expect(text).not.toContain('OLD FACTS');
      expect(text).not.toContain('## Summary');
      expect(text).not.toContain('- Summary:');
    }
  });

  it('the response carries the block alone for history stores', () => {
    const response = renderForPrompt(storeWithV2ThenV3().get('c1')!);
    expect(response.context_block).toBe(BLOCK);
    expect(response.compaction_id).toBe('k2');
  });

  it('a summary that is the block is recognised', () => {
    const store = new ShortTermStore();
    store.applyCompaction(bundle({ summary: BLOCK }));
    expect(store.get('c1')!.contextBlock).toBe(BLOCK);
  });

  it('a v2 compaction renders as before', () => {
    const store = new ShortTermStore();
    store.applyCompaction(bundle({ summary: 'S', factual_paragraph: 'F' }));
    const text = String(renderForPrompt(store.get('c1')!).formatted_context);
    expect(text).toContain('## Summary\n\nS');
    expect(text).toContain('## Facts\n\nF');
  });

  it('agent model precedence', () => {
    const defaults = { model: 'gpt-4o', contextWindow: 128000 };
    expect(withAgentModel(undefined, {}, {})).toBeUndefined();
    expect(withAgentModel({ a: 1 }, {}, defaults)).toEqual({ a: 1, agent_model: 'gpt-4o', agent_context_window: 128000 });
    expect(withAgentModel(undefined, { model: 'gpt-5' }, defaults)!['agent_model']).toBe('gpt-5');
    expect(withAgentModel(undefined, { contextWindow: 400000 }, defaults, true)).toEqual({
      agent_model: 'gpt-4o', agent_context_window: '400000',
    });
  });
});
