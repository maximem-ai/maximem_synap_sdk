/**
 * Synap's compaction in place of the caller's earlier turns.
 *
 * The AI SDK has no compaction: the caller sends the whole message list on
 * every call. Once Synap has compacted the conversation, those earlier turns
 * are what the summary already covers, and sending both duplicates them.
 */
import { describe, it, expect, vi } from 'vitest';
import type { LanguageModelV1Prompt } from '@ai-sdk/provider';
import {
  compactedHistoryFromResponse,
  fetchCompactedHistory,
  spliceHistory,
  isSynapHistory,
  type CompactedHistory,
} from '../history.js';

const user = (t: string) => ({ role: 'user' as const, content: [{ type: 'text' as const, text: t }] });
const asst = (t: string) => ({ role: 'assistant' as const, content: [{ type: 'text' as const, text: t }] });
const sys = (t: string) => ({ role: 'system' as const, content: t });

const history = (turns: Array<['user' | 'assistant', string]> = []): CompactedHistory => ({
  preamble: '<synap_conversation_history>\nSUMMARY\n</synap_conversation_history>',
  turns: turns.map(([role, text]) => ({ role, text })),
  compactionId: 'cmp_1',
});

function texts(prompt: LanguageModelV1Prompt): string[] {
  return prompt.map((m) => {
    if (m.role === 'system') return `system:${m.content}`;
    if (m.role === 'tool') return `tool:${m.content.map((p) => p.toolCallId).join(',')}`;
    return `${m.role}:${m.content.map((p) => (p.type === 'text' ? p.text : p.type)).join('')}`;
  });
}

describe('compactedHistoryFromResponse', () => {
  it('reads the flat route response', () => {
    const h = compactedHistoryFromResponse({
      context_block: '## Conversation so far (compacted)\nfacts',
      compaction_id: 'cmp_9',
      recent_messages: [
        { role: 'user', content: 'a' }, { role: 'tool', content: 'ignored' },
        { role: 'assistant', content: 'b' }, { role: 'user', content: '  ' },
      ],
    });
    expect(h?.preamble).toBe(
      '<synap_conversation_history>\n## Conversation so far (compacted)\nfacts\n</synap_conversation_history>');
    expect(h?.turns).toEqual([{ role: 'user', text: 'a' }, { role: 'assistant', text: 'b' }]);
    expect(h?.compactionId).toBe('cmp_9');
  });

  it('reads the context_for_prompt envelope the JS SDK unwraps', () => {
    const h = compactedHistoryFromResponse({ context_for_prompt: { context_block: 'B', recent_messages: [] } });
    expect(h?.preamble).toContain('B');
  });

  it('is null when there is no compaction block (nothing to replace)', () => {
    expect(compactedHistoryFromResponse({ available: true, formatted_context: 'x', recent_messages: [{ role: 'user', content: 'hi' }] })).toBeNull();
    expect(compactedHistoryFromResponse({ context_block: '   ' })).toBeNull();
    expect(compactedHistoryFromResponse(null)).toBeNull();
  });
});

describe('fetchCompactedHistory', () => {
  const credentials = { api_key: 'k', client_id: 'cli', instance_id: 'inst' };

  it('GETs the context-for-prompt route and parses it', async () => {
    const fetchImpl = vi.fn(async () => ({
      ok: true, status: 200, json: async () => ({ context_block: 'B', recent_messages: [] }),
    })) as unknown as typeof fetch;
    const h = await fetchCompactedHistory({ credentials, conversationId: 'conv/1', baseUrl: 'https://x.test', fetchImpl });
    expect(h?.preamble).toContain('B');
    const [url, init] = (fetchImpl as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://x.test/v1/conversations/conv%2F1/context-for-prompt?style=structured');
    expect(init.method).toBe('GET');
    expect((init.headers as Record<string, string>)['X-Instance-ID']).toBe('inst');
  });

  it('is null on 404, on HTTP errors and on network errors, and never throws', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    for (const impl of [
      async () => ({ ok: false, status: 404, json: async () => ({}) }),
      async () => ({ ok: false, status: 500, json: async () => ({}) }),
      async () => { throw new Error('down'); },
    ]) {
      expect(await fetchCompactedHistory({ credentials, conversationId: 'c', fetchImpl: impl as never })).toBeNull();
    }
  });

  it('does not call the server without a conversation id', async () => {
    const fetchImpl = vi.fn();
    expect(await fetchCompactedHistory({ credentials, conversationId: '', fetchImpl: fetchImpl as never })).toBeNull();
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});

describe('spliceHistory', () => {
  it('replaces every earlier turn with the preamble and Synap\'s turns', () => {
    const prompt: LanguageModelV1Prompt = [
      sys('Be brief.'),
      user('q1'), asst('a1'), user('q2'), asst('a2'), user('q3'), asst('a3'),
      user('now'),
    ];
    const out = spliceHistory(prompt, history([['user', 'q3'], ['assistant', 'a3']]));
    expect(texts(out)).toEqual([
      'system:Be brief.',
      'system:<synap_conversation_history>\nSUMMARY\n</synap_conversation_history>',
      'user:q3', 'assistant:a3',
      'user:now',
    ]);
  });

  it('keeps the current turn whole: tool call and result stay paired', () => {
    const toolCall = { role: 'assistant' as const, content: [{ type: 'tool-call' as const, toolCallId: 't1', toolName: 'f', args: {} }] };
    const toolResult = { role: 'tool' as const, content: [{ type: 'tool-result' as const, toolCallId: 't1', toolName: 'f', result: 1 }] };
    const oldCall = { role: 'assistant' as const, content: [{ type: 'tool-call' as const, toolCallId: 'old', toolName: 'f', args: {} }] };
    const oldResult = { role: 'tool' as const, content: [{ type: 'tool-result' as const, toolCallId: 'old', toolName: 'f', result: 0 }] };
    const prompt: LanguageModelV1Prompt = [
      user('q1'), oldCall, oldResult, asst('a1'),
      user('now'), toolCall, toolResult,
    ];
    const out = spliceHistory(prompt, history([['user', 'q1'], ['assistant', 'a1']]));
    expect(texts(out)).toEqual([
      'system:<synap_conversation_history>\nSUMMARY\n</synap_conversation_history>',
      'user:q1', 'assistant:a1',
      'user:now', 'assistant:tool-call', 'tool:t1',
    ]);
    // The earlier tool exchange went with the turns the compaction covers.
    expect(texts(out).join()).not.toContain('old');
  });

  it('does not repeat the current user message Synap has already recorded', () => {
    const prompt: LanguageModelV1Prompt = [user('q1'), asst('a1'), user('now')];
    const out = spliceHistory(prompt, history([['user', 'q1'], ['assistant', 'a1'], ['user', 'now']]));
    expect(texts(out).filter((t) => t === 'user:now')).toHaveLength(1);
  });

  it('keeps turns the caller has that Synap has not recorded yet', () => {
    const prompt: LanguageModelV1Prompt = [user('q1'), asst('a1'), user('q2'), asst('a2'), user('now')];
    const out = spliceHistory(prompt, history([['user', 'q1'], ['assistant', 'a1']]));
    expect(texts(out)).toEqual([
      'system:<synap_conversation_history>\nSUMMARY\n</synap_conversation_history>',
      'user:q1', 'assistant:a1', 'user:q2', 'assistant:a2', 'user:now',
    ]);
  });

  it('drops a Synap block left in the prompt by an earlier pass', () => {
    const prompt: LanguageModelV1Prompt = [sys('<synap_conversation_history>\nOLD\n</synap_conversation_history>'), user('q1'), user('now')];
    const out = spliceHistory(prompt, history());
    expect(texts(out).filter((t) => t.includes('OLD'))).toHaveLength(0);
    expect(isSynapHistory(history().preamble)).toBe(true);
    expect(isSynapHistory('## Conversation so far (compacted, 12 turns)')).toBe(true);
    expect(isSynapHistory('hello')).toBe(false);
  });
});
