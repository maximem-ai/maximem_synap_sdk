/**
 * Tests for src/transform/messages.ts
 *
 * Covers:
 *   - rawItemToContextItem: field mapping, optional fields, fallbacks
 *   - buildContextSystemBlock: all section types, empty ctx, unicode, ordering
 *   - injectContextIntoPrompt: no-system, with-system, no-op on empty
 *   - extractSearchQuery: last user only, multi-part, 512 truncation, no user
 *   - promptToTranscript: filters system + non-text, joins multi-part, drops empty-content
 */

import { describe, it, expect } from 'vitest';
import {
  rawItemToContextItem,
  buildContextSystemBlock,
  injectContextIntoPrompt,
  extractSearchQuery,
  promptToTranscript,
} from '../transform/messages.js';
import type { RawContextItem, FetchedContext } from '../types.js';
import type { LanguageModelV1Prompt } from '@ai-sdk/provider';

// ─── Helpers ────────────────────────────────────────────────────────────────

function makeRaw(overrides: Partial<RawContextItem> = {}): RawContextItem {
  return {
    item_id: 'id-1',
    content: 'some content',
    context_type: 'fact',
    source: 'profile',
    confidence: 0.9,
    similarity_score: 0.85,
    relevance_score: 0.8,
    scope: 'user',
    entity_id: '',
    event_date: '',
    valid_until: '',
    temporal_category: '',
    temporal_confidence: 0,
    ...overrides,
  };
}

function emptyCtx(): FetchedContext {
  return {
    facts: [],
    preferences: [],
    episodes: [],
    emotions: [],
    temporalEvents: [],
    conversationContext: null,
    source: 'cloud',
    correlationId: 'corr-x',
  };
}

function richCtx(): FetchedContext {
  return {
    facts: [
      rawItemToContextItem(makeRaw({ item_id: 'f1', content: 'User is a senior engineer' })),
    ],
    preferences: [
      rawItemToContextItem(makeRaw({ item_id: 'p1', content: 'Prefers dark mode', context_type: 'preference' })),
    ],
    episodes: [
      rawItemToContextItem(makeRaw({ item_id: 'e1', content: 'Worked on project X', context_type: 'episode' })),
    ],
    emotions: [
      rawItemToContextItem(makeRaw({ item_id: 'em1', content: 'Currently excited', context_type: 'emotion' })),
    ],
    temporalEvents: [
      rawItemToContextItem(makeRaw({ item_id: 't1', content: 'Meeting on Friday', context_type: 'temporal_events' })),
    ],
    conversationContext: {
      summary: 'Discussing deployment options',
      currentState: { step: 1 },
      keyExtractions: { topic: 'deploy' },
      recentTurns: [
        { role: 'user', content: 'How do we deploy?', timestamp: '2026-01-01T00:00:00Z' },
        { role: 'assistant', content: 'Use CI/CD', timestamp: '2026-01-01T00:00:01Z' },
      ],
      compactionId: 'comp-1',
      conversationId: 'conv-1',
    },
    source: 'cloud',
    correlationId: 'corr-1',
  };
}

// ─── rawItemToContextItem ──────────────────────────────────────────────────

describe('rawItemToContextItem', () => {
  it('maps all required fields correctly', () => {
    const raw = makeRaw({
      item_id: 'item-abc',
      content: 'Test fact content',
      context_type: 'fact',
      source: 'profile',
      confidence: 0.95,
    });
    const item = rawItemToContextItem(raw);
    expect(item.id).toBe('item-abc');
    expect(item.content).toBe('Test fact content');
    expect(item.contextType).toBe('fact');
    expect(item.source).toBe('profile');
    expect(item.confidence).toBe(0.95);
  });

  it('returns undefined for empty optional string fields (eventDate)', () => {
    const item = rawItemToContextItem(makeRaw({ event_date: '' }));
    expect(item.eventDate).toBeUndefined();
  });

  it('returns the value when event_date is non-empty', () => {
    const item = rawItemToContextItem(makeRaw({ event_date: '2026-06-01' }));
    expect(item.eventDate).toBe('2026-06-01');
  });

  it('returns undefined for empty valid_until', () => {
    const item = rawItemToContextItem(makeRaw({ valid_until: '' }));
    expect(item.validUntil).toBeUndefined();
  });

  it('returns the value when valid_until is non-empty', () => {
    const item = rawItemToContextItem(makeRaw({ valid_until: '2026-12-31' }));
    expect(item.validUntil).toBe('2026-12-31');
  });

  it('returns undefined for empty temporal_category', () => {
    const item = rawItemToContextItem(makeRaw({ temporal_category: '' }));
    expect(item.temporalCategory).toBeUndefined();
  });

  it('returns the value when temporal_category is non-empty', () => {
    const item = rawItemToContextItem(makeRaw({ temporal_category: 'upcoming' }));
    expect(item.temporalCategory).toBe('upcoming');
  });

  it('uses empty string fallback when item_id is missing', () => {
    const raw = makeRaw({ item_id: undefined as unknown as string });
    const item = rawItemToContextItem(raw);
    expect(item.id).toBe('');
  });

  it('uses 0 fallback when confidence is missing', () => {
    const raw = makeRaw({ confidence: undefined as unknown as number });
    const item = rawItemToContextItem(raw);
    expect(item.confidence).toBe(0);
  });

  it('preserves unicode content', () => {
    const item = rawItemToContextItem(makeRaw({ content: '🚀 ships to 東京' }));
    expect(item.content).toBe('🚀 ships to 東京');
  });
});

// ─── buildContextSystemBlock ──────────────────────────────────────────────

describe('buildContextSystemBlock', () => {
  it('returns empty string for empty context', () => {
    expect(buildContextSystemBlock(emptyCtx())).toBe('');
  });

  it('wraps output in <synap_context> tags', () => {
    const block = buildContextSystemBlock(richCtx());
    expect(block).toMatch(/^<synap_context>/);
    expect(block).toMatch(/<\/synap_context>$/);
  });

  it('includes ## Known Facts section for facts', () => {
    const ctx = { ...emptyCtx(), facts: [rawItemToContextItem(makeRaw({ content: 'Fact A' }))] };
    const block = buildContextSystemBlock(ctx);
    expect(block).toContain('## Known Facts');
    expect(block).toContain('- Fact A');
  });

  it('includes ## User Preferences section', () => {
    const ctx = { ...emptyCtx(), preferences: [rawItemToContextItem(makeRaw({ content: 'Pref B' }))] };
    const block = buildContextSystemBlock(ctx);
    expect(block).toContain('## User Preferences');
    expect(block).toContain('- Pref B');
  });

  it('includes ## Past Episodes section', () => {
    const ctx = { ...emptyCtx(), episodes: [rawItemToContextItem(makeRaw({ content: 'Ep C' }))] };
    const block = buildContextSystemBlock(ctx);
    expect(block).toContain('## Past Episodes');
    expect(block).toContain('- Ep C');
  });

  it('includes ## Emotional Context section', () => {
    const ctx = { ...emptyCtx(), emotions: [rawItemToContextItem(makeRaw({ content: 'Happy' }))] };
    const block = buildContextSystemBlock(ctx);
    expect(block).toContain('## Emotional Context');
    expect(block).toContain('- Happy');
  });

  it('includes ## Upcoming / Temporal section', () => {
    const ctx = { ...emptyCtx(), temporalEvents: [rawItemToContextItem(makeRaw({ content: 'Meeting' }))] };
    const block = buildContextSystemBlock(ctx);
    expect(block).toContain('## Upcoming / Temporal');
    expect(block).toContain('- Meeting');
  });

  it('includes ## Conversation Summary when present', () => {
    const ctx = {
      ...emptyCtx(),
      conversationContext: {
        summary: 'We discussed deployment',
        currentState: {},
        keyExtractions: {},
        recentTurns: [],
        compactionId: null,
        conversationId: null,
      },
    };
    const block = buildContextSystemBlock(ctx);
    expect(block).toContain('## Conversation Summary');
    expect(block).toContain('We discussed deployment');
  });

  it('includes ## Recent Conversation when recentTurns present', () => {
    const ctx = {
      ...emptyCtx(),
      conversationContext: {
        summary: null,
        currentState: {},
        keyExtractions: {},
        recentTurns: [
          { role: 'user', content: 'What about sharding?', timestamp: '2026-01-01T00:00:00Z' },
          { role: 'assistant', content: 'Sharding works if...', timestamp: '2026-01-01T00:00:01Z' },
        ],
        compactionId: null,
        conversationId: null,
      },
    };
    const block = buildContextSystemBlock(ctx);
    expect(block).toContain('## Recent Conversation');
    expect(block).toContain('user: What about sharding?');
    expect(block).toContain('assistant: Sharding works if...');
  });

  it('limits recentTurns to last 6', () => {
    const turns = Array.from({ length: 10 }, (_, i) => ({
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: `turn-${i}`,
      timestamp: `2026-01-01T00:00:0${i}Z`,
    }));
    const ctx = {
      ...emptyCtx(),
      conversationContext: {
        summary: null,
        currentState: {},
        keyExtractions: {},
        recentTurns: turns,
        compactionId: null,
        conversationId: null,
      },
    };
    const block = buildContextSystemBlock(ctx);
    // Should NOT contain turn-0 (only last 6 = turns 4-9)
    expect(block).not.toContain('turn-0');
    expect(block).toContain('turn-9');
    expect(block).toContain('turn-4');
  });

  it('does not include empty sections', () => {
    const ctx = { ...emptyCtx(), facts: [rawItemToContextItem(makeRaw({ content: 'Only fact' }))] };
    const block = buildContextSystemBlock(ctx);
    expect(block).not.toContain('## User Preferences');
    expect(block).not.toContain('## Past Episodes');
    expect(block).not.toContain('## Emotional Context');
  });

  it('includes all sections for rich context', () => {
    const block = buildContextSystemBlock(richCtx());
    expect(block).toContain('## Conversation Summary');
    expect(block).toContain('## Recent Conversation');
    expect(block).toContain('## Known Facts');
    expect(block).toContain('## User Preferences');
    expect(block).toContain('## Past Episodes');
    expect(block).toContain('## Emotional Context');
    expect(block).toContain('## Upcoming / Temporal');
  });

  it('preserves unicode in context items', () => {
    const ctx = {
      ...emptyCtx(),
      facts: [rawItemToContextItem(makeRaw({ content: '🚀 ships to 東京 in "Q2"' }))],
    };
    const block = buildContextSystemBlock(ctx);
    expect(block).toContain('🚀');
    expect(block).toContain('東京');
  });

  it('omits recentTurns section when array is empty', () => {
    const ctx = {
      ...emptyCtx(),
      conversationContext: {
        summary: 'Summary here',
        currentState: {},
        keyExtractions: {},
        recentTurns: [],
        compactionId: null,
        conversationId: null,
      },
    };
    const block = buildContextSystemBlock(ctx);
    expect(block).toContain('## Conversation Summary');
    expect(block).not.toContain('## Recent Conversation');
  });
});

// ─── injectContextIntoPrompt ──────────────────────────────────────────────

describe('injectContextIntoPrompt', () => {
  const userOnly: LanguageModelV1Prompt = [
    { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
  ];
  const withSystem: LanguageModelV1Prompt = [
    { role: 'system', content: 'You are an assistant.' },
    { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
  ];

  it('returns prompt unchanged when context is empty', () => {
    const out = injectContextIntoPrompt(userOnly, emptyCtx());
    expect(out).toBe(userOnly);
  });

  it('inserts a system message at the head when none exists', () => {
    const out = injectContextIntoPrompt(userOnly, richCtx());
    expect(out).toHaveLength(2);
    expect(out[0].role).toBe('system');
    expect((out[0] as { role: 'system'; content: string }).content).toContain('<synap_context>');
    expect(out[1]).toBe(userOnly[0]);
  });

  it('prepends context block to existing system message content', () => {
    const out = injectContextIntoPrompt(withSystem, richCtx());
    expect(out).toHaveLength(2); // no extra messages added
    expect(out[0].role).toBe('system');
    const content = (out[0] as { role: 'system'; content: string }).content;
    expect(content).toContain('<synap_context>');
    expect(content).toContain('You are an assistant.');
    // Context block comes BEFORE the original system content
    expect(content.indexOf('<synap_context>')).toBeLessThan(content.indexOf('You are an assistant.'));
  });

  it('does not mutate the original prompt array', () => {
    const original = [...withSystem];
    injectContextIntoPrompt(withSystem, richCtx());
    expect(withSystem).toEqual(original);
  });

  it('preserves non-system messages in their original order', () => {
    const multiTurn: LanguageModelV1Prompt = [
      { role: 'system', content: 'Sys' },
      { role: 'user', content: [{ type: 'text', text: 'First' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Reply' }] },
      { role: 'user', content: [{ type: 'text', text: 'Second' }] },
    ];
    const out = injectContextIntoPrompt(multiTurn, richCtx());
    expect(out).toHaveLength(4);
    expect(out[1].role).toBe('user');
    expect(out[2].role).toBe('assistant');
    expect(out[3].role).toBe('user');
  });
});

// ─── extractSearchQuery ────────────────────────────────────────────────────

describe('extractSearchQuery', () => {
  it('returns empty array for empty prompt', () => {
    expect(extractSearchQuery([])).toEqual([]);
  });

  it('returns empty array when no user messages', () => {
    const prompt: LanguageModelV1Prompt = [
      { role: 'system', content: 'System message' },
    ];
    expect(extractSearchQuery(prompt)).toEqual([]);
  });

  it('extracts text from the last user message', () => {
    const prompt: LanguageModelV1Prompt = [
      { role: 'user', content: [{ type: 'text', text: 'first question' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'reply' }] },
      { role: 'user', content: [{ type: 'text', text: 'second question' }] },
    ];
    expect(extractSearchQuery(prompt)).toEqual(['second question']);
  });

  it('concatenates multiple text parts in last user message', () => {
    const prompt: LanguageModelV1Prompt = [
      { role: 'user', content: [{ type: 'text', text: 'part A' }, { type: 'text', text: 'part B' }] },
    ];
    // The product joins parts with ' ' → 'part A part B'
    expect(extractSearchQuery(prompt)).toEqual(['part A part B']);
  });

  it('ignores non-text content parts (e.g. image)', () => {
    const prompt: LanguageModelV1Prompt = [
      {
        role: 'user',
        content: [
          { type: 'image', image: new Uint8Array() as unknown as URL, mimeType: 'image/png' },
          { type: 'text', text: 'describe this' },
        ],
      },
    ];
    expect(extractSearchQuery(prompt)).toEqual(['describe this']);
  });

  it('truncates query at 512 characters', () => {
    const long = 'a'.repeat(1000);
    const prompt: LanguageModelV1Prompt = [
      { role: 'user', content: [{ type: 'text', text: long }] },
    ];
    const result = extractSearchQuery(prompt);
    expect(result).toHaveLength(1);
    expect(result[0]).toHaveLength(512);
  });

  it('returns empty array when user content has only non-text parts', () => {
    const prompt: LanguageModelV1Prompt = [
      {
        role: 'user',
        content: [
          { type: 'image', image: new Uint8Array() as unknown as URL, mimeType: 'image/png' },
        ],
      },
    ];
    expect(extractSearchQuery(prompt)).toEqual([]);
  });

  it('returns a single-element array with the text', () => {
    const prompt: LanguageModelV1Prompt = [
      { role: 'user', content: [{ type: 'text', text: 'What is sharding?' }] },
    ];
    const result = extractSearchQuery(prompt);
    expect(result).toHaveLength(1);
    expect(result[0]).toBe('What is sharding?');
  });
});

// ─── promptToTranscript ────────────────────────────────────────────────────

describe('promptToTranscript', () => {
  it('returns empty array for empty prompt', () => {
    expect(promptToTranscript([])).toEqual([]);
  });

  it('filters out system messages', () => {
    const prompt: LanguageModelV1Prompt = [
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: [{ type: 'text', text: 'hi' }] },
    ];
    const result = promptToTranscript(prompt);
    expect(result).toHaveLength(1);
    expect(result[0].role).toBe('user');
  });

  it('includes user and assistant messages', () => {
    const prompt: LanguageModelV1Prompt = [
      { role: 'user', content: [{ type: 'text', text: 'hello' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'hi there' }] },
    ];
    const result = promptToTranscript(prompt);
    expect(result).toEqual([
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'hi there' },
    ]);
  });

  it('joins multiple text parts in a single message', () => {
    const prompt: LanguageModelV1Prompt = [
      { role: 'user', content: [{ type: 'text', text: 'part 1' }, { type: 'text', text: ' part 2' }] },
    ];
    const result = promptToTranscript(prompt);
    expect(result[0].content).toBe('part 1 part 2');
  });

  it('filters out non-text content parts', () => {
    const prompt: LanguageModelV1Prompt = [
      {
        role: 'user',
        content: [
          { type: 'image', image: new Uint8Array() as unknown as URL, mimeType: 'image/png' },
          { type: 'text', text: 'describe this image' },
        ],
      },
    ];
    const result = promptToTranscript(prompt);
    expect(result[0].content).toBe('describe this image');
  });

  it('drops messages with no text content (only non-text parts)', () => {
    const prompt: LanguageModelV1Prompt = [
      {
        role: 'user',
        content: [
          { type: 'image', image: new Uint8Array() as unknown as URL, mimeType: 'image/png' },
        ],
      },
    ];
    const result = promptToTranscript(prompt);
    expect(result).toHaveLength(0);
  });

  it('preserves original message order', () => {
    const prompt: LanguageModelV1Prompt = [
      { role: 'user', content: [{ type: 'text', text: 'q1' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'a1' }] },
      { role: 'user', content: [{ type: 'text', text: 'q2' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'a2' }] },
    ];
    const result = promptToTranscript(prompt);
    expect(result.map(m => m.content)).toEqual(['q1', 'a1', 'q2', 'a2']);
  });
});
