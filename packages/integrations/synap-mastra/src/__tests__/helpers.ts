// Shared mock helpers for synap-mastra tests.
// Mirrors the mock_data.ts from smoke_tests_ts but adapted for vitest's vi.fn().

import { vi } from 'vitest';
import type {
  SynapSdkLike,
  SynapFetchResponseLike,
  SynapPromptContext,
  SynapRecentMessage,
  SynapMemoryCreateResult,
} from '../types.js';

// ── Date anchors ─────────────────────────────────────────────────────────────

export const NOW = new Date('2026-06-08T00:00:00.000Z');
export const YESTERDAY = new Date(NOW.getTime() - 24 * 60 * 60 * 1000);
export const NEXT_WEEK = new Date(NOW.getTime() + 7 * 24 * 60 * 60 * 1000);

// ── Scenario factories ────────────────────────────────────────────────────────

export function RICH(): SynapFetchResponseLike {
  return {
    formatted_context:
      '## User Context\n### Facts\n- Senior engineer at Acme\n- Timezone PT\n' +
      '### Preferences\n- Email > Slack\n- Dark mode\n',
    facts: [
      { id: 'f1', content: 'User is a senior engineer at Acme', confidence: 0.92 },
      { id: 'f2', content: "User's timezone is America/Los_Angeles", confidence: 0.99 },
    ],
    preferences: [
      { id: 'p1', content: 'Prefers email updates over Slack', strength: 0.9 },
    ],
    episodes: [],
    emotions: [],
    temporal_events: [],
  };
}

export function EMPTY(): SynapFetchResponseLike {
  return {
    formatted_context: '',
    facts: [],
    preferences: [],
    episodes: [],
    emotions: [],
    temporal_events: [],
  };
}

export const DIALOGUE: SynapRecentMessage[] = [
  {
    role: 'user',
    content: 'Can you remind me when my trial expires?',
    timestamp: new Date(YESTERDAY.getTime() + 60_000).toISOString(),
    message_id: 'msg-001',
  },
  {
    role: 'assistant',
    content: 'Your trial expires next Friday.',
    timestamp: new Date(YESTERDAY.getTime() + 120_000).toISOString(),
    message_id: 'msg-002',
  },
  {
    role: 'user',
    content: 'Thanks. Can you log that I prefer email updates?',
    timestamp: new Date(YESTERDAY.getTime() + 180_000).toISOString(),
    message_id: 'msg-003',
  },
  {
    role: 'assistant',
    content: 'Noted — you prefer email for updates.',
    timestamp: new Date(YESTERDAY.getTime() + 240_000).toISOString(),
    message_id: 'msg-004',
  },
];

export function promptContext(
  withMessages = true,
  available = true,
): SynapPromptContext {
  return {
    formatted_context: available
      ? 'User is a senior engineer. Prefers email updates.'
      : '',
    available,
    recent_messages: withMessages ? [...DIALOGUE] : [],
    recent_message_count: withMessages ? DIALOGUE.length : 0,
    total_message_count: withMessages ? DIALOGUE.length : 0,
  };
}

// ── SDK mock factory ──────────────────────────────────────────────────────────

export interface MakeSdkOptions {
  fetchResponse?: SynapFetchResponseLike;
  promptCtx?: SynapPromptContext;
  ingestionId?: string;
}

export function makeSdk(opts: MakeSdkOptions = {}): SynapSdkLike {
  const fetchResponse = opts.fetchResponse ?? RICH();
  const ctx = opts.promptCtx ?? promptContext();
  const ingestionId = opts.ingestionId ?? 'ing-test-001';

  return {
    fetch: vi.fn().mockResolvedValue(fetchResponse),
    conversation: {
      record_message: vi.fn().mockResolvedValue({
        message_id: 'm-001',
        conversation_id: 'conv-test',
        session_id: 'sess-test',
        recorded_at: NOW.toISOString(),
      }),
      context: {
        get_context_for_prompt: vi.fn().mockResolvedValue(ctx),
      },
    },
    memories: {
      create: vi.fn().mockResolvedValue({ ingestion_id: ingestionId } satisfies SynapMemoryCreateResult),
    },
  };
}

export function makeFailingSdk(err?: Error): SynapSdkLike {
  const e = err ?? new Error('simulated-sdk-failure');
  const sdk = makeSdk();
  (sdk.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(e);
  (sdk.conversation.record_message as ReturnType<typeof vi.fn>).mockRejectedValue(e);
  (sdk.conversation.context.get_context_for_prompt as ReturnType<typeof vi.fn>).mockRejectedValue(e);
  (sdk.memories.create as ReturnType<typeof vi.fn>).mockRejectedValue(e);
  return sdk;
}

// Partial-failure SDK: only specific endpoints fail
export interface PartialSdkOpts {
  fetchOk?: boolean;
  createOk?: boolean;
  recordOk?: boolean;
  ctxOk?: boolean;
}

export function makePartialSdk(opts: PartialSdkOpts = {}): SynapSdkLike {
  const { fetchOk = true, createOk = true, recordOk = true, ctxOk = true } = opts;
  const sdk = makeSdk();
  const err = new Error('endpoint-down');
  if (!fetchOk) (sdk.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(err);
  if (!createOk) (sdk.memories.create as ReturnType<typeof vi.fn>).mockRejectedValue(err);
  if (!recordOk)
    (sdk.conversation.record_message as ReturnType<typeof vi.fn>).mockRejectedValue(err);
  if (!ctxOk)
    (sdk.conversation.context.get_context_for_prompt as ReturnType<typeof vi.fn>).mockRejectedValue(err);
  return sdk;
}

// ── Mastra-format message builder ────────────────────────────────────────────

export function msg(role: string, text: string, threadId: string) {
  return {
    id: `m-${Math.random().toString(36).slice(2, 8)}`,
    role,
    createdAt: new Date(),
    threadId,
    resourceId: 'alice',
    type: 'text',
    content: {
      format: 2 as const,
      parts: [{ type: 'text' as const, text }],
    },
  };
}
