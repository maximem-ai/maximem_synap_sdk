/**
 * Put Synap's compaction where the caller's own history would go.
 *
 * The AI SDK has no compaction of its own: the caller sends the whole message
 * list on every call. Synap compacts the conversation server-side, so once it
 * has, the earlier turns in that list are the same turns Synap's summary
 * already covers. Sending both duplicates them and keeps paying for the long
 * list.
 *
 * - `fetchCompactedHistory` asks the server for the conversation as it holds
 *   it: the compaction block and the turns after the compaction.
 * - `spliceHistory` takes the prompt and replaces every turn before the
 *   current one with that.
 *
 * Mirrors `synap_integrations_common.history` (Python), minus the SDK's local
 * history-search excerpts, which need the SDK's local store.
 */

import type { LanguageModelV1Message, LanguageModelV1Prompt } from '@ai-sdk/provider';
import type { Credentials } from './types.js';
import { newCorrelationId } from './util/correlation.js';
import { SDK_VERSION } from './version.js';

const DEFAULT_BASE_URL = 'https://synap-cloud-prod.maximem.ai';
const FETCH_TIMEOUT_MS = 10_000;

/** What the server's compaction block starts with. */
export const COMPACTION_PREFIX = '## Conversation so far (compacted';
export const HISTORY_OPEN = '<synap_conversation_history>';
export const HISTORY_CLOSE = '</synap_conversation_history>';

export interface CompactedHistory {
  /** The compaction block, wrapped in `<synap_conversation_history>` tags. */
  preamble: string;
  /** Turns after the compaction, oldest first. */
  turns: Array<{ role: 'user' | 'assistant'; text: string }>;
  compactionId: string | null;
}

/**
 * Whether a message is the history this module put there, not a turn.
 * Anything that writes messages back to Synap skips these, or the compaction
 * would be recorded as something somebody said.
 */
export function isSynapHistory(text: unknown): boolean {
  if (typeof text !== 'string') return false;
  const t = text.trimStart();
  return t.startsWith(HISTORY_OPEN) || t.startsWith(COMPACTION_PREFIX);
}

/**
 * Read a `context-for-prompt` response into a CompactedHistory. Null when the
 * conversation has no compaction block yet: the caller's list is still short
 * and is the better record.
 */
export function compactedHistoryFromResponse(data: unknown): CompactedHistory | null {
  if (!data || typeof data !== 'object') return null;
  const outer = data as Record<string, unknown>;
  // The route answers flat; the JS SDK also unwraps a `context_for_prompt` envelope.
  const r = (outer['context_for_prompt'] && typeof outer['context_for_prompt'] === 'object'
    ? outer['context_for_prompt'] : outer) as Record<string, unknown>;

  const block = r['context_block'];
  if (typeof block !== 'string' || !block.trim()) return null;

  const turns: CompactedHistory['turns'] = [];
  const recent = Array.isArray(r['recent_messages']) ? r['recent_messages'] : [];
  for (const m of recent as Array<Record<string, unknown>>) {
    const role = m?.['role'];
    const content = m?.['content'];
    if ((role === 'user' || role === 'assistant') && typeof content === 'string' && content.trim()) {
      turns.push({ role, text: content });
    }
  }
  const id = r['compaction_id'];
  return {
    preamble: `${HISTORY_OPEN}\n${block.trim()}\n${HISTORY_CLOSE}`,
    turns,
    compactionId: typeof id === 'string' && id ? id : null,
  };
}

export interface FetchCompactedHistoryParams {
  credentials: Credentials;
  conversationId: string;
  baseUrl?: string;
  fetchImpl?: typeof fetch;
}

/**
 * The compacted conversation, or null when there is nothing to replace: no
 * compaction yet, and on any error (the caller's own history still stands).
 * Never throws.
 */
export async function fetchCompactedHistory(
  params: FetchCompactedHistoryParams,
): Promise<CompactedHistory | null> {
  const { credentials, conversationId } = params;
  if (!conversationId) return null;
  const baseUrl = params.baseUrl ?? DEFAULT_BASE_URL;
  const doFetch = params.fetchImpl ?? fetch;
  try {
    const res = await doFetch(
      `${baseUrl}/v1/conversations/${encodeURIComponent(conversationId)}/context-for-prompt?style=structured`,
      {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${credentials.api_key}`,
          'X-Client-ID': credentials.client_id,
          'X-Instance-ID': credentials.instance_id,
          'X-Correlation-ID': newCorrelationId(),
          'Accept': 'application/json',
          'User-Agent': `@maximem/synap-vercel-adk/${SDK_VERSION}`,
        },
        signal: typeof AbortSignal !== 'undefined' && 'timeout' in AbortSignal
          ? AbortSignal.timeout(FETCH_TIMEOUT_MS) : undefined,
      },
    );
    if (!res.ok) {
      // 404 is "nothing compacted yet" for a conversation the server has not seen.
      if (res.status !== 404) {
        console.warn(`[synap] compacted history unavailable (HTTP ${res.status})`);
      }
      return null;
    }
    return compactedHistoryFromResponse(await res.json());
  } catch (err) {
    console.warn('[synap] compacted history unavailable:', err);
    return null;
  }
}

// ─── Splice ───────────────────────────────────────────────────────────────────

/** The text of a prompt message: its string content, or its text parts joined. */
export function messageText(m: LanguageModelV1Message): string {
  if (m.role === 'system') return m.content;
  if (m.role === 'tool') return '';
  return m.content
    .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
    .map((p) => p.text)
    .join('');
}

function norm(text: string): string {
  return text.split(/\s+/).filter(Boolean).join(' ');
}

function makeMessage(role: 'system' | 'user' | 'assistant', text: string): LanguageModelV1Message {
  if (role === 'system') return { role: 'system', content: text };
  return { role, content: [{ type: 'text', text }] } as LanguageModelV1Message;
}

/**
 * Replace every turn before the current one with Synap's history.
 *
 * The result, in order:
 *  1. the caller's own system messages from before the current turn;
 *  2. the preamble, as one system message;
 *  3. Synap's turns after the compaction;
 *  4. earlier turns the caller has and Synap does not yet (a write still in
 *     flight), rebuilt as plain text;
 *  5. the current turn untouched: the last user message and everything after
 *     it, so a tool call stays paired with its result.
 *
 * Tool messages from earlier turns go with the turns they belonged to: the
 * compaction covers them.
 */
export function spliceHistory(
  prompt: LanguageModelV1Prompt,
  history: CompactedHistory,
): LanguageModelV1Prompt {
  let currentStart = prompt.length;
  for (let i = prompt.length - 1; i >= 0; i--) {
    if (prompt[i]!.role === 'user') { currentStart = i; break; }
  }
  const earlier = prompt.slice(0, currentStart);
  const current = prompt.slice(currentStart);

  const systems = earlier.filter((m) => m.role === 'system' && !isSynapHistory(messageText(m)));
  const frameworkTurns = earlier
    .filter((m) => (m.role === 'user' || m.role === 'assistant') && norm(messageText(m)))
    .map((m) => ({ role: m.role as 'user' | 'assistant', text: messageText(m) }));

  const turns = [...history.turns];
  // The current user message is usually recorded before the model is called,
  // so Synap already has it, and maybe a reply from an earlier step of this
  // turn. Both belong to the current turn, which the caller still holds.
  if (current.length && current[0]!.role === 'user') {
    const now = norm(messageText(current[0]!));
    for (let i = turns.length - 1; i >= 0; i--) {
      if (turns[i]!.role === 'user' && norm(turns[i]!.text) === now) {
        turns.splice(i);
        break;
      }
    }
  }

  // Turns the caller has after the last one Synap has.
  let pending: typeof frameworkTurns = [];
  if (turns.length) {
    const last = turns[turns.length - 1]!;
    const lastText = norm(last.text);
    for (let i = frameworkTurns.length - 1; i >= 0; i--) {
      if (frameworkTurns[i]!.role === last.role && norm(frameworkTurns[i]!.text) === lastText) {
        pending = frameworkTurns.slice(i + 1);
        break;
      }
    }
  }

  return [
    ...systems,
    makeMessage('system', history.preamble),
    ...turns.map((t) => makeMessage(t.role, t.text)),
    ...pending.map((t) => makeMessage(t.role, t.text)),
    ...current,
  ];
}
