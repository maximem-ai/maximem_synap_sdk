/**
 * Render `get_context_for_prompt` locally from the short-term store.
 *
 * A direct port of Python's `formatter/context_for_prompt.py`. It exists for
 * the same reason: when the SDK is authoritative for short-term context, a
 * warm conversation can answer this call without a cloud round trip at all,
 * which is a latency saving and a metered-call saving both.
 *
 * The three styles and their exact heading text are copied from Python
 * deliberately. A prompt assembled by one SDK and one assembled by the other
 * must be the same string, or an application that switches languages silently
 * changes what its model sees.
 */

import type { ShortTermEntry, ShortTermTurn } from '../cache/short-term-store.js';
import type { Json } from './types.js';

export const SUPPORTED_STYLES = ['structured', 'narrative', 'bullet_points'] as const;
export type PromptStyle = (typeof SUPPORTED_STYLES)[number];
export const DEFAULT_STYLE: PromptStyle = 'structured';

export interface LocalRecentMessage {
  role: string;
  content: string;
  /** ISO-8601. */
  timestamp: string;
  message_id: string;
}

export interface LocalPromptResponse {
  formatted_context: string | null;
  available: boolean;
  is_stale: boolean;
  compression_ratio: number | null;
  validation_score: number | null;
  compaction_age_seconds: number | null;
  quality_warning: boolean;
  recent_messages: LocalRecentMessage[];
  recent_message_count: number;
  compacted_message_count: number;
  total_message_count: number;
  [key: string]: unknown;
}

function stringify(value: unknown): string {
  if (value === null || value === undefined) return '';
  const t = typeof value;
  if (t === 'string' || t === 'number' || t === 'boolean') return String(value);
  try {
    return JSON.stringify(value) ?? '';
  } catch {
    return String(value);
  }
}

/** Key extractions arrive as a list, a bare string, or a single object. */
function asListOfObjects(value: unknown): Record<string, unknown>[] {
  if (!value) return [];
  if (Array.isArray(value)) {
    const out: Record<string, unknown>[] = [];
    for (const v of value) {
      if (v && typeof v === 'object' && !Array.isArray(v)) out.push(v as Record<string, unknown>);
      else if (typeof v === 'string') out.push({ content: v });
    }
    return out;
  }
  if (typeof value === 'object') return [value as Record<string, unknown>];
  return [];
}

function itemText(item: Record<string, unknown>): string {
  return (item['content'] as string) || (item['text'] as string) || stringify(item);
}

function buildRecentMessages(turns: readonly ShortTermTurn[]): LocalRecentMessage[] {
  return (turns ?? []).map((t, idx) => ({
    role: t.role || 'user',
    content: t.content ?? '',
    timestamp: t.timestamp,
    message_id: `local-${idx}`,
  }));
}

/**
 * The SDK does not know how many messages a compaction covered: the server
 * does not put that count in the bundle. Python reports 0 here for the same
 * reason, and both must agree.
 */
function compactedMessageCount(_entry: ShortTermEntry): number {
  return 0;
}

function formatStructured(entry: ShortTermEntry, recent: LocalRecentMessage[]): string {
  const parts: string[] = [];
  if (entry.summary) { parts.push('## Summary'); parts.push(entry.summary.trim()); }
  if (entry.factualParagraph) { parts.push('## Facts'); parts.push(entry.factualParagraph.trim()); }
  if (entry.conversationalParagraph) {
    parts.push('## Conversation'); parts.push(entry.conversationalParagraph.trim());
  }

  const cs = entry.currentState ?? {};
  if (Object.keys(cs).length > 0) {
    parts.push('## Current State');
    for (const [k, v] of Object.entries(cs)) parts.push(`- ${k}: ${stringify(v)}`);
  }

  const ke = entry.keyExtractions ?? {};
  if (Object.keys(ke).length > 0) {
    parts.push('## Key Extractions');
    for (const [cat, items] of Object.entries(ke)) {
      if (!items) continue;
      parts.push(`### ${cat}`);
      for (const it of asListOfObjects(items)) parts.push(`- ${itemText(it)}`);
    }
  }

  if (recent.length > 0) {
    parts.push(`## Recent Turns (${recent.length})`);
    for (const m of recent) parts.push(`**${m.role}**: ${m.content}`);
  }

  return parts.filter(Boolean).join('\n\n').trim();
}

function formatNarrative(entry: ShortTermEntry, recent: LocalRecentMessage[]): string {
  const parts: string[] = [];
  if (entry.summary) parts.push(entry.summary.trim());
  if (entry.conversationalParagraph) parts.push(entry.conversationalParagraph.trim());
  else if (entry.factualParagraph) parts.push(entry.factualParagraph.trim());

  if (recent.length > 0) {
    const tail = recent.map((m) => `${m.role}: ${m.content}`).join('\n');
    parts.push(`Recent exchanges:\n${tail}`);
  }
  return parts.filter(Boolean).join('\n\n').trim();
}

function formatBullets(entry: ShortTermEntry, recent: LocalRecentMessage[]): string {
  const parts: string[] = [];
  if (entry.summary) parts.push(`- Summary: ${entry.summary.trim()}`);

  for (const [k, v] of Object.entries(entry.currentState ?? {})) {
    parts.push(`- ${k}: ${stringify(v)}`);
  }
  for (const [cat, items] of Object.entries(entry.keyExtractions ?? {})) {
    for (const it of asListOfObjects(items)) parts.push(`- (${cat}) ${itemText(it)}`);
  }
  for (const m of recent) parts.push(`- ${m.role}: ${m.content}`);

  return parts.join('\n').trim();
}

function formatBlock(entry: ShortTermEntry, recent: LocalRecentMessage[], style: string): string {
  if (style === 'narrative') return formatNarrative(entry, recent);
  if (style === 'bullet_points') return formatBullets(entry, recent);
  return formatStructured(entry, recent);
}

/** Build a context-for-prompt response from a cached entry. */
export function renderForPrompt(
  entry: ShortTermEntry,
  style: string = DEFAULT_STYLE,
  now: () => number = Date.now,
): LocalPromptResponse {
  const chosen: string = (SUPPORTED_STYLES as readonly string[]).includes(style)
    ? style
    : DEFAULT_STYLE;

  const recent = buildRecentMessages(entry.recentTurns);
  const formatted = formatBlock(entry, recent, chosen);
  const available = entry.compactionId !== null || entry.recentTurns.length > 0;

  let compactionAge: number | null = null;
  if (entry.compactedAt !== null) {
    const parsed = Date.parse(entry.compactedAt);
    if (!Number.isNaN(parsed)) compactionAge = Math.floor((now() - parsed) / 1000);
  }

  const compacted = compactedMessageCount(entry);
  return {
    formatted_context: available ? formatted : null,
    available,
    is_stale: false,
    compression_ratio: null,
    validation_score: null,
    compaction_age_seconds: compactionAge,
    quality_warning: false,
    recent_messages: recent,
    recent_message_count: recent.length,
    compacted_message_count: compacted,
    total_message_count: recent.length + compacted,
  };
}

/** Build a compaction response from a cached entry, or null if none landed. */
export function renderCompacted(
  entry: ShortTermEntry,
  format = 'structured',
  correlationId = '',
): Json | null {
  if (entry.compactionId === null) return null;

  const formatted = formatBlock(
    entry,
    buildRecentMessages(entry.recentTurns),
    format === 'narrative' ? 'narrative' : 'structured',
  );

  return {
    compacted_context: formatted,
    compaction_id: entry.compactionId,
    compacted_at: entry.compactedAt,
    conversation_id: entry.conversationId,
    current_state: entry.currentState,
    correlation_id: correlationId,
    original_token_count: null,
    compacted_token_count: null,
    compression_ratio: null,
    level_applied: null,
    validation_score: null,
    validation_passed: null,
    quality_warning: false,
  };
}
