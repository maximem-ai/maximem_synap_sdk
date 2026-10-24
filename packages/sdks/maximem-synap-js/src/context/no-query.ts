/**
 * A fetch with no query. Mirrors `has_query`, `conversation_state_context` and
 * `profile_context` in the Python SDK's `models/context.py`; the corpus both
 * run is `CONTRACT/conformance/no_query_fetch.json`.
 *
 * Without a query the server embedded an empty string and ranked memories
 * against it, so the answer was close to arbitrary. A conversation fetch now
 * answers with the conversation's state (what `get_context_for_prompt` gives)
 * and a user fetch with the user's profile when there is one.
 */

import type { Json, RawContext } from './types.js';

const MEMORY_TYPES = ['facts', 'preferences', 'episodes', 'emotions', 'temporal_events'] as const;

const emptyLists = (): Record<string, unknown[]> =>
  Object.fromEntries(MEMORY_TYPES.map((t) => [t, []]));

/** Whether a fetch names something to search for. Blank strings are no query. */
export function hasQuery(searchQuery: unknown): boolean {
  if (typeof searchQuery === 'string') return searchQuery.trim() !== '';
  if (!Array.isArray(searchQuery)) return false;
  return searchQuery.some((q) => typeof q === 'string' && q.trim() !== '');
}

/** The answer to a conversation fetch without a query, in the fetch's wire shape. */
export function conversationStateContext(prompt: Json, conversationId: string): RawContext {
  const recent = Array.isArray(prompt['recent_messages']) ? (prompt['recent_messages'] as unknown[]) : [];
  const turns = recent
    .filter((m): m is Record<string, unknown> => typeof m === 'object' && m !== null && !Array.isArray(m))
    .map((m) => ({ role: m['role'] ?? null, content: m['content'] ?? null, timestamp: m['timestamp'] ?? null }));
  return {
    ...emptyLists(),
    conversation_context: {
      conversation_id: conversationId,
      summary: (prompt['context_block'] as string | undefined) || null,
      recent_turns: turns,
      compaction_id: (prompt['compaction_id'] as string | undefined) || null,
      formatted_context: (prompt['formatted_context'] as string | undefined) || null,
    },
  } as unknown as RawContext;
}

/**
 * The answer to a user fetch without a query, or null when there is no profile.
 * Takes the profile document flat or wrapped in `{ profile: ... }`. One with no
 * attributes and no overview counts as none.
 */
export function profileContext(profile: unknown): RawContext | null {
  if (typeof profile !== 'object' || profile === null || Array.isArray(profile)) return null;
  let doc = profile as Record<string, unknown>;
  const inner = doc['profile'];
  if (!('attributes' in doc) && typeof inner === 'object' && inner !== null && !Array.isArray(inner)) {
    doc = inner as Record<string, unknown>;
  }
  const attributes = doc['attributes'];
  const hasAttributes = typeof attributes === 'object' && attributes !== null && Object.keys(attributes).length > 0;
  if (!hasAttributes && !doc['overview']) return null;
  return { ...emptyLists(), profile: doc } as unknown as RawContext;
}
