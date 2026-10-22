/**
 * Read-your-writes overlay for fetch responses.
 *
 * Ports Python's `_overlay_local_recent_turns` and `_budget_recent_turns`.
 *
 * The server's `recent_turns` reflects what it has persisted and compacted. A
 * turn recorded moments ago may not be in there yet, so a fast follow-up would
 * fetch context missing the thing the user just said. The overlay replaces
 * `recent_turns` with the locally buffered tail while leaving every summary
 * field to the server.
 *
 * On by default, matching Python. `SYNAP_ST_VERBATIM_OVERLAY` is the
 * kill-switch.
 */

import type { Json, RawContext } from './types.js';
import { isComplete, turnsAfter } from '../cache/short-term-store.js';
import type { ShortTermStore, ShortTermTurn } from '../cache/short-term-store.js';
import { getEnv } from '../util/env.js';

/**
 * The server trims `recent_turns` to this window before returning it. The
 * overlay re-applies the same budget so a long pre-compaction local tail
 * cannot silently re-inflate the prompt past what the server would have
 * allowed.
 */
const MAX_TURNS = 20;
const TOKEN_BUDGET = 2000;

/**
 * Whether to splice the local tail in.
 *
 * Default ON: this is the correctness switch (a just-written turn must surface
 * on the next fetch), separate from the cost-side `st_authoritative` flag that
 * tells the server to skip assembling short-term context at all.
 */
export function verbatimOverlayEnabled(configured?: boolean): boolean {
  // Resolution order matches Python's `_is_st_verbatim_overlay`: the env var
  // is an explicit override / kill-switch and wins over the config field.
  const raw = (getEnv('SYNAP_ST_VERBATIM_OVERLAY') ?? '').trim().toLowerCase();
  if (['0', 'false', 'no', 'off'].includes(raw)) return false;
  if (['1', 'true', 'yes', 'on'].includes(raw)) return true;
  return configured ?? true;
}

/** ~4 characters per token, matching the server-side heuristic. */
function estimateTokens(turn: ShortTermTurn): number {
  return Math.max(1, Math.floor(String(turn.content ?? '').length / 4));
}

/**
 * Cap the tail to the server's window, then drop OLDEST turns until it fits
 * the token budget. Oldest-first because the newest turn is the one the caller
 * is most likely to be following up on.
 */
export function budgetRecentTurns(turns: readonly ShortTermTurn[]): ShortTermTurn[] {
  const windowed = turns.slice(-MAX_TURNS);
  let total = windowed.reduce((n, t) => n + estimateTokens(t), 0);
  while (windowed.length > 0 && total > TOKEN_BUDGET) {
    const dropped = windowed.shift() as ShortTermTurn;
    total -= estimateTokens(dropped);
  }
  return windowed;
}

/**
 * Splice the local tail into a fetch response, in place.
 *
 * Deliberately never throws: this runs inside the fetch path, and a bookkeeping
 * problem here must not fail a retrieval that otherwise succeeded.
 */
export function overlayLocalRecentTurns(
  response: RawContext,
  store: ShortTermStore,
  /** `null` too: callers spell "no conversation" as `convId || null`. */
  conversationId: string | null | undefined,
  /** `st_verbatim_overlay` from the client options. The env var still wins. */
  configured?: boolean,
): void {
  if (conversationId === undefined || conversationId === null || conversationId === '') return;
  try {
    const entry = store.get(conversationId);
    // Cold store: nothing locally, so leave the server's response untouched.
    if (entry === null) return;

    const existing = (response.conversation_context ?? null) as Json | null;

    // The server returned a conversation_context AND the overlay is switched
    // off: leave it exactly as sent. When the server OMITTED one we still
    // splice from local, so the skip-server-ST optimisation keeps working
    // regardless of this switch.
    if (existing !== null && !verbatimOverlayEnabled(configured)) return;

    // Which turns the tail is built from. The local list replaces the
    // server's only when it is known to be the whole conversation and to
    // follow the same summary the server sent. Otherwise the server's turns
    // stand, and the local list adds only what comes after them. A process
    // that has just restarted holds one turn, and replacing the server's list
    // with it dropped every turn before the restart.
    const serverCompaction = existing === null ? null : existing['compaction_id'];
    const localIsWhole =
      isComplete(entry) && (!serverCompaction || serverCompaction === entry.compactionId);
    let sourceTurns: ShortTermTurn[] = entry.recentTurns;
    if (existing !== null && !localIsWhole) {
      const raw = Array.isArray(existing['recent_turns']) ? existing['recent_turns'] : [];
      const serverTurns: ShortTermTurn[] = (raw as Array<Record<string, unknown>>).map((t) => ({
        role: String(t['role'] ?? 'user'),
        content: String(t['content'] ?? ''),
        timestamp: String(t['timestamp'] ?? ''),
      }));
      sourceTurns = [...serverTurns, ...turnsAfter(entry.recentTurns, serverTurns)];
    }
    // Role, text and time only, as Python sends: the server's id for a turn is
    // bookkeeping for the store, not part of the response.
    const localTurns = budgetRecentTurns(sourceTurns).map((t) => ({
      role: t.role,
      content: t.content,
      timestamp: t.timestamp,
    }));

    // `x || fallback`, not `??`, matching Python: the server sends '' and {}
    // for absent fields rather than null, and those must not win over a local
    // value that actually has content.
    const serverField = (name: string): unknown =>
      existing === null ? undefined : existing[name];
    const orLocal = <T>(name: string, local: T): unknown => {
      const value = serverField(name);
      if (value === undefined || value === null || value === '') return local;
      if (typeof value === 'object' && Object.keys(value as object).length === 0) return local;
      return value;
    };

    // Server wins on every summary field; local only ever supplies the tail.
    response.conversation_context = {
      summary: orLocal('summary', entry.summary),
      current_state: orLocal('current_state', { ...entry.currentState }),
      key_extractions: orLocal('key_extractions', { ...entry.keyExtractions }),
      recent_turns: localTurns,
      compaction_id: orLocal('compaction_id', entry.compactionId),
      compacted_at: orLocal('compacted_at', entry.compactedAt),
      conversation_id: conversationId,
    } as NonNullable<RawContext['conversation_context']>;
  } catch {
    // Non-fatal by design. See the note above.
  }
}
