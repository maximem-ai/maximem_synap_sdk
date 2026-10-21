/**
 * Cross-scope merge + prompt formatting for `client.fetch()`.
 *
 * Ports Python's `UnifiedContextResponse.merge` and `format_for_prompt` from
 * `models/context.py`. The formatted string is the part customers actually
 * paste into a system prompt, so it is reproduced character for character:
 * section order, the `- ` bullet, the `[scope: x]` / `strength: 0.8` bracket
 * suffixes, the `## User Context` wrapper, and the blank-line joins. A
 * cosmetic drift here shows up as a diff in every prompt a customer builds.
 */

import { rankedLabel, sortByRankScore } from './ranked.js';
import type { Json, RawContext, RawContextItem } from './types.js';

/** The five item collections, in Python's iteration order. */
export const ITEM_COLLECTIONS = [
  'facts',
  'preferences',
  'episodes',
  'emotions',
  'temporal_events',
] as const;

export type ItemCollection = (typeof ITEM_COLLECTIONS)[number];

export interface UnifiedContext {
  facts: RawContextItem[];
  preferences: RawContextItem[];
  episodes: RawContextItem[];
  emotions: RawContextItem[];
  temporal_events: RawContextItem[];
  /**
   * `response_shape: "ranked"` only: every item once, best first across
   * scopes and types, renumbered from 1. Empty for a by-type fetch.
   */
  items: RawContextItem[];
  /** `response_shape: "ranked"` only: the chunks (quotes from source documents). */
  excerpts: RawContextItem[];
  /** item id -> the scope that supplied it. */
  scope_map: Record<string, string>;
  conversation_context: Json | null;
  profile: Json | null;
  conversations: Json[] | null;
  formatted_context: string;
  scopes_queried: string[];
  total_items: number;
  metadata: Json | null;
}

function itemId(item: RawContextItem): string | undefined {
  const id = item.id ?? item.item_id;
  return typeof id === 'string' && id !== '' ? id : undefined;
}

/**
 * Merge per-scope responses, deduplicating by item id.
 *
 * First scope wins on a duplicate, so the caller's scope order is significant.
 * An item with no usable id is kept rather than dropped: Python indexes
 * `scope_map` by `item.id`, and a missing id there would collide every such
 * item onto one key, so keeping-and-not-mapping is the closest faithful
 * behaviour.
 */
export function mergeScopeResults(
  scopeResults: ReadonlyArray<readonly [string, RawContext]>,
): UnifiedContext {
  const seen = new Set<string>();
  const out: UnifiedContext = {
    facts: [], preferences: [], episodes: [], emotions: [], temporal_events: [],
    items: [], excerpts: [],
    scope_map: {},
    conversation_context: null,
    profile: null,
    conversations: null,
    formatted_context: '',
    scopes_queried: [],
    total_items: 0,
    metadata: null,
  };

  const rankedSeen = new Set<string>();
  const rankedItems: RawContextItem[] = [];

  for (const [scopeName, response] of scopeResults) {
    out.scopes_queried.push(scopeName);
    if (out.metadata === null && response.metadata !== undefined) {
      out.metadata = response.metadata as Json;
    }
    // The C4 summary sections ride up from whichever scope produced them,
    // which is always the user-scope sub-fetch. See the fan-out rule in fetch().
    if (out.profile === null && response['profile'] !== undefined && response['profile'] !== null) {
      out.profile = response['profile'] as Json;
    }
    if (
      out.conversations === null &&
      response['conversations'] !== undefined &&
      response['conversations'] !== null
    ) {
      out.conversations = response['conversations'] as Json[];
    }

    for (const collection of ITEM_COLLECTIONS) {
      const items = response[collection];
      if (!Array.isArray(items)) continue;
      for (const item of items as RawContextItem[]) {
        const id = itemId(item);
        if (id !== undefined) {
          if (seen.has(id)) continue;
          seen.add(id);
          out.scope_map[id] = scopeName;
        }
        out[collection].push(item);
      }
    }

    // Ranked responses only, and in Python's order: excerpts share the typed
    // lists' dedup and scope map; the ranked items have a dedup of their own,
    // because they are the same items again, in order.
    const excerpts = response['excerpts'];
    if (Array.isArray(excerpts)) {
      for (const item of excerpts as RawContextItem[]) {
        const id = itemId(item);
        if (id !== undefined) {
          if (seen.has(id)) continue;
          seen.add(id);
          out.scope_map[id] = scopeName;
        }
        out.excerpts.push(item);
      }
    }
    const ranked = response['items'];
    if (Array.isArray(ranked)) {
      for (const item of ranked as RawContextItem[]) {
        const id = itemId(item);
        if (id !== undefined) {
          if (rankedSeen.has(id)) continue;
          rankedSeen.add(id);
        }
        rankedItems.push(item);
      }
    }
  }

  // One order across scopes, by each item's rank_score. The sort is stable, so
  // equal scores keep scope order, as Python's does, and unscored items go last.
  out.items = sortByRankScore(rankedItems).map((item, n) => ({ ...item, rank: n + 1 }));
  out.total_items =
    ITEM_COLLECTIONS.reduce((n, c) => n + out[c].length, 0) + out.excerpts.length;
  return out;
}

/** Python's `f"{x:.1f}"`. */
function oneDecimal(value: number): string {
  return value.toFixed(1);
}

/** Python's `datetime.strftime("%Y-%m-%d")` on an ISO string. */
function isoDate(value: unknown): string | null {
  if (typeof value !== 'string' || value === '') return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toISOString().slice(0, 10);
}

function numberOf(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function scopeOf(ctx: UnifiedContext, item: RawContextItem): string {
  const id = itemId(item);
  return (id !== undefined ? ctx.scope_map[id] : undefined) ?? 'unknown';
}

/** `- text [a, b]`, with the bracket omitted when there are no extras. */
function bulletWithExtras(text: string, extras: string[]): string {
  return extras.length > 0 ? `- ${text} [${extras.join(', ')}]` : `- ${text}`;
}

export interface FormatOptions {
  includeScope?: boolean;
  includeConversationContext?: boolean;
}

/**
 * Render the merged context for prompt injection.
 *
 * Mirrors `UnifiedContextResponse.format_for_prompt`. Note the two different
 * bracket styles: facts and episodes only ever carry `[scope: x]` appended
 * directly, while preferences, emotions and temporal events build a joined
 * extras list. That asymmetry is in Python and is preserved.
 */
export function formatForPrompt(ctx: UnifiedContext, options: FormatOptions = {}): string {
  const includeScope = options.includeScope ?? false;
  const includeConversationContext = options.includeConversationContext ?? true;
  const sections: string[] = [];

  // A ranked response is one numbered list in rank order, as in Python: the
  // typed sections below would split the order back up by type.
  const ranked = ctx.items ?? [];
  if (ranked.length > 0) {
    const lines = ranked.map((it, n) => {
      let label = rankedLabel(String(it['category'] ?? ''));
      if (includeScope) label += `, scope: ${scopeOf(ctx, it)}`;
      return `${n + 1}. [${label}] ${String(it.content ?? '')}`;
    });
    sections.push(`### Memories\n${lines.join('\n')}`);
  }
  const typed = ranked.length === 0;

  if (typed && ctx.facts.length > 0) {
    const lines = ctx.facts.map((f) => {
      let line = `- ${String(f.content ?? '')}`;
      if (includeScope) line += ` [scope: ${scopeOf(ctx, f)}]`;
      return line;
    });
    sections.push(`### Facts\n${lines.join('\n')}`);
  }

  if (typed && ctx.preferences.length > 0) {
    const lines = ctx.preferences.map((p) => {
      const extras: string[] = [];
      if (includeScope) extras.push(`scope: ${scopeOf(ctx, p)}`);
      const strength = numberOf(p['strength']);
      // Python: `if p.strength and p.strength >= 0.7` -- 0 is falsy there, so a
      // zero strength is omitted rather than printed.
      if (strength !== null && strength !== 0 && strength >= 0.7) {
        extras.push(`strength: ${oneDecimal(strength)}`);
      }
      return bulletWithExtras(String(p.content ?? ''), extras);
    });
    sections.push(`### Preferences\n${lines.join('\n')}`);
  }

  if (typed && ctx.episodes.length > 0) {
    const lines = ctx.episodes.map((e) => {
      // Episodes render `summary`, not `content`.
      let line = `- ${String(e['summary'] ?? '')}`;
      if (includeScope) line += ` [scope: ${scopeOf(ctx, e)}]`;
      return line;
    });
    sections.push(`### Episodes\n${lines.join('\n')}`);
  }

  if (typed && ctx.emotions.length > 0) {
    const lines = ctx.emotions.map((em) => {
      const extras: string[] = [];
      if (includeScope) extras.push(`scope: ${scopeOf(ctx, em)}`);
      const intensity = numberOf(em['intensity']);
      if (intensity !== null && intensity !== 0 && intensity >= 0.5) {
        extras.push(`intensity: ${oneDecimal(intensity)}`);
      }
      // Emotions render "type: context", not a bare content field.
      const text = `${String(em['emotion_type'] ?? '')}: ${String(em['context'] ?? '')}`;
      return bulletWithExtras(text, extras);
    });
    sections.push(`### Emotions\n${lines.join('\n')}`);
  }

  if (typed && ctx.temporal_events.length > 0) {
    const lines = ctx.temporal_events.map((te) => {
      const extras: string[] = [];
      if (includeScope) extras.push(`scope: ${scopeOf(ctx, te)}`);
      const validUntil = isoDate(te['valid_until']);
      if (validUntil !== null) extras.push(`valid until: ${validUntil}`);
      return bulletWithExtras(String(te.content ?? ''), extras);
    });
    sections.push(`### Temporal Events\n${lines.join('\n')}`);
  }

  if (includeConversationContext && ctx.conversation_context !== null) {
    const formatted = ctx.conversation_context['formatted_context'];
    if (typeof formatted === 'string' && formatted !== '') {
      sections.push(`### Conversation History\n${formatted}`);
    }
  }

  const blocks: string[] = [];
  if (sections.length > 0) blocks.push(`## User Context\n${sections.join('\n\n')}`);

  const profileBlock = formatProfileBlock(ctx.profile);
  if (profileBlock !== null) blocks.push(profileBlock);

  const conversationsBlock = formatConversationsBlock(ctx.conversations);
  if (conversationsBlock !== null) blocks.push(conversationsBlock);

  return blocks.length > 0 ? blocks.join('\n\n') : '';
}

/** `## Caller Profile`, or null when there is nothing to show. */
function formatProfileBlock(profile: Json | null): string | null {
  if (profile === null) return null;
  const attributes = profile['attributes'];
  const lines: string[] = [];
  if (typeof attributes === 'object' && attributes !== null) {
    for (const [name, attr] of Object.entries(attributes as Json)) {
      const value =
        typeof attr === 'object' && attr !== null
          ? (attr as Json)['value']
          : undefined;
      let valueStr: string;
      if (Array.isArray(value)) valueStr = value.map((v) => String(v)).join(', ');
      else if (value === null || value === undefined) valueStr = '';
      else valueStr = String(value);
      // Python does `f"- {name}: {value}".rstrip()`, so an empty value leaves
      // "- name:" with no trailing space.
      lines.push(`- ${name}: ${valueStr}`.replace(/\s+$/, ''));
    }
  }
  const bodyParts: string[] = [];
  if (lines.length > 0) bodyParts.push(lines.join('\n'));
  const overview = profile['overview'];
  if (typeof overview === 'string' && overview !== '') bodyParts.push(overview);
  if (bodyParts.length === 0) return null;
  return `## Caller Profile\n${bodyParts.join('\n\n')}`;
}

/** `## Previous Conversations`, or null. */
function formatConversationsBlock(conversations: Json[] | null): string | null {
  if (conversations === null || conversations.length === 0) return null;
  const callBlocks = conversations.map((conv) => {
    const when =
      isoDate(conv['started_at']) ?? isoDate(conv['last_message_at']) ?? isoDate(conv['ended_at']);
    let header = `### Call on ${when ?? 'unknown date'}`;
    const convType = conv['conversation_type'];
    if (typeof convType === 'string' && convType !== '') header += ` (${convType})`;
    const callLines = [header];
    const overview = overviewText(conv);
    if (overview !== null) callLines.push(`Overview: ${overview}`);
    const outcome = outcomeText(conv);
    if (outcome !== null) callLines.push(`Outcome: ${outcome}`);
    if (overview === null && outcome === null) {
      // No compaction yet, or it failed. Still tell the model the call
      // happened so it can acknowledge the prior contact.
      // Python's model defaults summary_status to "pending", so an absent
      // field renders as "pending", not as an empty string.
      callLines.push(`Status: ${String(conv['summary_status'] ?? 'pending')}`);
    }
    return callLines.join('\n');
  });
  return `## Previous Conversations\n${callBlocks.join('\n\n')}`;
}

/**
 * Mirrors `ConversationSummaryModel.overview_text()`.
 *
 * Reads the nested `summary` OBJECT (not a string) and takes the first of four
 * keys that holds non-blank text, in this exact order.
 */
function overviewText(conv: Json): string | null {
  const summary = conv['summary'];
  if (typeof summary !== 'object' || summary === null || Array.isArray(summary)) return null;
  for (const key of ['narrative_summary', 'overview', 'narrative', 'summary'] as const) {
    const val = (summary as Json)[key];
    if (typeof val === 'string' && val.trim() !== '') return val.trim();
  }
  return null;
}

/**
 * Mirrors `ConversationSummaryModel.outcome_text()`.
 *
 * Joins up to three classification fields with " / ", skipping blanks, and
 * returns null when nothing survives.
 */
function outcomeText(conv: Json): string | null {
  const classification = conv['classification'];
  if (typeof classification !== 'object' || classification === null || Array.isArray(classification)) {
    return null;
  }
  const c = classification as Json;
  const parts = [c['primary_category'], c['subcategory'], c['objective']]
    .filter((p): p is string => typeof p === 'string' && p.trim() !== '');
  return parts.length > 0 ? parts.join(' / ') : null;
}
