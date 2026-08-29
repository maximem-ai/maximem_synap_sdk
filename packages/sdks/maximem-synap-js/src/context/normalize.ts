/**
 * camelCase normalisers for the legacy top-level methods.
 *
 * A faithful port of the normalisers in the removed `src/synap-client.js`,
 * including their quirks:
 *
 *  - `score` and `source` on a memory item are passed through WITHOUT a
 *    default, so `undefined` survives. Adding `?? 0` here would change the
 *    shape consumers already branch on.
 *  - `strength` falls back to `confidence`, `summary` falls back to `content`,
 *    `intensity` falls back to `confidence`. Those fallbacks exist because the
 *    server does not always send the specific field.
 *  - `pickDefined` treats `null` as a real value and only skips `undefined`,
 *    so an explicit server `null` is preserved rather than replaced.
 *
 * See gotcha G-B: this shape is one of two live shapes and must not converge
 * with the raw one before 2.0.
 */

import type {
  ConversationContext, ContextMetadata, Emotion, Episode, Fact, Json,
  NormalisedContext, Preference, RawContext, RawContextItem,
  RawConversationContext, TemporalEvent, TemporalFields,
} from './types.js';

/** Returns the first argument that is not `undefined`. `null` counts as a value. */
export function pickDefined<T>(...values: (T | undefined)[]): T | undefined {
  for (const v of values) if (v !== undefined) return v;
  return undefined;
}

function pick<T>(...values: (T | undefined)[]): T {
  return pickDefined(...values) as T;
}

const obj = (v: unknown): Json => (v as Json) ?? {};

function normalizeTemporalFields(item: RawContextItem = {}): TemporalFields {
  return {
    eventDate: pick(item['eventDate'] as string, item.event_date, null),
    validUntil: pick(item['validUntil'] as string, item.valid_until, null),
    temporalCategory: pick(item['temporalCategory'] as string, item.temporal_category, null),
    temporalConfidence: pick(item['temporalConfidence'] as number, item.temporal_confidence, 0),
  };
}

export function normalizeFact(item: RawContextItem = {}): Fact {
  return {
    id: item.id || '',
    content: item.content || '',
    confidence: pick(item.confidence, 0),
    source: item.source || '',
    extractedAt: pick(item['extractedAt'] as string, item.extracted_at, null),
    metadata: obj(item.metadata),
    ...normalizeTemporalFields(item),
  };
}

export function normalizePreference(item: RawContextItem = {}): Preference {
  return {
    id: item.id || '',
    category: (item['category'] as string) || '',
    content: item.content || '',
    strength: pick(item['strength'] as number, item.confidence, 0),
    source: item.source || '',
    extractedAt: pick(item['extractedAt'] as string, item.extracted_at, null),
    metadata: obj(item.metadata),
    ...normalizeTemporalFields(item),
  };
}

export function normalizeEpisode(item: RawContextItem = {}): Episode {
  return {
    id: item.id || '',
    summary: pick(item['summary'] as string, item.content, ''),
    occurredAt: pick(item['occurredAt'] as string, item['occurred_at'] as string, null),
    significance: pick(item['significance'] as number, item.confidence, 0),
    participants: (item['participants'] as unknown[]) || [],
    metadata: obj(item.metadata),
    ...normalizeTemporalFields(item),
  };
}

export function normalizeEmotion(item: RawContextItem = {}): Emotion {
  return {
    id: item.id || '',
    emotionType: pick(item['emotionType'] as string, item['emotion_type'] as string, ''),
    intensity: pick(item['intensity'] as number, item.confidence, 0),
    detectedAt: pick(item['detectedAt'] as string, item['detected_at'] as string, null),
    context: (item['context'] as string) || '',
    metadata: obj(item.metadata),
    ...normalizeTemporalFields(item),
  };
}

export function normalizeTemporalEvent(item: RawContextItem = {}): TemporalEvent {
  return {
    id: item.id || '',
    content: item.content || '',
    eventDate: pick(item['eventDate'] as string, item.event_date, null),
    validUntil: pick(item['validUntil'] as string, item.valid_until, null),
    temporalCategory: pick(item['temporalCategory'] as string, item.temporal_category, ''),
    temporalConfidence: pick(item['temporalConfidence'] as number, item.temporal_confidence, 0),
    confidence: pick(item.confidence, 0),
    source: item.source || '',
    extractedAt: pick(item['extractedAt'] as string, item.extracted_at, null),
    metadata: obj(item.metadata),
  };
}

export function normalizeContextMetadata(metadata: Json = {}): ContextMetadata {
  return {
    correlationId: pick(metadata['correlationId'] as string, metadata['correlation_id'] as string, ''),
    ttlSeconds: pick(metadata['ttlSeconds'] as number, metadata['ttl_seconds'] as number, 0),
    source: (metadata['source'] as string) || 'unknown',
    retrievedAt: pick(metadata['retrievedAt'] as string, metadata['retrieved_at'] as string, null),
    compactionApplied: pick(
      metadata['compactionApplied'] as boolean,
      metadata['compaction_applied'] as boolean,
      null,
    ),
  };
}

export function normalizeConversationContext(
  value: RawConversationContext | null | undefined,
): ConversationContext | null {
  if (!value) return null;
  return {
    summary: value.summary || null,
    currentState: pick(value['currentState'] as Json, value.current_state, {}),
    keyExtractions: pick(value['keyExtractions'] as Json, value.key_extractions, {}),
    recentTurns: pick(value['recentTurns'] as unknown[], value.recent_turns, []),
    compactionId: pick(value['compactionId'] as string, value.compaction_id, null),
    compactedAt: pick(value['compactedAt'] as string, value.compacted_at, null),
    conversationId: pick(value['conversationId'] as string, value.conversation_id, null),
  };
}

/** Accepts either a bare context or a `{ context: ... }` envelope, as the wrapper did. */
export function normalizeContextResponse(result: Json = {}): NormalisedContext {
  const context = (result['context'] as RawContext) ?? (result as RawContext);
  return {
    facts: (context.facts || []).map((i) => normalizeFact(i)),
    preferences: (context.preferences || []).map((i) => normalizePreference(i)),
    episodes: (context.episodes || []).map((i) => normalizeEpisode(i)),
    emotions: (context.emotions || []).map((i) => normalizeEmotion(i)),
    temporalEvents: (
      pick(context['temporalEvents'] as RawContextItem[], context.temporal_events, []) || []
    ).map((i) => normalizeTemporalEvent(i)),
    conversationContext: normalizeConversationContext(
      pick(
        context['conversationContext'] as RawConversationContext,
        context.conversation_context,
        null,
      ),
    ),
    metadata: normalizeContextMetadata(obj(context.metadata)),
    rawResponse: pick(context['rawResponse'] as Json, context['raw_response'] as Json, {}),
  };
}
