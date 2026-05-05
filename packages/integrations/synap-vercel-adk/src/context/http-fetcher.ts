import type { Credentials, FetchedContext, SynapModelOptions, RawContextItem, RawConversationContext } from '../types.js';
import { rawItemToContextItem } from '../transform/messages.js';

const DEFAULT_BASE_URL = 'https://synap-cloud-prod.maximem.ai';

interface FetchContextParams {
  credentials: Credentials;
  modelOptions: SynapModelOptions;
  searchQuery: string[];
  baseUrl?: string;
}

// ─── HTTP context fetch — mirrors Python SDK HTTP transport ───────────────────

export async function fetchContext(params: FetchContextParams): Promise<FetchedContext> {
  const { credentials, modelOptions, searchQuery, baseUrl = DEFAULT_BASE_URL } = params;

  // Determine which scope endpoint to hit
  const { endpoint, body } = buildRequest(modelOptions, searchQuery);
  const url = `${baseUrl}${endpoint}`;

  const correlationId = crypto.randomUUID();

  const res = await fetch(url, {
    method: 'POST',
    headers: buildHeaders(credentials, correlationId),
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Synap context fetch failed (HTTP ${res.status}): ${text}`);
  }

  const data = await res.json() as Record<string, unknown>;
  return parseContextResponse(data, correlationId);
}

function buildRequest(opts: SynapModelOptions, searchQuery: string[]): {
  endpoint: string;
  body: Record<string, unknown>;
} {
  const base = {
    search_query: searchQuery,
    max_results: opts.maxContextResults ?? 10,
    types: opts.contextTypes ?? ['all'],
    mode: 'fast',
  };

  // Pick the most specific scope available
  if (opts.conversationId) {
    return {
      endpoint: '/v1/context/conversation/fetch',
      body: { ...base, conversation_id: opts.conversationId, user_id: opts.userId ?? '', customer_id: opts.customerId ?? '' },
    };
  }
  if (opts.userId) {
    return {
      endpoint: '/v1/context/user/fetch',
      body: { ...base, user_id: opts.userId, customer_id: opts.customerId ?? '' },
    };
  }
  if (opts.customerId) {
    return {
      endpoint: '/v1/context/customer/fetch',
      body: { ...base, customer_id: opts.customerId },
    };
  }
  return {
    endpoint: '/v1/context/client/fetch',
    body: base,
  };
}

function buildHeaders(creds: Credentials, correlationId: string): Record<string, string> {
  return {
    'Authorization': `Bearer ${creds.api_key}`,
    'X-Client-ID': creds.client_id,
    'X-Instance-ID': creds.instance_id,
    'X-Correlation-ID': correlationId,
    'Content-Type': 'application/json',
    'User-Agent': '@maximem/synap-vercel-adk/0.2.5',
  };
}

function parseContextResponse(data: Record<string, unknown>, correlationId: string): FetchedContext {
  const itemsByType = (data['items_by_type'] as Record<string, { items: RawContextItem[] }> | undefined) ?? {};
  const convCtxRaw = data['conversation_context'] as RawConversationContext | undefined;

  return {
    facts: (itemsByType['facts']?.items ?? []).map(rawItemToContextItem),
    preferences: (itemsByType['preferences']?.items ?? []).map(rawItemToContextItem),
    episodes: (itemsByType['episodes']?.items ?? []).map(rawItemToContextItem),
    emotions: (itemsByType['emotions']?.items ?? []).map(rawItemToContextItem),
    temporalEvents: (itemsByType['temporal_events']?.items ?? []).map(rawItemToContextItem),
    conversationContext: convCtxRaw ? {
      summary: convCtxRaw.summary ?? null,
      currentState: parseJson(convCtxRaw.current_state_json),
      keyExtractions: parseJson(convCtxRaw.key_extractions_json),
      recentTurns: convCtxRaw.recent_turns ?? [],
      compactionId: convCtxRaw.compaction_id ?? null,
      conversationId: convCtxRaw.conversation_id ?? null,
    } : null,
    source: (data['metadata'] as Record<string, string> | undefined)?.['source'] === 'cache' ? 'cache' : 'cloud',
    correlationId,
  };
}

function parseJson(s: string | undefined): Record<string, unknown> {
  if (!s) return {};
  try { return JSON.parse(s) as Record<string, unknown>; }
  catch { return {}; }
}
