/**
 * Synap short-term context — TypeScript port for the Vercel AI SDK adapter.
 *
 * Mirrors the Python LangGraph adapter template's quality contract:
 *
 *   - conversationId is required + explicit; never inferred from any
 *     framework-level session id.
 *   - SDK / HTTP failures never break the model call by default
 *     (onError="fallback"): logged at WARN, returns empty string and the
 *     caller's prompt is passed through unchanged. onError="raise" is
 *     available for tests / strict environments.
 *   - An empty short-term result (no compaction yet AND no recent turns
 *     server-side) is a no-op — must NOT wipe the user's system prompt.
 *   - Preamble tags are overridable; set both to undefined for raw concat.
 *
 * The v1 implementation hits the dedicated server endpoint
 *   GET /v1/conversations/{id}/context-for-prompt?style=...
 * directly. Cache-first behaviour (SDK-authoritative
 * `SYNAP_SDK_ST_AUTHORITATIVE` mode) is a follow-up — porting the
 * `ShortTermContextStore` + `compaction_update` bundle handling from
 * the Python SDK to TypeScript is a larger lift that lives outside
 * this PR.
 */

import type { LanguageModelV1Message, LanguageModelV1Prompt } from '@ai-sdk/provider';
import type { Credentials } from './types.js';

const DEFAULT_BASE_URL =
  // mirror the Python SDK env-var precedence: explicit option > env var > prod default
  (typeof process !== 'undefined' && process.env?.SYNAP_BASE_URL) ||
  'https://synap-cloud-prod.maximem.ai';

const SUPPORTED_STYLES = ['structured', 'narrative', 'bullet_points'] as const;
export type SynapShortTermStyle = (typeof SUPPORTED_STYLES)[number];

const DEFAULT_PREAMBLE_OPEN = '<synap_short_term_context>';
const DEFAULT_PREAMBLE_CLOSE = '</synap_short_term_context>';

export type SynapShortTermOnError = 'fallback' | 'raise';

export interface SynapShortTermResponse {
  /** Cache-first formatted block ready to embed in a system prompt; empty when none. */
  formattedContext: string;
  /** Whether the server reported any short-term content (compaction or recent turns). */
  available: boolean;
  /** Raw HTTP response payload, for callers that want to inspect deeper. */
  raw?: Record<string, unknown>;
}

export interface FetchShortTermContextOptions {
  credentials: Credentials;
  conversationId: string;
  style?: SynapShortTermStyle;
  baseUrl?: string;
  /** Defaults to 'fallback' — never throw on HTTP failure. */
  onError?: SynapShortTermOnError;
  /** Optional fetch override (lets tests inject a mock). Defaults to global fetch. */
  fetchImpl?: typeof fetch;
}

/**
 * Fetch Synap short-term context for a conversation.
 *
 * Returns `{ formattedContext: '', available: false }` on no-content
 * AND on HTTP failure with the default `onError: 'fallback'`.
 *
 * @throws if `conversationId` is empty.
 * @throws on HTTP failure when `onError: 'raise'`.
 */
export async function fetchShortTermContext(
  opts: FetchShortTermContextOptions,
): Promise<SynapShortTermResponse> {
  const { credentials, conversationId, style = 'narrative' } = opts;
  if (!credentials) {
    throw new TypeError('fetchShortTermContext requires credentials');
  }
  if (!conversationId || !conversationId.trim()) {
    throw new TypeError('fetchShortTermContext requires a non-empty conversationId');
  }
  if (!SUPPORTED_STYLES.includes(style as SynapShortTermStyle)) {
    throw new TypeError(
      `fetchShortTermContext: unsupported style=${JSON.stringify(style)}; expected one of ${SUPPORTED_STYLES.join(', ')}`,
    );
  }

  const onError: SynapShortTermOnError = opts.onError ?? 'fallback';
  const baseUrl = opts.baseUrl ?? DEFAULT_BASE_URL;
  const fetchImpl = opts.fetchImpl ?? fetch;
  const url =
    `${baseUrl}/v1/conversations/${encodeURIComponent(conversationId)}/context-for-prompt` +
    `?style=${encodeURIComponent(style)}`;
  const correlationId =
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `synap-${Date.now()}-${Math.random().toString(36).slice(2)}`;

  try {
    const res = await fetchImpl(url, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${credentials.api_key}`,
        'X-Synap-Instance-Id': credentials.instance_id ?? '',
        'X-Synap-Client-Id': credentials.client_id ?? '',
        'X-Correlation-Id': correlationId,
        Accept: 'application/json',
      },
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      const err = new Error(
        `Synap short-term context fetch failed (HTTP ${res.status}): ${text}`,
      );
      if (onError === 'raise') throw err;
      console.warn('[synap-vercel-adk] fetchShortTermContext failed:', err.message);
      return { formattedContext: '', available: false };
    }
    const data = (await res.json()) as Record<string, unknown>;
    const available = Boolean(data.available);
    const formattedContext =
      typeof data.formatted_context === 'string' ? data.formatted_context.trim() : '';
    if (!available || !formattedContext) {
      return { formattedContext: '', available, raw: data };
    }
    return { formattedContext, available, raw: data };
  } catch (err) {
    if (onError === 'raise') throw err;
    console.warn('[synap-vercel-adk] fetchShortTermContext failed:', err);
    return { formattedContext: '', available: false };
  }
}

export interface BuildShortTermSystemBlockOptions {
  /** Wrapping tags. Pass `null` to drop the wrapper and emit raw text. */
  preambleOpen?: string | null;
  preambleClose?: string | null;
}

/**
 * Wrap the response's `formattedContext` in the configured preamble.
 * Returns the empty string when there's no content — caller should
 * skip prepending a SystemMessage in that case.
 */
export function buildShortTermSystemBlock(
  response: SynapShortTermResponse,
  opts: BuildShortTermSystemBlockOptions = {},
): string {
  const body = response.formattedContext.trim();
  if (!body) return '';
  const open = opts.preambleOpen === undefined ? DEFAULT_PREAMBLE_OPEN : opts.preambleOpen;
  const close = opts.preambleClose === undefined ? DEFAULT_PREAMBLE_CLOSE : opts.preambleClose;
  if (open && close) {
    return `${open}\n${body}\n${close}`;
  }
  return body;
}

export interface InjectShortTermOptions extends BuildShortTermSystemBlockOptions {}

/**
 * Splice a Synap short-term context block above the prompt's system
 * message (or insert a new SystemMessage at the head when none exists).
 *
 * Mirrors the existing `injectContextIntoPrompt` shape but for ST only.
 * No-op when `response` has no content.
 */
export function injectShortTermIntoPrompt(
  prompt: LanguageModelV1Prompt,
  response: SynapShortTermResponse,
  opts: InjectShortTermOptions = {},
): LanguageModelV1Prompt {
  const block = buildShortTermSystemBlock(response, opts);
  if (!block) return prompt;

  const hasSystem = prompt.some(m => m.role === 'system');
  if (hasSystem) {
    return prompt.map((m): LanguageModelV1Message => {
      if (m.role !== 'system') return m;
      return { role: 'system', content: `${block}\n\n${m.content}` };
    });
  }
  const systemMsg: LanguageModelV1Message = { role: 'system', content: block };
  return [systemMsg, ...prompt];
}
