// Duck-typed Synap SDK surface + eve session-context shape.
//
// Following the TS-integration convention (see synap-vercel-adk / synap-mastra):
// we do NOT import the real `@maximem/synap-js-sdk` type. Production users pass
// the real SDK; smoke tests pass a mock conforming to `SynapSdkLike`. The eve
// runtime context is likewise duck-typed (`EveSessionLike`) so surface code and
// tests never depend on eve's internal type graph — only the two public entry
// points (`eve/tools`, `eve/instructions`) are imported for real.

// ── Synap SDK (duck-typed) ────────────────────────────────────────────────────

export interface SynapFetchResponseLike {
  formatted_context?: string | null;
  facts?: unknown[];
  preferences?: unknown[];
  episodes?: unknown[];
  emotions?: unknown[];
  temporal_events?: unknown[];
}

export interface SynapFetchArgs {
  conversation_id?: string | null;
  user_id: string;
  customer_id?: string | null;
  search_query?: string[] | null;
  max_results?: number;
  mode?: string;
  include_conversation_context?: boolean;
}

export interface SynapMemoryCreateArgs {
  document: string;
  user_id: string;
  customer_id?: string | null;
  metadata?: Record<string, unknown>;
}

export interface SynapMemoryCreateResult {
  ingestion_id: string;
}

export interface SynapRecentMessage {
  role: string;
  content: string;
  timestamp: string;
  message_id: string;
}

export interface SynapPromptContext {
  formatted_context: string;
  available: boolean;
  recent_messages: SynapRecentMessage[];
  recent_message_count: number;
  total_message_count: number;
}

export interface SynapSdkLike {
  fetch(args: SynapFetchArgs): Promise<SynapFetchResponseLike>;
  conversation: {
    context: {
      get_context_for_prompt(args: {
        conversation_id: string;
        /** Optional formatting style — structured | narrative | bullet_points. */
        style?: string;
      }): Promise<SynapPromptContext>;
    };
  };
  memories: {
    create(args: SynapMemoryCreateArgs): Promise<SynapMemoryCreateResult>;
  };
}

// ── eve runtime context (duck-typed) ──────────────────────────────────────────
//
// Mirrors eve's `SessionContext.session` / `DynamicResolveContext.session`
// (id + auth), and `SessionAuth` (current + initiator `SessionAuthContext`).
// Both a tool's `ctx` and an instructions resolver's `ctx` satisfy this.

export interface EveAuthContextLike {
  readonly principalId?: string;
  readonly subject?: string;
}

export interface EveSessionLike {
  readonly session: {
    readonly id: string;
    readonly auth?: {
      readonly current?: EveAuthContextLike | null;
      readonly initiator?: EveAuthContextLike | null;
    } | null;
  };
}

// ── Error policy ──────────────────────────────────────────────────────────────
//
// The TS packages have no shared `synap-integrations-common`, so we re-declare
// the error type locally (same contract as the Python integrations):
//   reads degrade → [] / empty; writes raise; deletes warn + no-op.

export class SynapIntegrationError extends Error {
  readonly operation: string;
  readonly cause?: unknown;

  constructor(operation: string, message: string, cause?: unknown) {
    super(`${operation}: ${message}`);
    this.name = "SynapIntegrationError";
    this.operation = operation;
    this.cause = cause;
  }
}

// ── Identity resolution ───────────────────────────────────────────────────────

/**
 * Resolve the Synap user scope for a turn. Prefers an explicit id, then the
 * turn's *current* principal. Returns `undefined` when neither is present —
 * callers apply the read/write policy accordingly.
 *
 * Deliberately does NOT fall back to `auth.initiator`: on a delegated or
 * system-initiated session `initiator` is whoever *started* the session (e.g.
 * an admin/service), not the end user this turn acts for — using it would scope
 * memory to the wrong principal. When `current` is null (unauthenticated
 * channel), pass an explicit `userId` instead.
 */
export function resolveUserId(
  explicit: string | undefined,
  ctx?: EveSessionLike,
): string | undefined {
  if (explicit && explicit.trim()) return explicit;
  const principal = ctx?.session?.auth?.current?.principalId;
  if (principal && principal.trim()) return principal;
  return undefined;
}

/** Resolve the conversation scope: explicit id, else the eve session id. */
export function resolveConversationId(
  explicit: string | undefined,
  ctx?: EveSessionLike,
): string | undefined {
  if (explicit && explicit.trim()) return explicit;
  const id = ctx?.session?.id;
  return id && id.trim() ? id : undefined;
}
