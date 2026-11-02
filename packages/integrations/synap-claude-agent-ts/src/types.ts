// Duck-typed SDK shape. Consumers pass any object conforming to this — the
// real Synap JS SDK is a Python-bridge wrapper; smoke tests use a mock.
// Kept minimal: only the four async methods we actually call.

export interface SynapFetchResponseLike {
  formatted_context?: string | null;
  facts?: unknown[];
  preferences?: unknown[];
  episodes?: unknown[];
  emotions?: unknown[];
  temporal_events?: unknown[];
}

export interface SynapRecordMessageArgs {
  conversation_id: string;
  role: string;
  content: string;
  user_id: string;
  customer_id?: string;
  /** The level of a deeper scope ladder this conversation belongs to. Sent only when set. */
  scope_path?: Record<string, string>;
}

export interface SynapMemoryCreateArgs {
  document: string;
  user_id: string;
  customer_id?: string | null;
  metadata?: Record<string, unknown>;
  /**
   * The level of a deeper scope ladder to write at. NOTE the name: `scope` on
   * `memories.create`, `scope_path` everywhere else. Sent only when set.
   */
  scope?: Record<string, string>;
}

export interface SynapMemoryCreateResult {
  ingestion_id: string;
}

export interface SynapFetchArgs {
  conversation_id?: string | null;
  user_id: string;
  customer_id?: string | null;
  // Not `| null`. The two call sites (hooks.ts, mcp-server.ts) always pass an
  // array, and the SDK has no null handling for it the way it does for the
  // ids, so widening it here only promised something nobody needs and the SDK
  // would not honour.
  search_query?: string[];
  max_results?: number;
  mode?: string;
  include_conversation_context?: boolean;
  /** The level of a deeper scope ladder to read at. Sent only when set. */
  scope_path?: Record<string, string>;
}

export interface SynapShortTermResponseLike {
  /** Cache-first formatted block ready to embed in a system prompt. */
  formatted_context?: string | null;
  /** Whether the server reported any short-term content. */
  available?: boolean;
  /** The framed compaction block; absent before the first compaction and on older SDKs. */
  context_block?: string | null;
  /** Earlier turns the SDK found for the latest message, rendered. */
  history_block?: string | null;
  /** The turns after the compaction. */
  recent_messages?: Array<{ role: string; content: string }>;
  compaction_id?: string | null;
}

export interface SynapSdkLike {
  fetch(args: SynapFetchArgs): Promise<SynapFetchResponseLike>;
  /**
   * Present on `@maximem/synap-js-sdk` >= 0.7.1, the first release that sends
   * `scope_path`. Not called here: its presence is how a `scopePath` on an older
   * SDK, which would silently ignore it, is refused at construction.
   */
  get_scope_ladder?: (...args: never[]) => unknown;
  conversation: {
    record_message(args: SynapRecordMessageArgs): Promise<unknown>;
    /** Optional — present on real SDKs; the short-term hook requires it. */
    context?: {
      get_context_for_prompt(args: {
        conversation_id: string;
        style?: string;
      }): Promise<SynapShortTermResponseLike>;
    };
  };
  memories: {
    create(args: SynapMemoryCreateArgs): Promise<SynapMemoryCreateResult>;
  };
}

export interface SynapIdentityOptions {
  /** Configured Synap SDK. */
  sdk: SynapSdkLike;
  /** Synap user scope (required). */
  userId: string;
  /** B2B instances only, where it is required. NOT accepted on a B2C
   * instance: the server rejects a call carrying one with HTTP 400. */
  customerId?: string;
  /**
   * Static conversation id. When omitted, the hook falls back to the SDK's
   * per-session `session_id` on each invocation.
   */
  conversationId?: string;
  /**
   * The level of your account's scope ladder to read and write at, for an
   * account whose ladder has a level between customer and user, for example
   * `{ customer: "acme", team: "eng", user: "u1" }`. Fixed at construction.
   * Reads send it as `scope_path`, memory writes as `scope`, conversation
   * turns as `scope_path`. For the MCP tools it is never a parameter the model
   * sees, and a `scope_path` in the model's arguments is ignored. Unset on a
   * default ladder, where every call is exactly what it was.
   */
  scopePath?: Record<string, string>;
}

/** `{ [key]: path }` when a path was given and has entries, else `{}`, so an unchanged call stays unchanged. */
export function withScopePath<K extends string>(
  key: K,
  path: Record<string, string> | undefined | null,
): { [P in K]?: Record<string, string> } {
  return path && Object.keys(path).length > 0
    ? ({ [key]: { ...path } } as { [P in K]?: Record<string, string> })
    : {};
}
