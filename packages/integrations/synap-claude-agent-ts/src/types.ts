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

export interface SynapFetchArgs {
  conversation_id?: string | null;
  user_id: string;
  customer_id?: string | null;
  search_query?: string[] | null;
  max_results?: number;
  mode?: string;
  include_conversation_context?: boolean;
}

export interface SynapShortTermResponseLike {
  /** Cache-first formatted block ready to embed in a system prompt. */
  formatted_context?: string | null;
  /** Whether the server reported any short-term content. */
  available?: boolean;
}

export interface SynapSdkLike {
  fetch(args: SynapFetchArgs): Promise<SynapFetchResponseLike>;
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
  /** Optional customer/org scope. */
  customerId?: string;
  /**
   * Static conversation id. When omitted, the hook falls back to the SDK's
   * per-session `session_id` on each invocation.
   */
  conversationId?: string;
}
