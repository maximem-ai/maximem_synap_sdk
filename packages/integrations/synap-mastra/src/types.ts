// Duck-typed Synap SDK surface — the same four async methods we rely on
// everywhere. Smoke tests pass a mock conforming to this shape; production
// users pass the real @maximem/synap-js-sdk wrapper.

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
    record_message(args: SynapRecordMessageArgs): Promise<unknown>;
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

export interface SynapIdentityOptions {
  /** Configured Synap SDK. */
  sdk: SynapSdkLike;
  /** Synap user scope (required). */
  userId: string;
  /** Optional customer/org scope. */
  customerId?: string;
  /** Synap fetch mode ("accurate" default, or "fast"). */
  mode?: string;
}
