import type { Credentials, SynapModelOptions } from '../types.js';
import type { GrpcStreamClient } from '../grpc/stream-client.js';
import { isSynapHistory } from '../history.js';
import { newCorrelationId } from '../util/correlation.js';
import { SDK_VERSION } from '../version.js';

const DEFAULT_BASE_URL = 'https://synap-cloud-prod.maximem.ai';

export interface RecordTurnParams {
  credentials: Credentials;
  modelOptions: SynapModelOptions;
  role: 'user' | 'assistant';
  content: string;
  /** Resolves the live stream (opening it on first use), or null for REST. */
  getStream: () => Promise<GrpcStreamClient | null>;
  baseUrl?: string;
  fetchImpl?: typeof fetch;
}

/**
 * Record one conversation turn: the new user message or the model's reply.
 *
 * A turn is the conversation itself, not telemetry about it. Without turns the
 * server has nothing to compact. On the stream the server persists the event;
 * when the stream cannot carry it the turn goes to `POST /v1/conversations/messages`.
 * Never both. Never throws: a failed write must not break the caller's response.
 *
 * Replaces the old transcript ingest, which posted the whole prompt plus the
 * reply to the memories route on every call: history re-ingested each turn and
 * never recorded as turns.
 */
export async function recordTurn(params: RecordTurnParams): Promise<boolean> {
  const { credentials, modelOptions, role, content, baseUrl = DEFAULT_BASE_URL } = params;

  if (modelOptions.writeMemory === false) return false;
  if (!modelOptions.userId && !modelOptions.conversationId && !modelOptions.customerId) return false;
  if (!content || !content.trim()) return false;
  // The compaction block this package spliced into the prompt is not a turn.
  if (isSynapHistory(content)) return false;

  const grpc = await params.getStream();
  if (grpc) {
    const sent = await grpc.sendConversationEvent({
      conversation_id: modelOptions.conversationId ?? '',
      user_id: modelOptions.userId ?? '',
      customer_id: modelOptions.customerId ?? '',
      session_id: '',
      metadata: modelOptions.agentModel ? { agent_model: modelOptions.agentModel } : {},
      search_queries: [],
      context_types: [],
      tool_name: '',
      tool_args_json: '',
      event_type: role === 'user' ? 'user_message' : 'assistant_message',
      role,
      content,
      timestamp_ms: Date.now(),
    });
    // false: the stream dropped between the check and the write.
    if (sent !== false) return true;
  }

  // The REST route files a turn under a conversation; the stream would have minted one.
  if (!modelOptions.conversationId) return false;

  const correlationId = newCorrelationId();
  const body: Record<string, unknown> = {
    conversation_id: modelOptions.conversationId,
    role,
    content,
    user_id: modelOptions.userId ?? '',
    metadata: {
      source: 'vercel_ai_sdk',
      ...(modelOptions.agentModel ? { agent_model: modelOptions.agentModel } : {}),
    },
  };
  // Omitted rather than sent empty: a B2C instance refuses a call that carries
  // a customer_id at all.
  if (modelOptions.customerId) body['customer_id'] = modelOptions.customerId;

  try {
    const response = await (params.fetchImpl ?? fetch)(`${baseUrl}/v1/conversations/messages`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${credentials.api_key}`,
        'X-Client-ID': credentials.client_id,
        'X-Instance-ID': credentials.instance_id,
        'X-Correlation-ID': correlationId,
        'Content-Type': 'application/json',
        'User-Agent': `@maximem/synap-vercel-adk/${SDK_VERSION}`,
      },
      body: JSON.stringify(body),
    });
    // Checked: `fetch` does not throw on a 404, and not looking is how a write
    // to a wrong route went unnoticed for as long as it did.
    if (!response.ok) {
      console.warn(
        `[synap] turn write failed: ${response.status} ${response.statusText} `
        + `(correlation ${correlationId})`,
      );
      return false;
    }
    return true;
  } catch (err: unknown) {
    console.warn('[synap] turn write failed:', err);
    return false;
  }
}
