import type { Credentials, SynapModelOptions } from '../types.js';
import { newCorrelationId } from '../util/correlation.js';

const DEFAULT_BASE_URL = 'https://synap-cloud-prod.maximem.ai';

export interface MemoryWriteParams {
  credentials: Credentials;
  modelOptions: SynapModelOptions;
  messages: Array<{ role: string; content: string }>;
  assistantResponse: string;
  baseUrl?: string;
}

/**
 * Fire-and-forget memory write — posts the completed conversation turn back to
 * Synap so the server can update context for future requests.
 *
 * Mirrors: Python SDK sdk.conversation.add_memory() / memories.create()
 */
export async function writeMemory(params: MemoryWriteParams): Promise<void> {
  const { credentials, modelOptions, messages, assistantResponse, baseUrl = DEFAULT_BASE_URL } = params;

  if (modelOptions.writeMemory === false) return;
  if (!modelOptions.userId && !modelOptions.conversationId && !modelOptions.customerId) return;

  const turn = [
    ...messages,
    { role: 'assistant', content: assistantResponse },
  ];

  const correlationId = newCorrelationId();

  // ⚠ This posted to `/v1/memories/ingest`, which is not a route. Memories are
  // served under `/api/v1/memories`, the prefix split is deployed routing
  // rather than a typo, and `fetch` does not throw on a 404 — so every memory
  // write from this middleware 404'd and the result was never looked at. The
  // gRPC events worked, which is why it went unnoticed: context kept flowing
  // and nothing was ever written.
  const body: Record<string, unknown> = {
    document: turn.map((m) => `${m.role}: ${m.content}`).join('\n'),
    document_type: 'ai-chat-conversation',
    user_id: modelOptions.userId ?? '',
    metadata: {
      source: 'vercel_ai_sdk',
      conversation_id: modelOptions.conversationId ?? '',
    },
  };
  // Omitted rather than sent empty: a B2C instance refuses a call that carries
  // a customer_id at all.
  if (modelOptions.customerId) body['customer_id'] = modelOptions.customerId;

  try {
    const response = await fetch(`${baseUrl}/api/v1/memories/create`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${credentials.api_key}`,
        'X-Client-ID': credentials.client_id,
        'X-Instance-ID': credentials.instance_id,
        'X-Correlation-ID': correlationId,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    // Checked, because not checking is what hid this for as long as it hid.
    // Still non-fatal: a failed write must not break the caller's response.
    if (!response.ok) {
      console.warn(
        `[synap] memory write failed: ${response.status} ${response.statusText} `
        + `(correlation ${correlationId})`,
      );
    }
  } catch (err: unknown) {
    console.warn('[synap] memory write failed:', err);
  }
}
