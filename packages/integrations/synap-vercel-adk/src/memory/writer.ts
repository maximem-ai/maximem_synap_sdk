import type { Credentials, SynapModelOptions } from '../types.js';

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

  const correlationId = crypto.randomUUID();

  const body: Record<string, unknown> = {
    messages: turn,
    user_id: modelOptions.userId ?? '',
    customer_id: modelOptions.customerId ?? '',
    conversation_id: modelOptions.conversationId ?? '',
    source: 'vercel_ai_sdk',
  };

  try {
    await fetch(`${baseUrl}/v1/memories/ingest`, {
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
  } catch {
    // Non-fatal — memory write failure should never break the LLM response
  }
}
