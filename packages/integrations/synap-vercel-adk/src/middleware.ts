import type { LanguageModelV1Middleware, LanguageModelV1StreamPart } from 'ai';
import type { LanguageModelV1Prompt } from '@ai-sdk/provider';
import type { SynapModelOptions, FetchedContext } from './types.js';
import type { AnticipationCache } from './context/anticipation-cache.js';
import type { Credentials } from './types.js';
import { fetchContext } from './context/http-fetcher.js';
import { injectContextIntoPrompt, extractSearchQuery, promptToTranscript } from './transform/messages.js';
import { writeMemory } from './memory/writer.js';
import type { GrpcStreamClient } from './grpc/stream-client.js';

export interface SynapMiddlewareOptions extends SynapModelOptions {
  credentials: Credentials;
  anticipationCache: AnticipationCache;
  grpcClient: GrpcStreamClient | null;
  baseUrl?: string;
}

/**
 * Vercel AI SDK middleware that wraps any LanguageModelV1 with Synap context.
 *
 * Uses a per-middleware prompt stack to pass the original prompt from
 * transformParams → wrapGenerate/wrapStream for memory writing.
 *
 * Flow per call:
 *   transformParams:  fetch context → inject into prompt
 *   wrapGenerate:     call underlying model → write memory + emit gRPC event
 *   wrapStream:       stream underlying model → accumulate → write memory
 */
export function createSynapMiddleware(opts: SynapMiddlewareOptions): LanguageModelV1Middleware {
  // Stack captures the original (pre-injection) prompt so wrapGenerate/wrapStream
  // can write the correct messages to memory. Safe for sequential calls;
  // for concurrent calls, worst case is a swapped memory write (non-fatal, fire-and-forget).
  const promptStack: LanguageModelV1Prompt[] = [];

  return {
    // ── Step 1: inject context before the underlying model is called ──────
    transformParams: async ({ params }) => {
      promptStack.push(params.prompt);

      if (opts.injectContext === false) return params;
      if (!opts.userId && !opts.conversationId && !opts.customerId) return params;

      const ctx = await resolveContext(opts, params.prompt);
      if (!ctx) return params;

      return { ...params, prompt: injectContextIntoPrompt(params.prompt, ctx) };
    },

    // ── Step 2a: non-streaming — write memory after response ─────────────
    wrapGenerate: async ({ doGenerate }) => {
      const originalPrompt = promptStack.pop();
      const result = await doGenerate();

      if (result.text && originalPrompt) {
        const messages = promptToTranscript(originalPrompt);
        writeMemory({
          credentials: opts.credentials,
          modelOptions: opts,
          messages,
          assistantResponse: result.text,
          baseUrl: opts.baseUrl,
        }).catch((err: unknown) => console.warn('[synap] writeMemory failed:', err));
        emitConversationEvents(opts.grpcClient, opts, messages, result.text)
          .catch((err: unknown) => console.warn('[synap] gRPC conversation event failed:', err));
      }

      return result;
    },

    // ── Step 2b: streaming — accumulate then write memory on stream end ───
    wrapStream: async ({ doStream }) => {
      const originalPrompt = promptStack.pop();
      const { stream, ...rest } = await doStream();

      let accumulated = '';

      const wrappedStream = new ReadableStream<LanguageModelV1StreamPart>({
        start(controller) {
          const reader = stream.getReader();

          function pump(): void {
            reader.read().then(({ done, value }) => {
              if (done) {
                if (accumulated && originalPrompt) {
                  const messages = promptToTranscript(originalPrompt);
                  writeMemory({
                    credentials: opts.credentials,
                    modelOptions: opts,
                    messages,
                    assistantResponse: accumulated,
                    baseUrl: opts.baseUrl,
                  }).catch((err: unknown) => console.warn('[synap] writeMemory failed:', err));
                  emitConversationEvents(opts.grpcClient, opts, messages, accumulated)
                    .catch((err: unknown) => console.warn('[synap] gRPC conversation event failed:', err));
                }
                controller.close();
                return;
              }

              if (value.type === 'text-delta') {
                accumulated += value.textDelta;
              }
              controller.enqueue(value);
              pump();
            }).catch((err: unknown) => controller.error(err));
          }

          pump();
        },
      });

      return { stream: wrappedStream, ...rest };
    },
  };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

async function resolveContext(
  opts: SynapMiddlewareOptions,
  prompt: LanguageModelV1Prompt,
): Promise<FetchedContext | null> {
  const searchQuery = extractSearchQuery(prompt);

  // Anticipation cache first (gRPC-populated, zero HTTP latency)
  const cached = opts.anticipationCache.lookup({
    userId: opts.userId,
    customerId: opts.customerId,
    conversationId: opts.conversationId,
    searchQuery,
  });
  if (cached) return cached;

  // HTTP fallback — must not throw, context failure must not block LLM call
  try {
    return await fetchContext({
      credentials: opts.credentials,
      modelOptions: opts,
      searchQuery,
      baseUrl: opts.baseUrl,
    });
  } catch (err) {
    console.warn('[synap] context fetch failed — proceeding without context:', err);
    return null;
  }
}

async function emitConversationEvents(
  grpcClient: GrpcStreamClient | null,
  opts: SynapModelOptions,
  messages: Array<{ role: string; content: string }>,
  assistantText: string,
): Promise<void> {
  if (!grpcClient?.isConnected) return;

  const now = Date.now();
  const base = {
    conversation_id: opts.conversationId ?? '',
    user_id: opts.userId ?? '',
    customer_id: opts.customerId ?? '',
    session_id: '',
    metadata: {} as Record<string, string>,
    search_queries: [] as string[],
    context_types: [] as string[],
    tool_name: '',
    tool_args_json: '',
  };

  const lastUser = [...messages].reverse().find(m => m.role === 'user');
  if (lastUser) {
    await grpcClient.sendConversationEvent({
      ...base,
      event_type: 'user_message',
      role: 'user',
      content: lastUser.content,
      timestamp_ms: now - 1,
    });
  }

  await grpcClient.sendConversationEvent({
    ...base,
    event_type: 'assistant_message',
    role: 'assistant',
    content: assistantText,
    timestamp_ms: now,
  });
}
