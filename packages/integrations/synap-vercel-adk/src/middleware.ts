import type { LanguageModelV1Middleware, LanguageModelV1StreamPart } from 'ai';
import type { LanguageModelV1Prompt } from '@ai-sdk/provider';
import type { SynapModelOptions, FetchedContext } from './types.js';
import type { AnticipationCache } from './context/anticipation-cache.js';
import type { Credentials } from './types.js';
import { fetchContext } from './context/http-fetcher.js';
import { injectContextIntoPrompt, extractSearchQuery } from './transform/messages.js';
import { recordTurn } from './memory/turns.js';
import { fetchCompactedHistory, spliceHistory, messageText } from './history.js';
import type { GrpcStreamClient } from './grpc/stream-client.js';
import type { StreamManager } from './stream.js';
import { SDK_VERSION } from './version.js';
import { newCorrelationId } from './util/correlation.js';

export interface SynapMiddlewareOptions extends SynapModelOptions {
  credentials: Credentials;
  anticipationCache: AnticipationCache;
  /**
   * Opens the gRPC stream on first use and hands it back on every later call.
   * The provider passes this; it wins over `grpcClient`.
   */
  streams?: StreamManager;
  /** A stream the caller manages itself. Used only when `streams` is not set. */
  grpcClient?: GrpcStreamClient | null;
  baseUrl?: string;
}

/**
 * Vercel AI SDK middleware that wraps any LanguageModelV1 with Synap context.
 *
 * Flow per call:
 *   transformParams:  open the stream (first call) + fetch context + fetch the
 *                     compacted history in parallel; inject the context block;
 *                     replace earlier prompt turns with Synap's compaction
 *   wrapGenerate:     record the new user turn, call the model, record the reply
 *   wrapStream:       same, with the reply accumulated from the stream
 *
 * Turns go on the gRPC stream when it is open, else over REST; never both.
 */
export function createSynapMiddleware(opts: SynapMiddlewareOptions): LanguageModelV1Middleware {
  // The prompt as the caller sent it, keyed by the params object transformParams
  // returns (which the AI SDK hands to wrapGenerate/wrapStream unchanged), so
  // concurrent calls cannot swap each other's turns.
  const originals = new WeakMap<object, LanguageModelV1Prompt>();
  // One turn can take several model calls (tool steps) that all end in the same
  // user message; it is recorded once. Tool results likewise.
  let lastUserKey = '';
  const reportedToolResults = new Set<string>();

  const stream = () => getStream(opts);
  const originalPromptOf = (params: { prompt?: LanguageModelV1Prompt } | undefined): LanguageModelV1Prompt =>
    (params && originals.get(params)) || params?.prompt || [];

  /** The user's new message, and any tool results the prompt now carries. */
  const recordIncoming = (prompt: LanguageModelV1Prompt): void => {
    const idx = lastUserIndex(prompt);
    if (idx < 0) return;
    const text = messageText(prompt[idx]!);
    const key = `${idx}:${text}`;
    if (text.trim() && key !== lastUserKey) {
      lastUserKey = key;
      recordTurn({
        credentials: opts.credentials, modelOptions: opts, role: 'user',
        content: text, getStream: stream, baseUrl: opts.baseUrl,
      }).catch((err: unknown) => console.warn('[synap] user turn write failed:', err));
    }
    for (const msg of prompt.slice(idx + 1)) {
      if (msg.role !== 'tool') continue;
      for (const part of msg.content) {
        if (part.type !== 'tool-result' || reportedToolResults.has(part.toolCallId)) continue;
        reportedToolResults.add(part.toolCallId);
        emitToolResult(opts, part)
          .catch((err: unknown) => console.warn('[synap] tool result event failed:', err));
      }
    }
  };

  const recordReply = (text: string): void => {
    recordTurn({
      credentials: opts.credentials, modelOptions: opts, role: 'assistant',
      content: text, getStream: stream, baseUrl: opts.baseUrl,
    }).catch((err: unknown) => console.warn('[synap] assistant turn write failed:', err));
  };

  return {
    // ── Step 1: open the stream, fetch context + compaction, rewrite the prompt ──
    transformParams: async ({ params }) => {
      // Started first and never awaited here: the first call opens the stream
      // alongside the fetches, and each emitter awaits the same open.
      void stream();
      const original = params.prompt;
      const done = (next: typeof params) => { originals.set(next, original); return next; };

      if (opts.injectContext === false) return done(params);
      if (!opts.userId && !opts.conversationId && !opts.customerId) return done(params);

      const startedAt = Date.now();
      const [ctx, history] = await Promise.all([
        resolveContext(opts, original),
        opts.conversationId
          ? fetchCompactedHistory({
              credentials: opts.credentials, conversationId: opts.conversationId, baseUrl: opts.baseUrl,
            })
          : Promise.resolve(null),
      ]);

      let prompt = original;
      if (ctx) {
        // Fire-and-forget learning-loop telemetry (ids only, never prompt content).
        emitContextTelemetry(opts, ctx, Date.now() - startedAt)
          .catch((err: unknown) => console.warn('[synap] context telemetry failed:', err));
        // The caller sends the whole history, so the recent turns in the block
        // would be a second copy; the summary is Synap's compaction and is
        // already in the spliced preamble when there is one.
        prompt = injectContextIntoPrompt(prompt, ctx, {
          includeRecentTurns: false,
          includeSummary: !history,
        });
      }
      // Once Synap has compacted the conversation the model sees that instead
      // of the caller's earlier turns. None yet: the prompt is left alone.
      if (history) prompt = spliceHistory(prompt, history);

      return done(prompt === original ? params : { ...params, prompt });
    },

    // ── Step 2a: non-streaming: record the turn around the call ──────────
    wrapGenerate: async ({ doGenerate, params }) => {
      recordIncoming(originalPromptOf(params));
      const result = await doGenerate();

      for (const call of result.toolCalls ?? []) {
        emitToolCall(opts, call)
          .catch((err: unknown) => console.warn('[synap] tool call event failed:', err));
      }
      if (result.reasoning) {
        const reasoningText = typeof result.reasoning === 'string'
          ? result.reasoning
          : (result.reasoning as Array<{ text?: string }>)
            .map((r) => r.text ?? '').join('');
        if (reasoningText) {
          emitReasoning(opts, reasoningText)
            .catch((err: unknown) => console.warn('[synap] reasoning event failed:', err));
        }
      }
      if (result.text) recordReply(result.text);

      return result;
    },

    // ── Step 2b: streaming: accumulate, record the reply on stream end ───
    wrapStream: async ({ doStream, params }) => {
      recordIncoming(originalPromptOf(params));
      const { stream: inner, ...rest } = await doStream();

      let accumulated = '';
      // Reasoning arrives in deltas like text does. Accumulated and sent once
      // at the end rather than per delta: a thought is one step, and a stream
      // event per token would be noise.
      let reasoning = '';

      const wrappedStream = new ReadableStream<LanguageModelV1StreamPart>({
        start(controller) {
          const reader = inner.getReader();

          function pump(): void {
            reader.read().then(({ done, value }) => {
              if (done) {
                if (reasoning) {
                  emitReasoning(opts, reasoning)
                    .catch((err: unknown) => console.warn('[synap] reasoning event failed:', err));
                }
                if (accumulated) recordReply(accumulated);
                controller.close();
                return;
              }

              // Only text-delta was read here, so a tool call and any
              // reasoning the provider returned passed straight through and
              // Synap never saw them: anticipation got the answer and none of
              // the work that produced it.
              if (value.type === 'text-delta') {
                // A string check, not a cast: a v2 model's parts carry `delta`,
                // and `+= undefined` would record the word "undefined" as the reply.
                if (typeof value.textDelta === 'string') accumulated += value.textDelta;
              } else if (value.type === 'tool-call') {
                emitToolCall(opts, value)
                  .catch((err: unknown) => console.warn('[synap] tool call event failed:', err));
              } else if (value.type === 'reasoning') {
                reasoning += value.textDelta ?? '';
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

/** The stream, opened on first use; null when this call must use REST. Never throws. */
async function getStream(opts: SynapMiddlewareOptions): Promise<GrpcStreamClient | null> {
  if (opts.streams) return opts.streams.ensure();
  return opts.grpcClient?.isConnected ? opts.grpcClient : null;
}

function lastUserIndex(prompt: LanguageModelV1Prompt): number {
  for (let i = prompt.length - 1; i >= 0; i--) if (prompt[i]!.role === 'user') return i;
  return -1;
}

function resolveScope(opts: SynapModelOptions): string {
  if (opts.conversationId) return 'conversation';
  if (opts.userId) return 'user';
  if (opts.customerId) return 'customer';
  return 'client';
}

/**
 * Emit the learning-loop telemetry events after a context fetch resolves:
 *   - ContextAssembledEvent on every fetch (any source)
 *   - ContextUsedEvent only when served from the anticipation cache
 * Both are fire-and-forget and carry ids/metadata only — never prompt content.
 */
async function emitContextTelemetry(
  opts: SynapMiddlewareOptions,
  ctx: FetchedContext,
  durationMs: number,
): Promise<void> {
  const grpc = await getStream(opts);
  if (!grpc) return;

  const now = Date.now();
  const conversationId = opts.conversationId ?? '';
  const userId = opts.userId ?? '';
  const customerId = opts.customerId ?? '';

  await grpc.sendContextAssembledEvent({
    correlation_id: ctx.correlationId || newCorrelationId(),
    conversation_id: conversationId,
    user_id: userId,
    customer_id: customerId,
    final_item_ids: ctx.servedItemIds,
    final_total_tokens: ctx.totalTokens,
    compaction_id: ctx.conversationContext?.compactionId ?? '',
    recent_turn_count: ctx.conversationContext?.recentTurns.length ?? 0,
    compaction_end_timestamp: '',
    assembly_source: ctx.assemblySource,
    assembly_duration_ms: durationMs,
    cache_hit: ctx.cacheHit,
    timestamp_ms: now,
    sdk_version: SDK_VERSION,
  });

  if (ctx.source === 'anticipation') {
    await grpc.sendContextUsedEvent({
      bundle_id: ctx.bundleId,
      conversation_id: conversationId,
      user_id: userId,
      customer_id: customerId,
      served_item_ids: ctx.servedItemIds,
      timestamp_ms: now,
      scope: resolveScope(opts),
      source_bundle_ids: ctx.sourceBundleIds,
    });
  }
}

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

/**
 * A tool the model asked for.
 *
 * `role: 'assistant'` and `event_type: 'tool_call'` together: the server reads
 * the type first, and the pair has to agree or the call is filed as an
 * ordinary assistant reply.
 */
async function emitToolCall(
  opts: SynapMiddlewareOptions,
  call: { toolName?: string; toolCallId?: string; args?: unknown },
): Promise<void> {
  const grpcClient = await getStream(opts);
  if (!grpcClient) return;
  await grpcClient.sendConversationEvent({
    ...baseEvent(opts),
    event_type: 'tool_call',
    role: 'assistant',
    content: '',
    tool_name: call.toolName ?? '',
    tool_call_id: call.toolCallId ?? '',
    tool_args_json: call.args === undefined ? '' : JSON.stringify(call.args),
    timestamp_ms: Date.now(),
  });
}

/**
 * What a tool returned, read from the next prompt (the AI SDK runs tools
 * outside the model, so a result first reaches the middleware there).
 *
 * `role: 'tool'` and `event_type: 'tool_result'`, with the result in its own
 * field: it is not something a person said. A string is sent as itself.
 */
async function emitToolResult(
  opts: SynapMiddlewareOptions,
  part: { toolCallId: string; toolName?: string; result: unknown },
): Promise<void> {
  const grpcClient = await getStream(opts);
  if (!grpcClient) return;
  await grpcClient.sendConversationEvent({
    ...baseEvent(opts),
    event_type: 'tool_result',
    role: 'tool',
    content: '',
    tool_name: part.toolName ?? '',
    tool_call_id: part.toolCallId,
    tool_result_json: typeof part.result === 'string' ? part.result : JSON.stringify(part.result ?? null),
    timestamp_ms: Date.now(),
  });
}

/**
 * What the model was thinking, when the provider returned any.
 *
 * No role. A thought is not the assistant's reply, and a role of 'assistant'
 * is what made the server file every reasoning step as the final answer.
 */
async function emitReasoning(
  opts: SynapMiddlewareOptions,
  content: string,
): Promise<void> {
  if (!content) return;
  const grpcClient = await getStream(opts);
  if (!grpcClient) return;
  await grpcClient.sendConversationEvent({
    ...baseEvent(opts),
    event_type: 'agent_thinking',
    role: '',
    content,
    metadata: { thought_type: 'model_reasoning' },
    timestamp_ms: Date.now(),
  });
}

function baseEvent(opts: SynapModelOptions) {
  return {
    conversation_id: opts.conversationId ?? '',
    user_id: opts.userId ?? '',
    customer_id: opts.customerId ?? '',
    session_id: '',
    metadata: modelMetadata(opts),
    search_queries: [] as string[],
    context_types: [] as string[],
  };
}

/** `{ agent_model }` for the server's compaction sizing, or nothing. */
function modelMetadata(opts: SynapModelOptions): Record<string, string> {
  return opts.agentModel ? { agent_model: opts.agentModel } : {};
}
