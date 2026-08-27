/**
 * Instance namespace. Mirrors `InstanceInterface` in the Python SDK.
 *
 * Everything here rides the bidirectional gRPC Listen stream. The stream module
 * is loaded through a lazy `import()` so that importing this SDK on Edge or in
 * a Worker stays safe: grpc-js is built on `node:http2`/`node:net`/`node:tls`
 * and a static import fails the BUILD there, not the call.
 */

import {
  ListeningAlreadyActiveError, ListeningNotActiveError, SynapError,
} from '../errors.js';
import type { AnticipationCache } from '../context/anticipation-cache.js';
import type { StreamCredentials, StreamState } from '../grpc/stream-client.js';
import { getEnv } from '../util/env.js';

export interface ListenOptions {
  /** Called with the attempt count each time the stream reconnects. */
  on_reconnect?: (attempt: number) => void;
  onReconnect?: (attempt: number) => void;
  /** Called with a reason string when the stream drops. */
  on_disconnect?: (reason: string) => void;
  onDisconnect?: (reason: string) => void;
  /**
   * Called for every bundle that arrives. Bundles are ALSO stored in the
   * anticipation cache automatically, so `fetch()` finds them without a round
   * trip; this callback is for observing, not for wiring up storage.
   */
  on_context?: (bundle: Record<string, unknown>) => void;
  onContext?: (bundle: Record<string, unknown>) => void;
  /** Overrides `SYNAP_GRPC_HOST`. */
  host?: string;
  /** Overrides `SYNAP_GRPC_PORT`. */
  port?: number;
  /** Overrides `SYNAP_GRPC_USE_TLS`. Plaintext only makes sense against localhost. */
  use_tls?: boolean;
  useTls?: boolean;
}

export interface SendMessageOptions {
  content: string;
  role?: string;
  conversation_id?: string;
  conversationId?: string;
  user_id?: string;
  userId?: string;
  customer_id?: string;
  customerId?: string;
  session_id?: string;
  sessionId?: string;
  event_type?: string;
  eventType?: string;
  metadata?: Record<string, string>;
  /**
   * Report a tool call on the stream. Python's `send_message` takes both, and
   * the proto has `tool_name` / `tool_args_json`; the JS surface simply never
   * exposed them, so a tool call could not be reported from JavaScript.
   */
  tool_name?: string;
  toolName?: string;
  tool_args?: Record<string, unknown>;
  toolArgs?: Record<string, unknown>;
  search_queries?: string[];
  searchQueries?: string[];
  context_types?: string[];
  contextTypes?: string[];
}

export interface RecordThinkingOptions {
  content: string;
  /**
   * Ordinal of this thought within the current turn. Sent as a `step_index`
   * metadata key, which is how Python's `record_thinking` transmits it: the
   * proto has no dedicated field.
   */
  step_index?: number;
  stepIndex?: number;
  /** Free-form label for the kind of reasoning. Sent as `thought_type`. */
  thought_type?: string;
  thoughtType?: string;
  conversation_id?: string;
  conversationId?: string;
  user_id?: string;
  userId?: string;
  customer_id?: string;
  customerId?: string;
  session_id?: string;
  sessionId?: string;
  metadata?: Record<string, string>;
}

export interface InstanceNamespace {
  listen(options?: ListenOptions): Promise<void>;
  stop_listening(): Promise<void>;
  send_message(options: SendMessageOptions): Promise<void>;
  record_thinking(options: RecordThinkingOptions): Promise<void>;
  /** Property, not a method, matching Python's InstanceInterface.is_listening. */
  readonly is_listening: boolean;
  /** Not in Python: lets the compaction subscriptions share this one stream. */
  readonly _internal: InstanceInternals;
}

/** @internal Shared with the conversation namespace so both use one stream. */
export interface InstanceInternals {
  onCompactionUpdate(conversationId: string, bundle: Record<string, unknown>): void;
  /** Set by the client so a compaction can prune the locally buffered tail. */
  setCompactionApplier(apply: (bundle: Record<string, unknown>) => void): void;
  /** Learning-loop event, fired after a fetch is served from the cache. */
  sendContextUsed(event: Record<string, unknown>): void;
  /** Audit event, fired after every fetch resolves. */
  sendContextAssembled(event: Record<string, unknown>): void;
  setCompactionDispatcher(
    dispatch: (conversationId: string, bundle: Record<string, unknown>) => void,
  ): void;
  isListening(): boolean;
}

type StreamClient = import('../grpc/stream-client.js').GrpcStreamClient;

const DEFAULT_GRPC_HOST = 'synap-cloud-prod.maximem.ai';
const DEFAULT_GRPC_PORT = 443;

function envPort(): number | undefined {
  const raw = getEnv('SYNAP_GRPC_PORT');
  if (raw === undefined || raw === '') return undefined;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function envTls(): boolean | undefined {
  // SYNAP_GRPC_TLS is the 0.3.x wrapper's spelling. Honoured as a fallback,
  // because silently ignoring it turns a working plaintext deployment into one
  // that attempts TLS and cannot connect, with nothing pointing at the cause.
  const raw = getEnv('SYNAP_GRPC_USE_TLS') ?? getEnv('SYNAP_GRPC_TLS');
  if (raw === undefined || raw === '') return undefined;
  return !['0', 'false', 'no', 'off'].includes(raw.trim().toLowerCase());
}

/**
 * Fold `step_index` and `thought_type` into the metadata map.
 *
 * Python does exactly this (`sdk.py`, record_thinking): the proto carries no
 * dedicated fields, so the values ride in metadata under those two keys. A
 * caller-supplied metadata entry of the same name is not overwritten.
 */
function thinkingMetadata(options: RecordThinkingOptions): Record<string, string> {
  const md: Record<string, string> = { ...(options.metadata ?? {}) };
  const step = options.step_index ?? options.stepIndex;
  if (step !== undefined && md['step_index'] === undefined) {
    md['step_index'] = String(step);
  }
  const kind = options.thought_type ?? options.thoughtType;
  if (kind && md['thought_type'] === undefined) {
    md['thought_type'] = kind;
  }
  return md;
}

export function createInstanceNamespace(
  credentials: () => StreamCredentials,
  cache: AnticipationCache,
  /** Buffers a stream-sent turn locally, same as the HTTP write paths. */
  onTurn?: (turn: { conversationId: string; role: string; content: string }) => void,
): InstanceNamespace {
  let client: StreamClient | null = null;
  let dispatchCompaction:
    | ((conversationId: string, bundle: Record<string, unknown>) => void)
    | null = null;

  let applyCompaction: ((bundle: Record<string, unknown>) => void) | null = null;

  const internals: InstanceInternals = {
    onCompactionUpdate(conversationId, bundle) {
      // Prune the local tail FIRST, then notify subscribers, so a subscriber
      // that immediately fetches sees the post-compaction view.
      applyCompaction?.(bundle);
      dispatchCompaction?.(conversationId, bundle);
    },
    setCompactionApplier(apply) {
      applyCompaction = apply;
    },
    sendContextUsed(event) {
      // Fire-and-forget: the stream drops it silently when disconnected, which
      // is correct for telemetry that must never affect the caller's turn.
      client?.sendContextUsed(event);
    },
    sendContextAssembled(event) {
      client?.sendContextAssembled(event);
    },
    setCompactionDispatcher(dispatch) {
      dispatchCompaction = dispatch;
    },
    isListening() {
      return client !== null && client.isConnected;
    },
  };

  const namespace: InstanceNamespace = {
    async listen(options: ListenOptions = {}) {
      if (client !== null) {
        // Python raises this from InstanceController when a second listen()
        // lands on an already-active stream.
        throw new ListeningAlreadyActiveError('Listening is already active on this instance');
      }

      const { GrpcStreamClient } = await import('../grpc/stream-client.js');
      const onContext = options.on_context ?? options.onContext;
      const onDisconnect = options.on_disconnect ?? options.onDisconnect;
      const onReconnect = options.on_reconnect ?? options.onReconnect;
      let reconnectAttempts = 0;

      const created = new GrpcStreamClient(credentials(), cache, {
        host: options.host ?? getEnv('SYNAP_GRPC_HOST') ?? DEFAULT_GRPC_HOST,
        port: options.port ?? envPort() ?? DEFAULT_GRPC_PORT,
        useTls: options.use_tls ?? options.useTls ?? envTls() ?? true,
        ...(onContext !== undefined ? { onContext } : {}),
        onCompactionUpdate: (conversationId, bundle) => {
          internals.onCompactionUpdate(conversationId, bundle);
        },
        onStateChange: (state: StreamState) => {
          if (state === 'reconnecting') {
            reconnectAttempts += 1;
            onReconnect?.(reconnectAttempts);
          } else if (state === 'disconnected') {
            onDisconnect?.('stream disconnected');
          }
        },
      });

      try {
        await created.connect();
      } catch (error) {
        // Leave `client` null so a retry is possible rather than being
        // rejected as "already active" forever.
        if (error instanceof SynapError) throw error;
        throw new SynapError(
          `Could not open the Synap anticipation stream: ${(error as Error)?.message ?? String(error)}`,
          { cause: error },
        );
      }
      client = created;
    },

    async stop_listening() {
      // Resolves rather than rejecting: stopping something that was never
      // started is a no-op in Python too, and a cleanup path that throws is
      // worse than useless in a finally block.
      if (client === null) return;
      const active = client;
      client = null;
      await active.disconnect();
    },

    async send_message(options: SendMessageOptions) {
      if (client === null) throw new ListeningNotActiveError('Listening is not active');
      if (options.content === undefined) {
        throw new ListeningNotActiveError('content is required');
      }
      const conversationId = options.conversation_id ?? options.conversationId ?? '';
      onTurn?.({
        conversationId,
        role: options.role ?? 'user',
        content: options.content,
      });
      client.sendConversationEvent({
        event_type: options.event_type ?? options.eventType ?? 'user_message',
        content: options.content,
        role: options.role ?? 'user',
        conversation_id: options.conversation_id ?? options.conversationId ?? '',
        user_id: options.user_id ?? options.userId ?? '',
        customer_id: options.customer_id ?? options.customerId ?? '',
        session_id: options.session_id ?? options.sessionId ?? '',
        metadata: options.metadata ?? {},
        timestamp_ms: Date.now(),
        tool_name: options.tool_name ?? options.toolName ?? '',
        // Python serialises tool_args with json.dumps into tool_args_json.
        tool_args_json: (() => {
          const a = options.tool_args ?? options.toolArgs;
          return a === undefined ? '' : JSON.stringify(a);
        })(),
        search_queries: options.search_queries ?? options.searchQueries ?? [],
        context_types: options.context_types ?? options.contextTypes ?? [],
      });
    },

    async record_thinking(options: RecordThinkingOptions) {
      if (client === null) throw new ListeningNotActiveError('Listening is not active');
      client.sendConversationEvent({
        // Python emits reasoning as an `agent_thinking` conversation event.
        event_type: 'agent_thinking',
        content: options.content,
        role: 'assistant',
        conversation_id: options.conversation_id ?? options.conversationId ?? '',
        user_id: options.user_id ?? options.userId ?? '',
        customer_id: options.customer_id ?? options.customerId ?? '',
        session_id: options.session_id ?? options.sessionId ?? '',
        metadata: thinkingMetadata(options),
        timestamp_ms: Date.now(),
      });
    },

    get is_listening() {
      return client !== null && client.isConnected;
    },

    _internal: internals,
  };

  return namespace;
}
