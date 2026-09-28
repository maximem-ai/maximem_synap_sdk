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
import { newCorrelationId } from '../util/correlation.js';

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
  /**
   * What a tool returned. Travels in its own proto field, not in `content`: a
   * result is not something a person said, and everything that filters on
   * messages reads `content`.
   */
  tool_result?: unknown;
  toolResult?: unknown;
  /** Ties a result to the call that asked for it. */
  tool_call_id?: string;
  toolCallId?: string;
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

export interface RecordToolCallOptions {
  tool_name?: string;
  toolName?: string;
  tool_args?: Record<string, unknown>;
  toolArgs?: Record<string, unknown>;
  tool_call_id?: string;
  toolCallId?: string;
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

export interface RecordToolResultOptions {
  /** JSON-encodable, or a plain string. A string is sent as itself. */
  result: unknown;
  tool_name?: string;
  toolName?: string;
  tool_call_id?: string;
  toolCallId?: string;
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
  record_tool_call(options: RecordToolCallOptions): Promise<void>;
  record_tool_result(options: RecordToolResultOptions): Promise<void>;
  end_session(conversationId: string): Promise<void>;
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

/**
 * Event types the server has a reader for. Mirrors `KNOWN_EVENT_TYPES` in the
 * Python SDK. An unknown type falls through every classifier on the server into
 * UNKNOWN, where nothing reads it, so a typo used to cost the whole signal
 * without saying so.
 */
const KNOWN_EVENT_TYPES: ReadonlySet<string> = new Set([
  'user_message',
  'assistant_message',
  'tool_call',
  'tool_result',
  'agent_thinking',
  'context_request',
  'context_fetch',
  'session_start',
  'session_end',
]);

/**
 * Serialise a tool result the way Python does: a string travels as itself,
 * anything else as JSON. `JSON.stringify` on a plain-text answer would wrap it
 * in quotes and the agent would read the quotes as part of the result.
 */
function toolResultJson(result: unknown): string {
  if (result === undefined || result === null) return '';
  return typeof result === 'string' ? result : jsonOrText(result);
}

/**
 * JSON for whatever serialises, the value's text form for whatever does not.
 * Mirrors Python's `_json_or_text` in `sdk.py`.
 *
 * ⚠ This was a bare `JSON.stringify` and that lost events. It throws on a
 * circular object and on a `BigInt`, and it returns `undefined` rather than a
 * string for a function or a bare `undefined`. Every integration reports
 * through a helper that swallows exceptions on purpose, so a telemetry call can
 * never break somebody's agent loop, which means the throw went nowhere and the
 * whole tool event was silently never sent. Losing the exact shape of an object
 * is a far smaller loss than losing the event.
 */
function jsonOrText(value: unknown): string {
  try {
    const out = JSON.stringify(
      value,
      (_k, v: unknown) => (typeof v === 'bigint' ? String(v) : v),
    );
    // `undefined` for a function, a symbol, or `undefined` itself.
    return out === undefined ? String(value) : out;
  } catch {
    // A cycle, or a `toJSON` of the caller's that throws.
    try {
      return String(value);
    } catch {
      return '';
    }
  }
}

export function createInstanceNamespace(
  credentials: () => StreamCredentials,
  cache: AnticipationCache,
  /** Buffers a stream-sent turn locally, same as the HTTP write paths. */
  onTurn?: (turn: { conversationId: string; role: string; content: string }) => void,
): InstanceNamespace {
  let client: StreamClient | null = null;
  // conversationId -> sessionId, for conversations this stream has opened a
  // session for. Cleared on reconnect: a session belongs to the stream that
  // carried it, and the server's new stream has never seen it.
  const openSessions = new Map<string, string>();
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

  /**
   * Open a session for this conversation if the stream has not yet.
   *
   * The developer never calls this. `session_start` is what makes the server
   * run its warm-up prefetch before the first message arrives, and asking every
   * caller to remember a lifecycle call is how the first turn of every
   * conversation stayed a cold fetch: neither SDK sent one, ever.
   *
   * Best effort. A session that cannot be opened must not stop the event that
   * triggered it, so the conversation stays unmarked and the next event retries.
   */
  function ensureSession(
    conversationId: string, userId: string, customerId: string,
  ): void {
    // Both ids, or no session. The server refuses a session_start with no
    // user_id, and writing one only says it went out, not that it was
    // accepted — so opening on an event with no user_id marked the
    // conversation as open, never retried, and left `end_session` referring to
    // a session the server never had. The next event that carries a user_id
    // opens it.
    if (!conversationId || !userId || client === null) return;
    if (openSessions.has(conversationId)) return;
    // The same helper the correlation ids use: a session id is an identifier
    // for tracing, never a security value, and that helper is the one that is
    // safe on Edge and Workers, where `node:crypto` breaks the build.
    const sessionId = newCorrelationId();
    try {
      client.sendSessionControl({
        action: 'start',
        session_id: sessionId,
        conversation_id: conversationId,
        ...(userId ? { user_id: userId } : {}),
        ...(customerId ? { customer_id: customerId } : {}),
      });
      openSessions.set(conversationId, sessionId);
    } catch {
      // Leave it unmarked so the next event opens it.
    }
  }

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

      // Assigned immediately below; the callbacks that read it only run later,
      // once the stream is live.
      let stream: InstanceType<typeof GrpcStreamClient> | null = null;

      const created = new GrpcStreamClient(credentials(), cache, {
        host: options.host ?? getEnv('SYNAP_GRPC_HOST') ?? DEFAULT_GRPC_HOST,
        port: options.port ?? envPort() ?? DEFAULT_GRPC_PORT,
        useTls: options.use_tls ?? options.useTls ?? envTls() ?? true,
        ...(onContext !== undefined ? { onContext } : {}),
        onCompactionUpdate: (conversationId, bundle) => {
          internals.onCompactionUpdate(conversationId, bundle);
        },
        onStateChange: (state: StreamState) => {
          // Any state change away from connected retires the sessions this
          // stream opened: the server's next stream has no memory of them, and
          // a client that believes one is open never warms up again.
          if (state !== 'connected') openSessions.clear();
          if (state === 'reconnecting') {
            reconnectAttempts += 1;
            onReconnect?.(reconnectAttempts);
          } else if (state === 'disconnected') {
            // A credit refusal names itself, the way Python's transport does,
            // so a handler can tell "the server ran out of patience" from
            // "this account cannot pay for the stream".
            const stopped = stream?.lastError ?? null;
            const reason = stopped === null
              ? 'stream disconnected'
              : `credit_stop:${(stopped as { reason?: string | null }).reason ?? 'unknown'}`;
            onDisconnect?.(reason);
          }
        },
      });
      stream = created;

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
      // Close what this stream opened. A session that is never closed leaves
      // the turn it was in the middle of uncommitted on the server: that row is
      // written when the NEXT user message arrives, and the last turn of a
      // conversation has no next one.
      for (const conversationId of [...openSessions.keys()]) {
        await namespace.end_session(conversationId);
      }
      const active = client;
      client = null;
      await active.disconnect();
    },

    async end_session(conversationId: string) {
      const sessionId = openSessions.get(conversationId);
      if (sessionId === undefined || client === null) return;
      openSessions.delete(conversationId);
      try {
        client.sendSessionControl({
          action: 'end',
          session_id: sessionId,
          conversation_id: conversationId,
        });
      } catch {
        // A teardown path must not throw.
      }
    },

    async send_message(options: SendMessageOptions) {
      if (client === null) throw new ListeningNotActiveError('Listening is not active');
      if (options.content === undefined) {
        throw new ListeningNotActiveError('content is required');
      }
      const eventType = options.event_type ?? options.eventType ?? 'user_message';
      if (!KNOWN_EVENT_TYPES.has(eventType)) {
        throw new SynapError(
          `unknown event_type '${eventType}'. Expected one of: `
          + [...KNOWN_EVENT_TYPES].sort().join(', '),
        );
      }
      const conversationId = options.conversation_id ?? options.conversationId ?? '';
      ensureSession(
        conversationId,
        options.user_id ?? options.userId ?? '',
        options.customer_id ?? options.customerId ?? '',
      );
      // Only conversational turns belong in the local short-term buffer. A tool
      // call or a result is not a turn, and buffering one splices it into the
      // caller's own prompt as if somebody had said it. Python mirrors user and
      // assistant messages only; this side mirrored everything.
      if (eventType === 'user_message' || eventType === 'assistant_message') {
        onTurn?.({
          conversationId,
          role: options.role ?? 'user',
          content: options.content,
        });
      }
      client.sendConversationEvent({
        event_type: eventType,
        content: options.content,
        role: options.role ?? 'user',
        conversation_id: options.conversation_id ?? options.conversationId ?? '',
        user_id: options.user_id ?? options.userId ?? '',
        customer_id: options.customer_id ?? options.customerId ?? '',
        session_id: options.session_id ?? options.sessionId
          ?? openSessions.get(conversationId) ?? '',
        metadata: options.metadata ?? {},
        timestamp_ms: Date.now(),
        tool_name: options.tool_name ?? options.toolName ?? '',
        // Python serialises tool_args with json.dumps into tool_args_json.
        tool_args_json: (() => {
          const a = options.tool_args ?? options.toolArgs;
          return a === undefined ? '' : jsonOrText(a);
        })(),
        tool_result_json: toolResultJson(options.tool_result ?? options.toolResult),
        tool_call_id: options.tool_call_id ?? options.toolCallId ?? '',
        search_queries: options.search_queries ?? options.searchQueries ?? [],
        context_types: options.context_types ?? options.contextTypes ?? [],
      });
    },

    async record_tool_call(options: RecordToolCallOptions) {
      // Prefer this over `send_message({ event_type: 'tool_call' })`. The event
      // carries both an event_type and a role and the two have to agree;
      // getting the role wrong was silent, and our own published example got it
      // wrong. Here you cannot: the method sets both.
      await namespace.send_message({
        content: '',
        role: 'assistant',
        event_type: 'tool_call',
        tool_name: options.tool_name ?? options.toolName ?? '',
        ...(options.tool_args ?? options.toolArgs
          ? { tool_args: options.tool_args ?? options.toolArgs }
          : {}),
        tool_call_id: options.tool_call_id ?? options.toolCallId ?? '',
        conversation_id: options.conversation_id ?? options.conversationId ?? '',
        user_id: options.user_id ?? options.userId ?? '',
        customer_id: options.customer_id ?? options.customerId ?? '',
        session_id: options.session_id ?? options.sessionId ?? '',
        metadata: options.metadata ?? {},
      });
    },

    async record_tool_result(options: RecordToolResultOptions) {
      // ⚠ A tool result is usually the caller's customer data. It is an
      // anticipation hint and is not written into long-term memory, but it does
      // reach us: send what the agent needs, not the whole row.
      await namespace.send_message({
        content: '',
        role: 'tool',
        event_type: 'tool_result',
        tool_result: options.result,
        tool_name: options.tool_name ?? options.toolName ?? '',
        tool_call_id: options.tool_call_id ?? options.toolCallId ?? '',
        conversation_id: options.conversation_id ?? options.conversationId ?? '',
        user_id: options.user_id ?? options.userId ?? '',
        customer_id: options.customer_id ?? options.customerId ?? '',
        session_id: options.session_id ?? options.sessionId ?? '',
        metadata: options.metadata ?? {},
      });
    },

    async record_thinking(options: RecordThinkingOptions) {
      if (client === null) throw new ListeningNotActiveError('Listening is not active');
      const conversationId = options.conversation_id ?? options.conversationId ?? '';
      ensureSession(
        conversationId,
        options.user_id ?? options.userId ?? '',
        options.customer_id ?? options.customerId ?? '',
      );
      client.sendConversationEvent({
        // Python emits reasoning as an `agent_thinking` conversation event.
        event_type: 'agent_thinking',
        content: options.content,
        // Deliberately no role. A thought is not the assistant's reply, and
        // this went out as 'assistant' from the day it shipped: the server
        // checked role before event_type, so every reasoning step any client
        // sent was filed as the final answer and the reasoning path never ran.
        // Sending no role is correct against both the fixed server and one that
        // has not been updated yet.
        role: '',
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
