/**
 * Conversation namespace. Mirrors `ConversationInterface` and
 * `ConversationContextInterface` in the Python SDK.
 */

import { InvalidInputError } from '../errors.js';
import { validateConversationId } from '../util/validators.js';
import type { HttpTransport } from '../transport/http.js';
import { fetchContext, type FetchHooks } from '../context/fetch.js';
import { agentModelOf, withAgentModel, type AgentModel } from '../agent-model.js';
import type { FetchOptions, Json, RawContext } from '../context/types.js';

export interface RecordMessageOptions {
  conversation_id?: string;
  conversationId?: string;
  role: string;
  content: string;
  user_id?: string;
  userId?: string;
  customer_id?: string;
  customerId?: string;
  session_id?: string;
  sessionId?: string;
  metadata?: Json;
  /**
   * The rung this conversation belongs to, named in full, e.g.
   * `{ tenant: 'acme', practice: 'high_street' }`.
   *
   * Only relevant to a client whose ladder is deeper than the default three
   * rungs (root, tenant, person). On the default ladder `customer_id` and
   * `user_id` already say everything a path could.
   *
   * Sent as `scope` on the wire, matching Python's `scope_path`. Sending it
   * once per conversation is enough: the server fills a missing path and never
   * overwrites one.
   */
  scope_path?: Record<string, string>;
  scopePath?: Record<string, string>;
  /**
   * The model the agent runs on, e.g. `'gpt-4o'`. Overrides the client's
   * `agent_model` for this turn. The server sizes compaction to this model's
   * context window. Python's `model=`.
   */
  model?: string;
  /** The agent model's context window in tokens, for a model the server would not know. */
  context_window?: number;
  contextWindow?: number;
}

export interface TranscriptTurn {
  role: string;
  content: string;
  [key: string]: unknown;
}

export interface IngestTranscriptOptions {
  conversation_id?: string;
  conversationId?: string;
  user_id?: string;
  userId?: string;
  transcript: string | TranscriptTurn[];
  customer_id?: string;
  customerId?: string;
  conversation_type?: string;
  conversationType?: string;
  analysis?: Json;
  metadata?: Json;
  started_at?: string | Date;
  startedAt?: string | Date;
  ended_at?: string | Date;
  endedAt?: string | Date;
  /**
   * The rung this conversation belongs to, named in full, e.g.
   * `{ tenant: 'acme', practice: 'high_street' }`.
   *
   * Only relevant to a client whose ladder is deeper than the default three
   * rungs (root, tenant, person). On the default ladder `customer_id` and
   * `user_id` already say everything a path could.
   *
   * Sent as `scope` on the wire, matching Python's `scope_path`. Sending it
   * once per conversation is enough: the server fills a missing path and never
   * overwrites one.
   */
  scope_path?: Record<string, string>;
  scopePath?: Record<string, string>;
}

/**
 * Compaction levels, mirroring Python's `CompactionLevel` enum. The JS SDK
 * exported no equivalent, so `compaction_level` was an untyped `string` and a
 * typo reached the server instead of the compiler.
 */
export const COMPACTION_LEVELS = [
  'low', 'medium', 'high', 'adaptive', 'aggressive', 'balanced', 'conservative',
] as const;

export type CompactionLevel = (typeof COMPACTION_LEVELS)[number];

export interface CompactOptions {
  conversation_id?: string;
  conversationId?: string;
  strategy?: string;
  compaction_level?: CompactionLevel | string;
  compactionLevel?: CompactionLevel | string;
  target_tokens?: number;
  targetTokens?: number;
  force?: boolean;
}

function either<T>(o: Record<string, unknown>, snake: string, camel: string): T | undefined {
  const a = o[snake];
  if (a !== undefined) return a as T;
  const b = o[camel];
  return b === undefined ? undefined : (b as T);
}

function iso(v: string | Date | undefined): string | undefined {
  if (v === undefined) return undefined;
  return v instanceof Date ? v.toISOString() : v;
}

export function buildRecordMessageBody(options: RecordMessageOptions): Json {
  const o = options as unknown as Record<string, unknown>;
  const conversationId = either<string>(o, 'conversation_id', 'conversationId');
  const userId = either<string>(o, 'user_id', 'userId');
  const customerId = either<string>(o, 'customer_id', 'customerId');

  if (!conversationId) throw new InvalidInputError('conversation_id is required');
  // Python validates the format here (and in record_messages_batch, which
  // reuses this builder per message). Note it deliberately does NOT validate
  // in ingest_transcript.
  validateConversationId(conversationId);
  if (!options.role) throw new InvalidInputError('role is required');
  // Checked before sending, as Python checks it. The server refuses any other
  // role with a 422; until 0.5.17 JS sent it to find out.
  if (!(MESSAGE_ROLES as readonly string[]).includes(options.role)) {
    throw new InvalidInputError(`Invalid role '${options.role}'. Must be 'user' or 'assistant'.`);
  }
  if (options.content === undefined || options.content === null) {
    throw new InvalidInputError('content is required');
  }
  if (!userId) throw new InvalidInputError('user_id is required');
  // customer_id is NOT required. A B2C instance rejects one, so demanding it
  // here made the only correct B2C shape impossible to express.
  // checked by the caller, which holds the transport; see createConversationNamespace

  const body: Json = {
    conversation_id: conversationId,
    role: options.role,
    content: options.content,
    user_id: userId,
    customer_id: customerId,
    // Python always sends this, defaulting to {}.
    metadata: options.metadata ?? {},
  };
  // Only when set, like Python's `if session_id:`. An empty one reached the
  // server, which turns any non-UUID session id into its UUID5, so every
  // conversation sent "" shared one session.
  const sessionId = either<string>(o, 'session_id', 'sessionId');
  if (sessionId) body['session_id'] = sessionId;
  // `scope` on the wire, `scope_path` in the SDK, the same split Python makes.
  // Omitted when unset so a client on the default ladder sends exactly the
  // body it sends today.
  const scopePath = either<Record<string, string>>(o, 'scope_path', 'scopePath');
  // Omitted when absent or empty, like Python's `if scope_path:`.
  if (scopePath !== undefined && scopePath !== null && Object.keys(scopePath).length > 0) {
    body['scope'] = scopePath;
  }
  return body;
}

export function buildIngestTranscriptBody(options: IngestTranscriptOptions): Json {
  const o = options as unknown as Record<string, unknown>;
  const conversationId = either<string>(o, 'conversation_id', 'conversationId');
  const userId = either<string>(o, 'user_id', 'userId');
  if (!conversationId) throw new InvalidInputError('conversation_id is required');
  if (!userId) throw new InvalidInputError('user_id is required');
  if (options.transcript === undefined || options.transcript === null) {
    throw new InvalidInputError('transcript is required');
  }

  const body: Json = {
    conversation_id: conversationId,
    user_id: userId,
    transcript: options.transcript,
  };
  // Unset optionals are omitted so the server applies its own defaults, which
  // is what Python does here (and notably NOT what it does for memories.create).
  const customerId = either<string>(o, 'customer_id', 'customerId');
  const conversationType = either<string>(o, 'conversation_type', 'conversationType');
  const startedAt = iso(either<string | Date>(o, 'started_at', 'startedAt'));
  const endedAt = iso(either<string | Date>(o, 'ended_at', 'endedAt'));
  if (customerId !== undefined) body['customer_id'] = customerId;
  if (conversationType !== undefined) body['conversation_type'] = conversationType;
  if (options.analysis !== undefined) body['analysis'] = options.analysis;
  if (options.metadata !== undefined) body['metadata'] = options.metadata;
  if (startedAt !== undefined) body['started_at'] = startedAt;
  if (endedAt !== undefined) body['ended_at'] = endedAt;
  // Python 0.4.10 sends this on `ingest_transcript` and JS did not, so a
  // TypeScript client on a deep ladder had no route that could name a rung.
  // Omitted when absent or empty, like Python's `if scope_path:`.
  const scopePath = either<Record<string, string>>(o, 'scope_path', 'scopePath');
  if (scopePath !== undefined && scopePath !== null && Object.keys(scopePath).length > 0) {
    body['scope'] = scopePath;
  }
  return body;
}

export function buildCompactBody(options: CompactOptions): Json {
  const o = options as unknown as Record<string, unknown>;
  const conversationId = either<string>(o, 'conversation_id', 'conversationId');
  if (!conversationId) throw new InvalidInputError('conversation_id is required');
  validateConversationId(conversationId);

  const body: Json = { conversation_id: conversationId, force: options.force ?? false };
  const strategy = options.strategy ?? either<string>(o, 'compaction_level', 'compactionLevel');
  const targetTokens = either<number>(o, 'target_tokens', 'targetTokens');
  if (strategy !== undefined) body['strategy'] = strategy;
  if (targetTokens !== undefined) body['target_tokens'] = targetTokens;
  return body;
}

/**
 * Result of `conversation.ingest_transcript`, mirroring Python's
 * `TranscriptIngestResponse`. It was `Json` here, so `ingestion_id` came back
 * as `unknown` and could not be handed to `wait_for_completion` without a cast.
 */
export interface TranscriptIngestResult {
  conversation_id: string;
  /** The raw client-supplied id, echoed back. */
  external_conversation_id: string;
  /** Always set on a 2xx, including the `duplicate` branch. */
  ingestion_id: string;
  status: 'queued' | 'duplicate';
  turns_recorded: number;
  summary_status: 'in_progress' | 'already_compacted' | 'skipped';
  /**
   * What the server could not do with this push: no event time supplied
   * (recency ranking cannot order these memories), or existing turns kept.
   * Always a list. Since 0.5.16.
   */
  warnings: string[];
  /** ISO-8601. */
  queued_at: string;
  [key: string]: unknown;
}

export interface ConversationNamespace {
  record_message(options: RecordMessageOptions): Promise<Json>;
  record_messages_batch(messages: RecordMessageOptions[]): Promise<Json>;
  ingest_transcript(options: IngestTranscriptOptions): Promise<TranscriptIngestResult>;
  context: {
    fetch(options?: FetchOptions): Promise<RawContext>;
    get_context_for_prompt(options?: {
      conversation_id?: string;
      conversationId?: string;
      style?: string;
      /** Search the earlier turns locally after a compaction (default true). */
      search_history?: boolean;
      searchHistory?: boolean;
      /** What to search for; defaults to the latest user message. */
      query?: string;
    }): Promise<Json>;
    compact(options: CompactOptions): Promise<Json>;
    get_compacted(options: { conversation_id?: string; conversationId?: string; version?: number; format?: string }): Promise<Json | null>;
    get_compaction_status(options: { conversation_id?: string; conversationId?: string } | string): Promise<Json>;
    subscribe_to_compaction_updates(
      conversationId: string,
      callback: CompactionCallback,
    ): () => void;
    unsubscribe_all_compaction_updates(conversationId?: string): number;
  };
}

export type CompactionCallback = (bundle: Json) => unknown;

/**
 * Registry of per-conversation compaction subscribers.
 *
 * Split out from the namespace so the instance stream can dispatch into it
 * without the two namespaces importing each other.
 */
export class CompactionSubscribers {
  readonly #byConversation = new Map<string, CompactionCallback[]>();

  add(conversationId: string, callback: CompactionCallback): () => void {
    const list = this.#byConversation.get(conversationId);
    if (list === undefined) this.#byConversation.set(conversationId, [callback]);
    else list.push(callback);

    let unsubscribed = false;
    return () => {
      // Idempotent, matching Python: calling the thunk twice is a no-op.
      if (unsubscribed) return;
      unsubscribed = true;
      const current = this.#byConversation.get(conversationId);
      if (current === undefined) return;
      const at = current.indexOf(callback);
      if (at !== -1) current.splice(at, 1);
      if (current.length === 0) this.#byConversation.delete(conversationId);
    };
  }

  removeAll(conversationId?: string): number {
    if (conversationId === undefined) {
      let count = 0;
      for (const list of this.#byConversation.values()) count += list.length;
      this.#byConversation.clear();
      return count;
    }
    const list = this.#byConversation.get(conversationId);
    if (list === undefined) return 0;
    this.#byConversation.delete(conversationId);
    return list.length;
  }

  /** Fires subscribers in registration order. A throwing listener must not break the stream reader. */
  dispatch(conversationId: string, bundle: Json, onError: (e: unknown) => void): void {
    // Copy first: a callback that unsubscribes itself would otherwise mutate
    // the array mid-iteration and skip the next listener.
    for (const callback of [...(this.#byConversation.get(conversationId) ?? [])]) {
      try {
        const result = callback(bundle) as unknown;
        // An async callback returns a promise; swallow its rejection the way
        // Python swallows one from an asyncio task.
        if (result instanceof Promise) result.catch(onError);
      } catch (error) {
        onError(error);
      }
    }
  }
}

/**
 * Pair each message a batch write stored with the server's row for it.
 *
 * The server's `results` holds the messages that were stored, in input order,
 * and leaves the failed ones out; those are in `errors`, each with the `index`
 * it had in the request. Buffering every body, or pairing `results` against
 * the whole input by position, puts a message the server refused into the
 * caller's next prompt.
 *
 * Returns nothing when the two cannot be lined up, as Python does. A row with
 * no `message_id` is treated as not stored.
 */
export function storedBatchRows(
  bodies: Array<Record<string, unknown>>,
  result: unknown,
): Array<[Record<string, unknown>, Record<string, unknown>]> {
  const r = (result ?? {}) as Record<string, unknown>;
  const results = Array.isArray(r['results']) ? (r['results'] as Array<Record<string, unknown>>) : [];
  const errors = Array.isArray(r['errors']) ? (r['errors'] as Array<Record<string, unknown>>) : [];
  const failed = new Set(errors.map((e) => e?.['index']));
  const stored = bodies.filter((_, index) => !failed.has(index));
  if (stored.length !== results.length) return [];
  const pairs: Array<[Record<string, unknown>, Record<string, unknown>]> = [];
  stored.forEach((body, i) => {
    const row = results[i];
    if (row && typeof row === 'object' && row['message_id']) pairs.push([body, row]);
  });
  return pairs;
}

/**
 * Called after a successful write with the ids that were written for.
 *
 * Backs `SYNAP_SDK_CACHE_INVALIDATE_ON_WRITE`. The cloud invalidates its own
 * Redis caches on ingest, but that stops at the process boundary: without this
 * the in-SDK anticipation cache keeps serving the pre-write view for the rest
 * of its TTL. Python calls its equivalent from these same three write paths.
 */
export type WriteInvalidator = (ids: {
  userId?: string | undefined;
  customerId?: string | undefined;
}) => void;

/**
 * Called with each turn a write path sends, so it can be buffered locally.
 *
 * This is what makes the next fetch see the turn that was just recorded, ahead
 * of the server having persisted and compacted it.
 */
export type TurnRecorder = (turn: {
  conversationId: string;
  role: string;
  content: string;
  timestamp?: string | undefined;
  /** The server's id for the turn, from the write's response. */
  messageId?: string | undefined;
  /**
   * The conversation's message count after this write, as the server reported
   * it. This is what tells the local store whether it holds the whole
   * conversation. Absent from a server that predates it.
   */
  serverMessageCount?: number | null | undefined;
  /** Other writes this process still has outstanding for the conversation. */
  inFlight?: number | undefined;
}) => void;

/** The roles the server accepts on a recorded message (RecordMessageRequest.role). */
export const MESSAGE_ROLES: readonly string[] = ['user', 'assistant'];

/** The styles the server renders (CompactedContext.get_injection_format). */
export const PROMPT_STYLES: readonly string[] = ['structured', 'narrative', 'bullet_points'];

export function createConversationNamespace(
  transport: HttpTransport,
  subscribers: CompactionSubscribers = new CompactionSubscribers(),
  onWrite: WriteInvalidator = () => {},
  onTurn: TurnRecorder = () => {},
  /**
   * Applied to a conversation-scoped response before it is returned.
   *
   * `scopePath` is carried because one of the things this hook does is splice
   * in a cached user summary, and a summary prefetched at one rung must not be
   * spliced into a response for a caller standing at another. Python reads the
   * path off the fetch arguments at that call site; here the call site is a
   * callback, so the path has to travel with it or the rung check is silently
   * skipped on this one path.
   */
  onFetched: (
    response: RawContext,
    conversationId: string | undefined,
    userId: string | undefined,
    scopePath: Record<string, string> | undefined,
  ) => void = () => {},
  /**
   * SDK-authoritative short-term context. Returns a locally rendered response
   * when the flag is on AND the store is warm for this conversation, which
   * skips the cloud round trip entirely; null falls through to the network.
   */
  localShortTerm: {
    prompt: (conversationId: string, style: string) => Json | null;
    compacted: (conversationId: string, format: string) => Json | null;
    /**
     * Handed the server's `/context-for-prompt` answer after the local store
     * could not answer, so the store can be brought back in step.
     */
    reseed?: (conversationId: string, serverResponse: Json) => void;
    /**
     * The response with the earlier turns that best match `query` appended,
     * found by a local search of the turn archive. Python:
     * `MaximemSynapSDK._with_history_excerpts`.
     */
    historyExcerpts?: (conversationId: string, response: Json, query: string | undefined) => Promise<Json>;
  } = { prompt: () => null, compacted: () => null },
  /** The client's `agent_model` / `agent_context_window`, folded into each turn's metadata. */
  agentModel: () => AgentModel = () => ({}),
  fetchHooks: FetchHooks = {},
): ConversationNamespace {
  // conversation_id -> REST writes sent and not yet buffered locally. Lets the
  // local store tell two of this process's own concurrent writes apart from a
  // second writer.
  const writesInFlight = new Map<string, number>();
  const withTurnModel = (body: Json, options: unknown): Json => {
    const metadata = withAgentModel(
      body['metadata'] as Record<string, unknown> | undefined, agentModelOf(options), agentModel(),
    );
    return metadata === undefined ? body : { ...body, metadata: metadata as Json };
  };
  const writeStarted = (conversationId: string): void => {
    writesInFlight.set(conversationId, (writesInFlight.get(conversationId) ?? 0) + 1);
  };
  const writeLanded = (conversationId: string): void => {
    const remaining = (writesInFlight.get(conversationId) ?? 0) - 1;
    if (remaining > 0) writesInFlight.set(conversationId, remaining);
    else writesInFlight.delete(conversationId);
  };
  const confirmation = (
    result: unknown,
    conversationId: string,
  ): { messageId?: string; serverMessageCount?: number | null; inFlight: number } => {
    const r = (result ?? {}) as Record<string, unknown>;
    const count = r['conversation_message_count'];
    return {
      ...(typeof r['message_id'] === 'string' ? { messageId: r['message_id'] } : {}),
      serverMessageCount: typeof count === 'number' ? count : null,
      inFlight: writesInFlight.get(conversationId) ?? 0,
    };
  };
  const convId = (v: { conversation_id?: string; conversationId?: string } | string): string => {
    const id = typeof v === 'string' ? v : (v.conversation_id ?? v.conversationId);
    if (!id) throw new InvalidInputError('conversation_id is required');
    // Covers context.fetch, get_compacted, get_compaction_status and
    // get_context_for_prompt: the four routes Python validates through this
    // same shape. ingest_transcript does not go through here, which is correct.
    validateConversationId(id);
    return id;
  };

  return {
    async record_message(options) {
      await transport.checkCustomerId(
        (options as unknown as Record<string, unknown>)['customer_id'] ??
          (options as unknown as Record<string, unknown>)['customerId'],
        'conversation.record_message',
      );
      const body = withTurnModel(buildRecordMessageBody(options), options);
      const conversationId = String(body['conversation_id'] ?? '');
      writeStarted(conversationId);
      let result: Json;
      try {
        result = await transport.request<Json>('conversations_messages', { body });
      } finally {
        writeLanded(conversationId);
      }
      // Buffer locally only AFTER the server accepted it. Recording a turn the
      // write failed on would surface a message that does not exist.
      onTurn({
        conversationId,
        role: String(body['role'] ?? 'user'),
        content: String(body['content'] ?? ''),
        timestamp: body['timestamp'] as string | undefined,
        ...confirmation(result, conversationId),
      });
      onWrite({
        userId: body['user_id'] as string | undefined,
        customerId: body['customer_id'] as string | undefined,
      });
      return result;
    },

    async record_messages_batch(messages) {
      if (!Array.isArray(messages) || messages.length === 0) {
        throw new InvalidInputError('messages must be a non-empty array');
      }
      // Every message checked BEFORE any is sent: a batch that fails on its
      // fifth message would otherwise be half-applied.
      for (const m of messages) {
        const mo = m as unknown as Record<string, unknown>;
        await transport.checkCustomerId(
          mo['customer_id'] ?? mo['customerId'], 'conversation.record_messages_batch',
        );
      }
      const bodies = messages.map((m) => withTurnModel(buildRecordMessageBody(m), m));
      const result = await transport.request<Json>('conversations_messages_batch', {
        body: { messages: bodies },
      });
      // Buffer only what the server stored, each paired with its own row.
      for (const [body, row] of storedBatchRows(bodies, result)) {
        const conversationId = String(body['conversation_id'] ?? '');
        onTurn({
          conversationId,
          role: String(body['role'] ?? 'user'),
          content: String(body['content'] ?? ''),
          timestamp: body['timestamp'] as string | undefined,
          ...confirmation(row, conversationId),
        });
      }
      // Every distinct entity in the batch, not just the first message's.
      for (const body of bodies) {
        onWrite({
          userId: body['user_id'] as string | undefined,
          customerId: body['customer_id'] as string | undefined,
        });
      }
      return result;
    },

    async ingest_transcript(options) {
      // First, as Python checks it and as every other write here does. Until
      // 0.5.16 this was the one write that skipped it, so a B2C customer_id
      // went to the server and came back as a plain InvalidInputError.
      const o = options as unknown as Record<string, unknown>;
      await transport.checkCustomerId(
        o['customer_id'] ?? o['customerId'], 'conversation.ingest_transcript',
      );
      const body = buildIngestTranscriptBody(options);
      const raw = await transport.request<TranscriptIngestResult>('conversations_ingest', { body });
      // Always a list, as Python's response model defaults it.
      const result: TranscriptIngestResult = {
        ...raw,
        warnings: Array.isArray(raw?.warnings) ? raw.warnings.map(String) : [],
      };
      onWrite({
        userId: body['user_id'] as string | undefined,
        customerId: body['customer_id'] as string | undefined,
      });
      return result;
    },

    context: {
      async fetch(options = {}) {
        // Conversation-scope fetch is one of Python's seven validation sites.
        // It does not route through convId() because the id travels in the
        // body rather than the path, so validate it explicitly.
        const id =
          (options as { conversation_id?: string; conversationId?: string }).conversation_id ??
          (options as { conversationId?: string }).conversationId;
        validateConversationId(id);
        const response = await fetchContext(transport, 'conversation', options, fetchHooks);
        // Read-your-writes. This is the path where it matters most: a fast
        // follow-up on the same conversation would otherwise retrieve context
        // missing the turn just recorded.
        onFetched(
          response,
          id,
          (options as { user_id?: string; userId?: string }).user_id ??
            (options as { userId?: string }).userId,
          (options as { scope_path?: Record<string, string> }).scope_path ??
            (options as { scopePath?: Record<string, string> }).scopePath,
        );
        return response;
      },

      get_context_for_prompt: async (options = {}) => {
        const id = convId(options);
        const style = options.style ?? 'structured';
        // Refused here, as Python refuses it. A style the server cannot render
        // was a 500 from it, or silently ignored when the compaction has a block.
        if (!PROMPT_STYLES.includes(style)) {
          throw new InvalidInputError(
            `Invalid style '${style}'. Must be one of: ${PROMPT_STYLES.join(', ')}`,
          );
        }
        const search = (options.search_history ?? options.searchHistory ?? true) && localShortTerm.historyExcerpts;
        const withHistory = async (response: Json): Promise<Json> => {
          if (!search) return response;
          try {
            return await localShortTerm.historyExcerpts!(id, response, options.query);
          } catch {
            return response; // never fail the prompt over its extras
          }
        };
        const local = localShortTerm.prompt(id, style);
        if (local !== null) return withHistory(local);
        const result = await transport.request<Json>('context_for_prompt', {
          pathParams: { conversation_id: id },
          query: { style },
        });
        const response = (result?.['context_for_prompt'] as Json) ?? result ?? {};
        localShortTerm.reseed?.(id, response);
        return withHistory(response);
      },

      async compact(options) {
        return transport.request<Json>('conversations_compact', { body: buildCompactBody(options) });
      },

      get_compacted: async (options) => {
        const id = convId(options);
        // Python only takes the local path when no explicit version is asked
        // for: a pinned version is a request for a specific server artefact.
        if (options.version === undefined) {
          const local = localShortTerm.compacted(id, options.format ?? 'structured');
          if (local !== null) return local;
        }
        const query: Record<string, string | number> = { format: options.format ?? 'structured' };
        if (options.version !== undefined) query['version'] = options.version;
        try {
          return await transport.request<Json>('conversations_compacted', {
            pathParams: { conversation_id: id },
            query,
          });
        } catch (error) {
          // Python returns None when there is nothing compacted yet, rather
          // than surfacing a 404. Matching that keeps callers identical.
          if ((error as { code?: string }).code === 'context_not_found') return null;
          throw error;
        }
      },

      async get_compaction_status(options) {
        return transport.request<Json>('conversations_compaction_status', {
          pathParams: { conversation_id: convId(options) },
        });
      },

      /**
       * Fire `callback` whenever a `compaction_update` bundle arrives on the
       * Listen stream for this conversation.
       *
       * Requires an active `client.instance.listen()`; without it no bundles
       * arrive and the callback never fires. Registering before listening is
       * fine, which is why this does not throw when the stream is down.
       *
       * Returns an idempotent unsubscribe thunk, matching Python.
       */
      subscribe_to_compaction_updates(conversationId, callback) {
        if (!conversationId) throw new InvalidInputError('conversation_id is required');
        if (typeof callback !== 'function') throw new InvalidInputError('callback is required');
        return subscribers.add(conversationId, callback);
      },

      /** Remove subscribers for one conversation, or all of them. Returns the count removed. */
      unsubscribe_all_compaction_updates(conversationId?: string) {
        return subscribers.removeAll(conversationId);
      },
    },
  };
}
