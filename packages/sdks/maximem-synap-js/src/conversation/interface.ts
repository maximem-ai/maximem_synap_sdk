/**
 * Conversation namespace. Mirrors `ConversationInterface` and
 * `ConversationContextInterface` in the Python SDK.
 */

import { InvalidInputError } from '../errors.js';
import { validateConversationId } from '../util/validators.js';
import type { HttpTransport } from '../transport/http.js';
import { fetchContext } from '../context/fetch.js';
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
  const sessionId = either<string>(o, 'session_id', 'sessionId');
  if (sessionId !== undefined) body['session_id'] = sessionId;
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
    get_context_for_prompt(options?: { conversation_id?: string; conversationId?: string; style?: string }): Promise<Json>;
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
}) => void;

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
  } = { prompt: () => null, compacted: () => null },
): ConversationNamespace {
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
      transport.checkCustomerId(
        (options as unknown as Record<string, unknown>)['customer_id'] ??
          (options as unknown as Record<string, unknown>)['customerId'],
        'conversation.record_message',
      );
      const body = buildRecordMessageBody(options);
      const result = await transport.request<Json>('conversations_messages', { body });
      // Buffer locally only AFTER the server accepted it. Recording a turn the
      // write failed on would surface a message that does not exist.
      onTurn({
        conversationId: String(body['conversation_id'] ?? ''),
        role: String(body['role'] ?? 'user'),
        content: String(body['content'] ?? ''),
        timestamp: body['timestamp'] as string | undefined,
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
        transport.checkCustomerId(
          mo['customer_id'] ?? mo['customerId'], 'conversation.record_messages_batch',
        );
      }
      const bodies = messages.map(buildRecordMessageBody);
      const result = await transport.request<Json>('conversations_messages_batch', {
        body: { messages: bodies },
      });
      for (const body of bodies) {
        onTurn({
          conversationId: String(body['conversation_id'] ?? ''),
          role: String(body['role'] ?? 'user'),
          content: String(body['content'] ?? ''),
          timestamp: body['timestamp'] as string | undefined,
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
      const body = buildIngestTranscriptBody(options);
      const result = await transport.request<TranscriptIngestResult>('conversations_ingest', { body });
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
        const response = await fetchContext(transport, 'conversation', options);
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
        const local = localShortTerm.prompt(id, options.style ?? 'structured');
        if (local !== null) return local;
        const result = await transport.request<Json>('context_for_prompt', {
          pathParams: { conversation_id: id },
          query: { style: options.style ?? 'structured' },
        });
        return (result?.['context_for_prompt'] as Json) ?? result ?? {};
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
