// SynapMemory — a MastraMemory implementation backed by Synap.
//
// Drop-in for `new Agent({ memory: new SynapMemory({...}) })`, with or without
// a Mastra storage attached. What Mastra's default Agent path calls on a memory
// is `getInputProcessors` and `getOutputProcessors` (see processors.ts), and
// those are what is Synap-backed:
//
//   - input: Synap's long-term context becomes a system message, and once
//     Synap has compacted the conversation its compaction replaces the turns
//     Mastra loaded before the current one
//   - output: the run's turns are written to Synap (the stream, opened on
//     first use; REST when it cannot carry them), plus tool calls, results and
//     reasoning on the stream
//
// The methods below are the explicit API, for callers that invoke them:
//
//   - recall(): loads prior messages via sdk.conversation.context.get_context_for_prompt
//   - saveMessages(): writes each turn the same way the output processor does
//   - getSystemMessage(): fetches Synap context via sdk.fetch and returns it
//
// Methods that Synap doesn't natively back (thread metadata, working memory,
// delete, clone) are implemented with in-process defaults + honest warnings.
// Listed below per method so users know what's real vs. best-effort.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
import { MastraMemory } from "@mastra/core/memory";
import type {
  StorageThreadType,
  MessageDeleteInput,
  MemoryConfig,
  MemoryConfigInternal,
  WorkingMemoryTemplate,
} from "@mastra/core/memory";
import type {
  StorageListMessagesInput,
  StorageListThreadsInput,
  StorageListThreadsOutput,
  StorageCloneThreadInput,
  StorageCloneThreadOutput,
} from "@mastra/core/storage";
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import type { MastraDBMessage } from "@mastra/core/agent/message-list";

import {
  COMPACTION_ID_PREFIX,
  SynapInputProcessor,
  SynapOutputProcessor,
  isHistoryMessage,
  messageText,
} from "./processors.js";
import { SynapMastraStreamReporter, type SynapStreamOptions } from "./stream.js";
import type {
  SynapSdkLike,
  SynapRecentMessage,
  SynapIdentityOptions,
} from "./types.js";

export interface SynapMemoryOptions extends SynapIdentityOptions, SynapStreamOptions {
  /** Optional unique id for this memory instance (Mastra base accepts this). */
  id?: string;
  /**
   * When true (default), a fresh recall() triggers sdk.fetch(search_query=[...])
   * and we return the formatted_context as a system message preamble. Set to
   * false if you want recall-only semantics without memory-context injection.
   */
  injectSystemContext?: boolean;
}

/**
 * Construct a Synap-backed Mastra memory.
 *
 * Error policy mirrors the rest of the Synap integrations:
 *   - read failures degrade gracefully (the framework's own history stands, no
 *     Synap context is added)
 *   - turn writes never throw: a failed write is logged and the run goes on
 */

export class SynapMemory extends MastraMemory {
  private readonly sdk: SynapSdkLike;
  private readonly userId: string;
  private readonly customerId: string;
  private readonly mode: string;
  private readonly injectSystemContext: boolean;
  private readonly stream: SynapMastraStreamReporter;

  // In-process store for thread metadata. Mastra's Agent flow consults this
  // primarily for thread IDs; durable thread metadata is a v0.2 concern.
  private readonly threads: Map<string, StorageThreadType> = new Map();

  // Guard rails so we warn once per unsupported op and don't spam logs.
  private deleteWarned = false;
  private workingMemoryWarned = false;

  constructor(options: SynapMemoryOptions) {
    if (!options || !options.sdk) {
      throw new Error("SynapMemory requires a non-null sdk");
    }
    if (!options.userId) {
      throw new Error("SynapMemory requires a non-empty userId");
    }
    super({ id: options.id, name: "synap" });
    this.sdk = options.sdk;
    this.userId = options.userId;
    this.customerId = options.customerId ?? "";
    this.mode = options.mode ?? "accurate";
    this.injectSystemContext = options.injectSystemContext ?? true;
    this.stream = new SynapMastraStreamReporter(
      options.sdk,
      this.userId,
      this.customerId,
      {
        reportToolResults: options.reportToolResults,
        reportThoughts: options.reportThoughts,
      },
    );
  }

  // ── essential methods (real, Synap-backed) ─────────────────────────────

  /**
   * Called before each turn. Returns a Synap-formatted context block that
   * Mastra injects as a system-message suffix.
   */
  async getSystemMessage(input: {
    threadId: string;
    resourceId?: string;
    memoryConfig?: MemoryConfigInternal;
  }): Promise<string | null> {
    if (!this.injectSystemContext) return null;

    try {
      const response = await this.sdk.fetch({
        conversation_id: input.threadId ?? null,
        user_id: this.userId,
        customer_id: this.customerId || null,
        max_results: 20,
        mode: this.mode,
        include_conversation_context: false,
      });
      const text = (response.formatted_context ?? "").trim();
      return text ? text : null;
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(
        `SynapMemory.getSystemMessage: sdk.fetch failed threadId=${input.threadId}`,
        err,
      );
      return null;
    }
  }

  /**
   * Load message history for a thread. Uses Synap's conversation context
   * endpoint rather than memories.* to keep the chat log round-trip clean.
   * On SDK failure, returns an empty page (read-side graceful degradation).
   */
  async recall(
    args: StorageListMessagesInput & {
      threadConfig?: MemoryConfigInternal;
      vectorSearchString?: string;
      includeSystemReminders?: boolean;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      observabilityContext?: any;
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): Promise<any> {
    const threadIdArr = Array.isArray(args.threadId) ? args.threadId : [args.threadId];
    const primaryThreadId = threadIdArr[0] ?? "";
    if (!primaryThreadId) {
      return this.emptyRecallResult();
    }

    try {
      const response = await this.sdk.conversation.context.get_context_for_prompt({
        conversation_id: primaryThreadId,
      });
      const raw = response.recent_messages ?? [];
      const messages = raw
        .filter((m) => !!m && typeof m.content === "string" && m.content.trim() !== "")
        .map((m) => this.toMastraMessage(m, primaryThreadId, args.resourceId));
      // The compaction block (and the earlier turns the SDK found for the
      // latest message) stands in for the turns it covers: recent_messages
      // holds only the turns after it. Older SDKs send neither.
      const preamble = [response.context_block, response.history_block].filter((p) => !!p).join("\n\n");
      if (preamble) {
        const first = raw.find((m) => !!m && typeof m.timestamp === "string");
        messages.unshift(
          this.toMastraMessage(
            {
              role: "system",
              content: preamble,
              // Just before the first recent turn, so it sorts ahead of them.
              timestamp: first ? new Date(Date.parse(first.timestamp) - 1).toISOString() : new Date(0).toISOString(),
              message_id: `${COMPACTION_ID_PREFIX}${primaryThreadId}`,
            },
            primaryThreadId,
            args.resourceId,
          ),
        );
      }
      return {
        messages,
        total: messages.length,
        page: 0,
        perPage: messages.length,
        hasMore: false,
      };
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(
        `SynapMemory.recall: get_context_for_prompt failed threadId=${primaryThreadId}`,
        err,
      );
      return this.emptyRecallResult();
    }
  }

  /**
   * The input processors Mastra's Agent runs before the model: Synap's
   * long-term context and history (see processors.ts).
   *
   * With a Mastra storage attached, Mastra's own processors (message history,
   * working memory, semantic recall) come first and Synap's runs after them, so
   * its compaction can replace the history they loaded. Without one the base
   * class would throw "Memory requires a storage provider", so only Synap's
   * runs and Synap's turns are the history.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async getInputProcessors(configured?: any[], context?: any): Promise<any[]> {
    const own = this._storage !== undefined
      ? await (super.getInputProcessors as (...a: unknown[]) => Promise<unknown[]>)
          .call(this, configured ?? [], context)
      : [];
    return [
      ...own,
      new SynapInputProcessor({
        sdk: this.sdk,
        userId: this.userId,
        customerId: this.customerId,
        mode: this.mode,
        injectSystemContext: this.injectSystemContext,
      }),
    ];
  }

  /** The output processors: Mastra's own when storage is attached, then Synap's. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async getOutputProcessors(configured?: any[], context?: any): Promise<any[]> {
    const own = this._storage !== undefined
      ? await (super.getOutputProcessors as (...a: unknown[]) => Promise<unknown[]>)
          .call(this, configured ?? [], context)
      : [];
    return [
      ...own,
      new SynapOutputProcessor({
        persist: (messages, threadId) => this.persist(messages, threadId),
      }),
    ];
  }

  /**
   * Write messages to Synap. Each turn goes through `recordTurn` (the stream,
   * opened on first use, REST when it cannot carry it; never both), and tool
   * calls, results and reasoning go on the stream. A message Mastra saves again
   * (it re-saves one when a tool result is filled in) is not written twice.
   * Never throws.
   */
  private async persist(messages: unknown[], fallbackThreadId = ""): Promise<unknown[]> {
    const out: unknown[] = [];
    for (const m of messages) {
      const role = typeof (m as { role?: unknown })?.role === "string" ? (m as { role: string }).role : "";
      if (role !== "user" && role !== "assistant") continue;
      // Synap's own history, put there by the input processor or recall(), is
      // not something somebody said.
      if (isHistoryMessage(m)) continue;

      const content = messageText(m);
      const own = (m as { threadId?: unknown })?.threadId;
      const threadId = typeof own === "string" && own ? own : fallbackThreadId;

      // Above the text guard on purpose. An assistant message whose only parts
      // are tool calls carries no text at all, and those tool events are the
      // whole reason to report it: they have no other durable home.
      const written = await this.stream.report(m, { threadId, role, text: content });
      if (written) out.push(m);
    }
    return out;
  }

  /**
   * Write each message to Synap, the way the output processor does. Mastra's
   * Agent calls this itself only with `savePerStep`.
   *
   * System messages are the instructions, not turns: they are written over
   * REST as before and never put on the stream.
   */
  async saveMessages(args: {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    messages: any[];
    memoryConfig?: MemoryConfig | undefined;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    observabilityContext?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  }): Promise<any> {
    const messages = args.messages ?? [];
    const out = await this.persist(messages);
    for (const m of messages) {
      if (m?.role !== "system" || isHistoryMessage(m)) continue;
      const content = messageText(m);
      const threadId = typeof m?.threadId === "string" ? m.threadId : "";
      if (!content || !threadId) continue;
      await this.sdk.conversation.record_message({
        conversation_id: threadId,
        role: "system",
        content,
        user_id: this.userId,
        ...(this.customerId ? { customer_id: this.customerId } : {}),
      });
      out.push(m);
    }
    return { messages: out };
  }

  // ── thread management (local Map, not Synap-backed) ────────────────────

  async getThreadById({
    threadId,
  }: {
    threadId: string;
  }): Promise<StorageThreadType | null> {
    return this.threads.get(threadId) ?? null;
  }

  async listThreads(args: StorageListThreadsInput): Promise<StorageListThreadsOutput> {
    const filter = args.filter ?? {};
    const all = [...this.threads.values()].filter((t) => {
      if (filter.resourceId && t.resourceId !== filter.resourceId) return false;
      if (filter.metadata) {
        for (const [k, v] of Object.entries(filter.metadata)) {
          if ((t.metadata ?? {})[k] !== v) return false;
        }
      }
      return true;
    });
    const page = args.page ?? 0;
    const perPage = args.perPage === false ? all.length : args.perPage ?? 100;
    const start = page * perPage;
    const end = start + perPage;
    const threads = all.slice(start, end);
    return {
      threads,
      total: all.length,
      page,
      perPage: args.perPage === false ? false : perPage,
      hasMore: end < all.length,
    };
  }

  async saveThread({
    thread,
  }: {
    thread: StorageThreadType;
    memoryConfig?: MemoryConfigInternal;
  }): Promise<StorageThreadType> {
    this.threads.set(thread.id, thread);
    return thread;
  }

  async deleteThread(threadId: string): Promise<void> {
    this.threads.delete(threadId);
  }

  async cloneThread(args: StorageCloneThreadInput): Promise<StorageCloneThreadOutput> {
    const source = this.threads.get(args.sourceThreadId);
    if (!source) {
      throw new Error(
        `SynapMemory.cloneThread: source thread ${args.sourceThreadId} not found`,
      );
    }
    const newId = args.newThreadId ?? `${args.sourceThreadId}-clone-${Date.now()}`;
    const now = new Date();
    const clone: StorageThreadType = {
      id: newId,
      title: args.title ?? source.title,
      resourceId: args.resourceId ?? source.resourceId,
      createdAt: now,
      updatedAt: now,
      metadata: {
        ...(source.metadata ?? {}),
        ...(args.metadata ?? {}),
        sourceThreadId: args.sourceThreadId,
        clonedAt: now,
      },
    };
    this.threads.set(newId, clone);
    return {
      thread: clone,
      clonedMessages: [],
      messageIdMap: {},
    };
  }

  // ── working memory (not supported) ─────────────────────────────────────

  async getWorkingMemory(): Promise<string | null> {
    this.warnWorkingMemoryOnce();
    return null;
  }

  async getWorkingMemoryTemplate(): Promise<WorkingMemoryTemplate | null> {
    this.warnWorkingMemoryOnce();
    return null;
  }

  async updateWorkingMemory(): Promise<void> {
    this.warnWorkingMemoryOnce();
  }

  async __experimental_updateWorkingMemoryVNext(): Promise<{
    success: boolean;
    reason: string;
  }> {
    this.warnWorkingMemoryOnce();
    return {
      success: false,
      reason: "Working memory is not supported by SynapMemory in v0.1",
    };
  }

  // ── delete (Synap has no public delete API) ────────────────────────────

  async deleteMessages(_messageIds: MessageDeleteInput): Promise<void> {
    if (!this.deleteWarned) {
      // eslint-disable-next-line no-console
      console.warn(
        "SynapMemory.deleteMessages: Synap has no public delete API; this is a no-op. This warning fires once.",
      );
      this.deleteWarned = true;
    }
  }

  // ── helpers ────────────────────────────────────────────────────────────

  private emptyRecallResult() {
    return { messages: [], total: 0, page: 0, perPage: 0, hasMore: false };
  }

  private toMastraMessage(
    rm: SynapRecentMessage,
    threadId: string,
    resourceId?: string,
  ) {
    const role =
      rm.role === "user" || rm.role === "assistant" || rm.role === "system"
        ? rm.role
        : "user";
    return {
      id: rm.message_id,
      role,
      createdAt: new Date(rm.timestamp),
      threadId,
      resourceId,
      type: "text",
      content: {
        format: 2 as const,
        parts: [{ type: "text" as const, text: rm.content }],
      },
    };
  }

  private warnWorkingMemoryOnce() {
    if (!this.workingMemoryWarned) {
      // eslint-disable-next-line no-console
      console.warn(
        "SynapMemory: working memory is not supported in v0.1 (Synap has no working-memory primitive). Calls no-op.",
      );
      this.workingMemoryWarned = true;
    }
  }
}
