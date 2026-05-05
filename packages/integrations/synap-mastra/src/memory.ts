// SynapMemory — a MastraMemory implementation backed by Synap.
//
// Drop-in for `new Agent({ memory: new SynapMemory({...}) })`. The methods
// that the common Agent flow reaches are backed by Synap:
//
//   - recall(): loads prior messages via sdk.conversation.context.get_context_for_prompt
//   - saveMessages(): persists each turn via sdk.conversation.record_message
//   - getSystemMessage(): fetches Synap context via sdk.fetch and returns it
//     as a preamble the Agent injects into the system prompt
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

import type {
  SynapSdkLike,
  SynapRecentMessage,
  SynapIdentityOptions,
} from "./types.js";

export interface SynapMemoryOptions extends SynapIdentityOptions {
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
 *   - read failures degrade gracefully (empty recall, null system message)
 *   - write failures throw so the Agent loop surfaces ingestion outages
 */
export class SynapMemory extends MastraMemory {
  private readonly sdk: SynapSdkLike;
  private readonly userId: string;
  private readonly customerId: string;
  private readonly mode: string;
  private readonly injectSystemContext: boolean;

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
        search_query: null,
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
   * Persist each message by relaying it through sdk.conversation.record_message.
   * Unknown/tool roles are skipped (Synap conversation records are
   * user/assistant/system only).
   */
  async saveMessages(args: {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    messages: any[];
    memoryConfig?: MemoryConfig | undefined;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    observabilityContext?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  }): Promise<any> {
    const out: unknown[] = [];
    for (const m of args.messages ?? []) {
      const role = typeof m?.role === "string" ? m.role : "";
      if (role !== "user" && role !== "assistant" && role !== "system") continue;

      const content = extractText(m);
      const threadId = typeof m?.threadId === "string" ? m.threadId : "";
      if (!content || !threadId) continue;

      await this.sdk.conversation.record_message({
        conversation_id: threadId,
        role,
        content,
        user_id: this.userId,
        customer_id: this.customerId,
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

function extractText(m: unknown): string {
  if (!m || typeof m !== "object") return "";
  const obj = m as Record<string, unknown>;
  const content = obj.content;
  if (typeof content === "string") return content.trim();
  if (content && typeof content === "object") {
    const c = content as Record<string, unknown>;
    if (Array.isArray(c.parts)) {
      const texts: string[] = [];
      for (const part of c.parts) {
        if (part && typeof part === "object" && (part as Record<string, unknown>).type === "text") {
          const t = (part as Record<string, unknown>).text;
          if (typeof t === "string") texts.push(t);
        }
      }
      return texts.join("").trim();
    }
    if (typeof c.text === "string") return c.text.trim();
  }
  return "";
}
