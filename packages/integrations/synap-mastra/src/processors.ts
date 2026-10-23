// The Synap processors Mastra's default Agent path actually calls.
//
// `new Agent({ memory })` does not read history through `MastraMemory.recall()`
// or `getSystemMessage()`, and it does not write through `saveMessages()`.
// Measured against @mastra/core 1.26 (`agent.generate` and `agent.stream` with a
// mock model): history loads through the `MessageHistory` INPUT processor, which
// calls `memory.storage.listMessages`; it persists through the same processor's
// OUTPUT hook, which calls `storage.saveMessages`; and the base class builds
// both processors in `getInputProcessors` / `getOutputProcessors`, which read
// `memory.storage` and throw "Memory requires a storage provider" when there is
// none. A memory that only implements `recall` / `saveMessages` was therefore
// never asked for anything.
//
// These are the processors `SynapMemory` returns from those two methods:
//
//   - input: Synap's long-term context becomes a system message, and Synap's
//     compaction replaces the turns the framework loaded before the current one
//     (`spliceHistory`). With no Mastra storage attached there is no framework
//     history at all, so Synap's own turns are the history.
//   - output: the turns of the run go to Synap through `recordTurn` (the stream,
//     opened if need be; REST when it cannot carry them), with the tool calls,
//     results and reasoning on the stream.
//
// Neither throws: a processor runs inside the Agent's own loop, and an error
// from telemetry or a failed read must not end the run.

import type { MastraDBMessage } from "@mastra/core/agent/message-list";

import {
  compactedHistoryFrom,
  isSynapHistory,
  spliceHistory,
} from "./stream-events.js";
import type { SynapSdkLike } from "./types.js";

/** Id of the system message `recall()` adds for the compaction block. */
export const COMPACTION_ID_PREFIX = "synap-compaction-";

/** Pulls the thread out of the way Mastra's own `MessageHistory` does. */
function threadOf(
  requestContext: unknown,
  messageList: { serialize(): { memoryInfo?: { threadId?: string; resourceId?: string } | null } },
): { threadId: string; resourceId?: string; readOnly: boolean } | null {
  const get = (requestContext as { get?: (k: string) => unknown } | null)?.get;
  const memory = typeof get === "function"
    ? (get.call(requestContext, "MastraMemory") as {
        thread?: { id?: string };
        resourceId?: string;
        memoryConfig?: { readOnly?: boolean };
      } | undefined)
    : undefined;
  if (memory?.thread?.id) {
    return {
      threadId: memory.thread.id,
      resourceId: memory.resourceId,
      readOnly: memory.memoryConfig?.readOnly === true,
    };
  }
  const info = messageList.serialize().memoryInfo;
  if (info?.threadId) {
    return { threadId: info.threadId, resourceId: info.resourceId, readOnly: false };
  }
  return null;
}

/** The text of a Mastra message, flattened the way `saveMessages` does. */
export function messageText(m: unknown): string {
  if (!m || typeof m !== "object") return "";
  const content = (m as { content?: unknown }).content;
  if (typeof content === "string") return content.trim();
  if (content && typeof content === "object") {
    const c = content as { parts?: unknown; text?: unknown; content?: unknown };
    if (Array.isArray(c.parts)) {
      const texts: string[] = [];
      for (const part of c.parts) {
        if (part && typeof part === "object" && (part as { type?: unknown }).type === "text") {
          const t = (part as { text?: unknown }).text;
          if (typeof t === "string") texts.push(t);
        }
      }
      return texts.join("").trim();
    }
    if (typeof c.text === "string") return c.text.trim();
    if (typeof c.content === "string") return c.content.trim();
  }
  return "";
}

export interface SynapInputProcessorOptions {
  sdk: SynapSdkLike;
  userId: string;
  customerId: string;
  mode: string;
  /** Whether Synap's long-term context is added as a system message. */
  injectSystemContext: boolean;
}

export class SynapInputProcessor {
  readonly id = "synap-context";
  readonly name = "SynapContext";

  constructor(private readonly options: SynapInputProcessorOptions) {}

  // The arguments are Mastra's `ProcessInputArgs`; typed loosely so a newer
  // @mastra/core adding fields does not break the build here.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async processInput(args: any): Promise<any> {
    const { messageList, requestContext } = args;
    try {
      const where = threadOf(requestContext, messageList);
      if (!where) return messageList;
      await this.addLongTermContext(messageList, where.threadId);
      await this.addHistory(messageList, where.threadId, where.resourceId);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("SynapMemory: could not add Synap context to the run:", (err as Error)?.message ?? err);
    }
    return messageList;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async addLongTermContext(messageList: any, threadId: string): Promise<void> {
    if (!this.options.injectSystemContext) return;
    try {
      const query = messageList.getLatestUserContent?.();
      const response = await this.options.sdk.fetch({
        conversation_id: threadId,
        user_id: this.options.userId,
        customer_id: this.options.customerId || null,
        // Omitted, never null: the real SDK rejects a non-array search_query.
        ...(typeof query === "string" && query.trim() ? { search_query: [query] } : {}),
        max_results: 20,
        mode: this.options.mode,
        include_conversation_context: false,
      });
      const text = (response.formatted_context ?? "").trim();
      if (text) messageList.addSystem(text, "synap-memory");
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(`SynapMemory: sdk.fetch failed threadId=${threadId}`, err);
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async addHistory(messageList: any, threadId: string, resourceId?: string): Promise<void> {
    let response: unknown;
    try {
      response = await this.options.sdk.conversation.context.get_context_for_prompt({
        conversation_id: threadId,
      });
    } catch (err) {
      // The framework's own history still stands.
      // eslint-disable-next-line no-console
      console.warn(`SynapMemory: compacted history unavailable threadId=${threadId}:`, (err as Error)?.message ?? err);
      return;
    }

    const remembered: MastraDBMessage[] = messageList.get.remembered.db();
    const input: MastraDBMessage[] = messageList.get.input.db();
    // The current turn: the last user message of the input, and what follows.
    let from = input.length;
    for (let i = input.length - 1; i >= 0; i -= 1) {
      if (input[i]!.role === "user") { from = i; break; }
    }
    const tail = input.slice(from);
    const stamp = tail[0]?.createdAt ? new Date(tail[0].createdAt).getTime() : Date.now();

    let counter = 0;
    const make = (role: string, text: string): MastraDBMessage => {
      counter += 1;
      return {
        id: `synap-history-${threadId}-${stamp}-${counter}`,
        role: role as MastraDBMessage["role"],
        // Strictly before the current turn, in order.
        createdAt: new Date(stamp - 1_000_000 + counter),
        threadId,
        resourceId,
        type: "text",
        content: { format: 2, parts: [{ type: "text", text }] },
      } as MastraDBMessage;
    };

    const history = compactedHistoryFrom(response);
    let spliced: MastraDBMessage[];
    if (history) {
      // Synap has compacted this conversation: its compaction stands in for the
      // framework's earlier turns, whatever they were.
      spliced = spliceHistory([...remembered, ...tail], history, {
        roleOf: (m) => m.role,
        textOf: messageText,
        makeMessage: make,
      });
    } else if (remembered.length === 0) {
      // No compaction, and the framework loaded nothing (no Mastra storage, or
      // a thread it has not seen): Synap's turns are the history.
      const turns = ((response as { recent_messages?: unknown })?.recent_messages ?? []) as Array<{
        role?: string; content?: unknown;
      }>;
      spliced = spliceHistory(tail, {
        preamble: "",
        turns: turns
          .filter((t) => (t.role === "user" || t.role === "assistant")
            && typeof t.content === "string" && t.content.trim() !== "")
          .map((t) => [t.role as "user" | "assistant", t.content as string] as const),
      }, { roleOf: (m) => m.role, textOf: messageText, makeMessage: make })
        // No compaction block, so no preamble to add.
        .filter((m) => !(m.role === "system" && messageText(m) === ""));
    } else {
      // The framework has a history and Synap has no compaction to replace it.
      return;
    }

    const head = spliced.slice(0, spliced.length - tail.length);
    if (history) messageList.removeByIds(remembered.map((m) => m.id));
    for (const m of head) {
      if (m.role === "system") {
        messageList.addSystem(messageText(m), "synap-history");
      } else {
        messageList.add(m, "memory");
      }
    }
  }
}

export interface SynapOutputProcessorOptions {
  /** Writes the run's messages to Synap; shared with `SynapMemory.saveMessages`. */
  persist: (messages: MastraDBMessage[], threadId: string) => Promise<unknown>;
}

export class SynapOutputProcessor {
  readonly id = "synap-record";
  readonly name = "SynapRecord";

  constructor(private readonly options: SynapOutputProcessorOptions) {}

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async processOutputResult(args: any): Promise<any> {
    const { messageList, requestContext } = args;
    try {
      const where = threadOf(requestContext, messageList);
      if (!where || where.readOnly) return messageList;
      const messages: MastraDBMessage[] = [
        ...messageList.get.input.db(),
        ...messageList.get.response.db(),
      ];
      if (messages.length > 0) await this.options.persist(messages, where.threadId);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("SynapMemory: could not record the run:", (err as Error)?.message ?? err);
    }
    return messageList;
  }
}

/** Whether a message is Synap's own history, not something somebody said. */
export function isHistoryMessage(m: unknown): boolean {
  const id = (m as { id?: unknown } | null)?.id;
  if (typeof id === "string" && (id.startsWith(COMPACTION_ID_PREFIX) || id.startsWith("synap-history-"))) {
    return true;
  }
  return isSynapHistory(messageText(m));
}
