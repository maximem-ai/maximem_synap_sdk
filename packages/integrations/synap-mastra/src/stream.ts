// Report a Mastra run to Synap: the turns, and on the stream everything between.
//
// A Mastra Agent run produces more than the two ends of a turn. The reasoning,
// the tools the agent called and what they returned have no durable home
// anywhere else, and they are exactly what lets the anticipation agent predict
// the next turn. Without them every fetch is a cold retrieval.
//
// The messages Mastra hands over (`MastraDBMessage`s, from the output processor
// or from `saveMessages`) carry all five events in `content.parts`:
//
//   ===========================  ==========================================
//   part                          what it is
//   ===========================  ==========================================
//   role `user`, `text` part      the user's turn
//   role `assistant`, `text`      the assistant's turn
//   `reasoning` part              the model's reasoning for this step
//   `tool-invocation` part        a tool the model called
//   the same part, `state:result` what that tool returned
//   ===========================  ==========================================
//
// `toolInvocation.toolCallId` is Mastra's own id and is the same value on the
// call and on the result, so the two travel paired.
//
// Rules, from `stream-events.ts`.
//
// *The stream is the default.* The first report opens it, so a caller who never
// called `sdk.instance.listen()` still gets all five events. The turns
// themselves are written through `recordTurn`: stream first, REST when the
// stream cannot carry them, never both. Reasoning and tool events are
// stream-only. `SYNAP_AUTO_LISTEN=false` keeps the stream closed.
//
// *Never throws.* This runs inside the Agent's own save path, where an
// exception from telemetry would end the run. No context is worth that.

import {
  recordTurn,
  reportReasoning,
  reportToolCall,
  reportToolResult,
} from "./stream-events.js";

export interface SynapStreamOptions {
  /**
   * Whether tool results are reported. On by default, because a result is what
   * tells anticipation how the turn is actually going.
   *
   * ⚠ A tool result is usually your own customer's data. It is an anticipation
   * hint and never becomes a long-term memory, but it does leave your process.
   * Turn this off if that is not something you want to send.
   */
  reportToolResults?: boolean;
  /**
   * Whether `reasoning` parts are reported as reasoning steps. On by default.
   * Most providers return a summary rather than raw reasoning and some return
   * nothing; a turn without reasoning is an ordinary turn.
   */
  reportThoughts?: boolean;
}

/** What `saveMessages` already knows about the message it is about to write. */
export interface SynapStreamMessageInfo {
  /** Mastra's thread id, which is the Synap conversation id. */
  threadId: string;
  /** `user`, `assistant` or `system`. Only the first two are turns. */
  role: string;
  /** The message's flattened text, as `saveMessages` computed it. */
  text: string;
}

/**
 * How many reported events this remembers, so a re-save is not reported twice.
 *
 * Bounded because one `SynapMemory` can live for the whole process and serve
 * every thread in it.
 */
const REMEMBERED_EVENTS = 512;

/** A bounded set with oldest-out eviction. */
class RecentKeys {
  private readonly keys = new Set<string>();
  private readonly order: string[] = [];

  has(key: string): boolean {
    return this.keys.has(key);
  }

  add(key: string): void {
    if (this.keys.has(key)) return;
    this.keys.add(key);
    this.order.push(key);
    if (this.order.length > REMEMBERED_EVENTS) {
      const oldest = this.order.shift();
      if (oldest !== undefined) this.keys.delete(oldest);
    }
  }
}

/**
 * Turns the messages Mastra saves into Synap stream events.
 *
 * One instance per `SynapMemory`, because the de-duplication below is state.
 */
export class SynapMastraStreamReporter {
  private readonly sent = new RecentKeys();
  private step = 0;

  constructor(
    private readonly sdk: unknown,
    private readonly userId: string,
    private readonly customerId: string,
    private readonly options: SynapStreamOptions = {},
  ) {}

  /**
   * Record one message: its turn, and on the stream its parts.
   *
   * Returns whether the *turn* is now Synap's, written on the stream or over
   * REST (`recordTurn` picks, and never does both). Reasoning and tool events
   * never persist as conversation, so they never make this `true`.
   *
   * A turn is remembered once written, on either path, so Mastra re-saving a
   * message it has already saved does not write it again.
   */
  async report(message: unknown, info: SynapStreamMessageInfo): Promise<boolean> {
    // A system message is the instructions, not a turn, and the stream has no
    // event for one.
    if (info.role !== "user" && info.role !== "assistant") return false;
    // Every event needs a conversation to belong to.
    if (!info.threadId) return false;
    try {
      return await this.dispatch(message, info);
    } catch {
      // A tap must never break the Agent's run.
      return false;
    }
  }

  private async dispatch(
    message: unknown,
    info: SynapStreamMessageInfo,
  ): Promise<boolean> {
    const scope = {
      conversationId: info.threadId,
      userId: this.userId,
      customerId: this.customerId,
    };
    const messageId = readText(message, "id");

    // The parts come first, and the turn last, for two reasons. The order the
    // anticipation agent wants is "why, then what I did, then what I said",
    // and `assistant_message` is the event that wakes it: waking it before the
    // tool calls it is meant to reason about would defeat the point. It also
    // means a throw part-way through cannot have already sent the turn, so the
    // REST fallback stays correct.
    if (info.role === "assistant") {
      await this.reportParts(message, messageId, scope);
    }

    if (!info.text) return false;

    // Mastra re-saves a message it has already saved: filling a tool result
    // into an existing assistant message moves that message back to the
    // unsaved set, and it arrives here a second time with the same text. The
    // key carries the text so a genuinely changed message is still reported.
    //
    // No id means no key and no de-duplication, which is the honest answer:
    // two separate turns can say the same short thing, and nothing but an id
    // tells them apart.
    const key = messageId
      ? `turn:${messageId}:${info.role}:${fingerprint(info.text)}`
      : "";
    // Already written, so the server already has this turn.
    if (key && this.sent.has(key)) return true;

    const sent = await recordTurn(this.sdk, {
      ...scope,
      role: info.role,
      content: info.text,
    });
    // Only remembered once it actually went out, so a failed write is not
    // suppressed on a retry.
    if (sent && key) this.sent.add(key);
    return sent;
  }

  private async reportParts(
    message: unknown,
    messageId: string,
    scope: { conversationId: string; userId: string; customerId: string },
  ): Promise<void> {
    let index = -1;
    for (const part of readParts(message)) {
      index += 1;
      const type = readText(part, "type");

      if (type === "reasoning") {
        if (this.options.reportThoughts === false) continue;
        const content = readText(part, "reasoning").trim();
        if (!content) continue;
        const key = messageId ? `think:${messageId}:${index}` : "";
        if (key && this.sent.has(key)) continue;
        this.step += 1;
        const sent = await reportReasoning(this.sdk, {
          ...scope,
          content,
          stepIndex: this.step,
          thoughtType: "model_thought",
        });
        if (sent && key) this.sent.add(key);
        continue;
      }

      if (type !== "tool-invocation") continue;
      const invocation = readField(part, "toolInvocation");
      if (!invocation || typeof invocation !== "object") continue;

      const state = readText(invocation, "state");
      // The arguments of a partial call are still streaming in. Reporting one
      // would claim the model asked for something it had not finished asking.
      if (state === "partial-call") continue;

      // Mastra's own id for this invocation, and the same value comes back on
      // the result. An empty one is dropped rather than invented, because an
      // invented id pairs a call with a result that is not its own.
      const toolCallId = readText(invocation, "toolCallId");
      const toolName = readText(invocation, "toolName") || "tool";
      const pairKey = toolCallId || `part-${index}`;

      const callKey = messageId ? `call:${messageId}:${pairKey}` : "";
      if (!callKey || !this.sent.has(callKey)) {
        const sent = await reportToolCall(this.sdk, {
          ...scope,
          toolName,
          // `args` is the parsed input; `rawInput` is what Mastra keeps when it
          // could not parse them. Either is better than no arguments at all,
          // which is what a tool call with an empty `tool_args` claims.
          toolArgs: readField(invocation, "args") ?? readField(invocation, "rawInput"),
          toolCallId,
        });
        if (sent && callKey) this.sent.add(callKey);
      }

      if (this.options.reportToolResults === false) continue;
      const result = toolResult(invocation, state);
      if (result === NO_RESULT) continue;
      const resultKey = messageId ? `result:${messageId}:${pairKey}` : "";
      if (resultKey && this.sent.has(resultKey)) continue;
      const sent = await reportToolResult(this.sdk, {
        ...scope,
        result,
        toolName,
        toolCallId,
      });
      if (sent && resultKey) this.sent.add(resultKey);
    }
  }
}

// --------------------------------------------------------------- internals

/** Distinct from `undefined`, which is a tool result a tool can return. */
const NO_RESULT = Symbol("no-result");

/**
 * What a tool invocation returned, or `NO_RESULT` when it has not returned.
 *
 * Mastra widens the AI SDK's three states with its own. `output-error` holds
 * the failure text in `errorText` rather than in `result`, and a failure is
 * still something the tool said: anticipation reads a failed search very
 * differently from one that is still running.
 */
function toolResult(invocation: unknown, state: string): unknown {
  if (state === "result") return readField(invocation, "result");
  if (state === "output-error") {
    const errorText = readField(invocation, "errorText");
    return errorText === undefined ? readField(invocation, "result") : errorText;
  }
  return NO_RESULT;
}

function readField(value: unknown, key: string): unknown {
  if (!value || typeof value !== "object") return undefined;
  return (value as Record<string, unknown>)[key];
}

function readText(value: unknown, key: string): string {
  const found = readField(value, key);
  return typeof found === "string" ? found : "";
}

function readParts(message: unknown): unknown[] {
  const parts = readField(readField(message, "content"), "parts");
  return Array.isArray(parts) ? parts : [];
}

/**
 * A short stand-in for a message's text, so remembering 512 reported events
 * does not mean holding 512 conversations in memory.
 *
 * FNV-1a, with the length alongside it: two different texts of the same length
 * would have to collide in 32 bits *and* share a message id to be confused.
 */
function fingerprint(text: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return `${text.length}.${(hash >>> 0).toString(36)}`;
}
