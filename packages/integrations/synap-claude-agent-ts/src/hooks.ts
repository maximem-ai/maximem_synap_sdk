// createSynapHooks — TypeScript mirror of the Python create_synap_hooks.
//
// Installs a UserPromptSubmit hook that fetches Synap context for each
// prompt and injects it via hookSpecificOutput.additionalContext. Optionally
// records the user's prompt to Synap conversation history so future turns
// recall it.
//
// It also reports the turn to the live gRPC stream, along with each tool call,
// each tool result, the assistant's reply and the end of the session. See
// "What reaches Synap, and by which path" below. The stream opens itself on the
// first report; `SYNAP_AUTO_LISTEN=false` keeps it closed.
//
// The hooks NEVER throw: SDK failures log and fall through to {} (no
// context injected, no block on the agent). Context providers must not
// crash the agent loop.
//
// ## What reaches Synap, and by which path
//
// Synap's `Listen` stream carries five per-turn events. Only `user_message`
// and `assistant_message` are persisted as conversation and feed long-term
// extraction; `tool_call`, `tool_result` and `agent_thinking` are
// anticipation-only and have no other durable home.
//
// | Synap event         | Hook                          | Field                   |
// | ------------------- | ----------------------------- | ----------------------- |
// | `user_message`      | UserPromptSubmit              | `prompt`                |
// | `tool_call`         | PreToolUse                    | `tool_input`, `tool_use_id` |
// | `tool_result`       | PostToolUse                   | `tool_response`         |
// | `tool_result`       | PostToolUseFailure            | `error`                 |
// | `assistant_message` | Stop                          | `last_assistant_message` |
// | end of session      | SessionEnd                    | —                       |
//
// ⚠ `agent_thinking` is NOT wired, because the Claude Agent SDK exposes no
// hook carrying the model's reasoning text. `HookEvent` has 27 members and
// none of them has a thinking or reasoning field; the only `thinking` in the
// SDK's types is the setting that turns it on. Reporting something else in its
// place would be inventing a signal, so the integration reports four of the
// five events and says so.
//
// **Stream first, REST only as a fallback, never both.** `recordTurn` opens the
// stream if there is none and sends the turn there; the server persists a turn
// it receives on the stream by itself, so calling `record_message` after a
// successful stream report would write it twice. Only when the stream cannot
// open (or `SYNAP_AUTO_LISTEN=false`) does a turn go over REST. Tool calls and
// results are stream-only.
//
// ## Synap's compaction, and what Claude Code does not let us replace
//
// Claude Code owns the transcript: the CLI subprocess keeps the message list,
// decides when to compact it and writes the summary itself. The hook surface
// cannot replace any of that:
//
//   - `PreCompactHookInput` (trigger, custom_instructions) and
//     `PostCompactHookInput` (compact_summary) are input only. `HookJSONOutput`
//     has no `hookSpecificOutput` member for `PreCompact` or `PostCompact`, so
//     a hook can neither supply the summary nor veto the compaction.
//   - `hookSpecificOutput.additionalContext` ADDS a message; nothing in
//     `SyncHookJSONOutput` removes or rewrites an earlier one.
//
// What can be done, and is: Synap's compaction (`fetchCompactedHistory`) is
// added as `additionalContext`, once per compaction, on the first prompt and
// whenever Synap compacts again; and `SessionStart` with `source: "compact"`
// adds it again right after the CLI replaced its history with its own summary,
// because that discards everything added before. Only the compaction block goes
// in, never the turns after it: the CLI's transcript already holds those.
//
// Measured against the real CLI (see the probe in this package's tests):
// a `SessionStart` callback fires for `compact` and never for `startup`. The
// CLI runs its startup hooks before the SDK has registered callbacks.
//
import type {
  HookCallback,
  HookCallbackMatcher,
  PostToolUseHookInput,
  PostToolUseFailureHookInput,
  PreToolUseHookInput,
  SessionStartHookInput,
  StopHookInput,
  UserPromptSubmitHookInput,
} from "@anthropic-ai/claude-agent-sdk";

import {
  endSession,
  fetchCompactedHistory,
  isSynapHistory,
  recordTurn,
  reportToolCall,
  reportToolResult,
} from "./stream-events.js";
import type { SynapIdentityOptions } from "./types.js";

export interface CreateSynapHooksOptions extends SynapIdentityOptions {
  /** Synap fetch mode; "accurate" (default) or "fast". */
  mode?: string;
  /** Cap on Synap fetch results. */
  maxResults?: number;
  /**
   * Format string with one `{body}` placeholder wrapping the fetched
   * context. Not applied when the fetched context is empty.
   */
  contextPreamble?: string;
  /**
   * When true (default), record the user's prompts to Synap. Disable to keep
   * prompts out of Synap's conversation record.
   *
   * This gates the user's prompt only. The assistant's reply (the `Stop` hook)
   * is always recorded: it is the event that tells the anticipation agent a
   * turn has ended, and `recordUserPrompts: false` used to switch that off too.
   *
   * Tool calls and tool results are not gated by this. They are
   * anticipation-only and never persisted.
   */
  recordUserPrompts?: boolean;
}

/** Every hook event this integration installs. */
type SynapHookEvent =
  | "UserPromptSubmit"
  | "PreToolUse"
  | "PostToolUse"
  | "PostToolUseFailure"
  | "SessionStart"
  | "Stop"
  | "SessionEnd";

const DEFAULT_CONTEXT_PREAMBLE =
  "<synap_memory>\n" +
  "Relevant context from the user's long-term memory:\n\n" +
  "{body}\n" +
  "</synap_memory>";

export function createSynapHooks(
  options: CreateSynapHooksOptions,
): Partial<Record<SynapHookEvent, HookCallbackMatcher[]>> {
  const {
    sdk,
    userId,
    customerId = "",
    conversationId,
    mode = "accurate",
    maxResults = 20,
    contextPreamble = DEFAULT_CONTEXT_PREAMBLE,
    recordUserPrompts = true,
  } = options;

  if (!sdk) {
    throw new Error("createSynapHooks requires a non-null sdk");
  }
  if (!userId) {
    throw new Error("createSynapHooks requires a non-empty userId");
  }

  /**
   * The compaction last put in front of the model, so a compaction is injected
   * once, not on every prompt: each injection is stored in the transcript with
   * the prompt, so repeating it would grow the transcript by the same block
   * every turn.
   */
  let injectedCompaction = "";

  /** Synap's compaction block for this conversation, or "" if there is none. */
  const compactionFor = async (convId: string, force: boolean): Promise<string> => {
    const history = await fetchCompactedHistory(sdk, convId);
    if (!history) return "";
    const marker = history.compactionId ?? history.preamble;
    if (!force && marker === injectedCompaction) return "";
    injectedCompaction = marker;
    return history.preamble;
  };

  const onUserPromptSubmit: HookCallback = async (input) => {
    const i = input as UserPromptSubmitHookInput;
    const prompt = typeof i.prompt === "string" ? i.prompt : "";
    if (!prompt.trim()) {
      return {};
    }

    const convId = conversationId ?? i.session_id ?? "";

    let formatted = "";
    try {
      const response = await sdk.fetch({
        conversation_id: convId || null,
        user_id: userId,
        customer_id: customerId || null,
        search_query: [prompt],
        max_results: maxResults,
        mode,
        include_conversation_context: false,
      });
      formatted = (response.formatted_context ?? "").trim();
    } catch (err) {
      // Read degrades gracefully — log, skip injection, do not throw.
      // eslint-disable-next-line no-console
      console.error(
        `synap_claude_agent.UserPromptSubmit: sdk.fetch failed userId=${userId} convId=${convId}`,
        err,
      );
    }

    // Not a turn: a hook that echoes Synap's own history back must not have it
    // recorded as something the user said.
    if (recordUserPrompts && convId && !isSynapHistory(prompt)) {
      await recordTurn(sdk, {
        role: "user",
        content: prompt,
        conversationId: convId,
        userId,
        customerId,
      });
    }

    // Synap's compaction replaces nothing here (see the header), so it goes in
    // beside what the CLI holds.
    const compaction = convId ? await compactionFor(convId, false) : "";
    const context = [
      compaction,
      formatted ? contextPreamble.replace("{body}", formatted) : "",
    ].filter(Boolean).join("\n\n");
    if (!context) {
      return {};
    }

    return {
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit" as const,
        additionalContext: context,
      },
    };
  };

  /** The conversation this hook call belongs to. */
  const convIdOf = (i: { session_id?: string }): string =>
    conversationId ?? i.session_id ?? "";

  const onPreToolUse: HookCallback = async (input) => {
    const i = input as PreToolUseHookInput;
    await reportToolCall(sdk, {
      toolName: i.tool_name,
      toolArgs: i.tool_input,
      // The SDK's own id for this call. The result below reads the same field,
      // so the pair cannot drift.
      toolCallId: i.tool_use_id,
      conversationId: convIdOf(i),
      userId,
      customerId,
    });
    return {};
  };

  const onPostToolUse: HookCallback = async (input) => {
    const i = input as PostToolUseHookInput;
    await reportToolResult(sdk, {
      result: i.tool_response,
      toolName: i.tool_name,
      toolCallId: i.tool_use_id,
      conversationId: convIdOf(i),
      userId,
      customerId,
    });
    return {};
  };

  const onPostToolUseFailure: HookCallback = async (input) => {
    // A tool that failed still told the agent something, and what it told it
    // is usually why the next turn goes the way it does. Reporting only the
    // successes would hide exactly the turns anticipation most needs.
    const i = input as PostToolUseFailureHookInput;
    await reportToolResult(sdk, {
      result: { error: i.error, is_interrupt: i.is_interrupt ?? false },
      toolName: i.tool_name,
      toolCallId: i.tool_use_id,
      conversationId: convIdOf(i),
      userId,
      customerId,
    });
    return {};
  };

  const onStop: HookCallback = async (input) => {
    const i = input as StopHookInput;
    // ⚠ The event that WAKES the anticipation agent. Without it the stream
    // carries the whole turn and nothing ever acts on it.
    //
    // `last_assistant_message` is optional in the SDK's own types. When it is
    // absent there is no assistant text to report and nothing is sent: an
    // empty assistant turn would be a worse signal than none.
    const content = typeof i.last_assistant_message === "string"
      ? i.last_assistant_message : "";
    if (!content.trim()) {
      return {};
    }
    // `recordTurn`, not `reportTurn`: the reply is part of the conversation and
    // goes over REST when the stream cannot carry it.
    await recordTurn(sdk, {
      role: "assistant",
      content,
      conversationId: convIdOf(i),
      userId,
      customerId,
    });
    return {};
  };

  const onSessionStart: HookCallback = async (input) => {
    const i = input as SessionStartHookInput;
    // The CLI just replaced its history with its own summary; what was added
    // before is gone with it.
    if (i.source !== "compact") return {};
    const preamble = await compactionFor(convIdOf(i), true);
    if (!preamble) return {};
    return {
      hookSpecificOutput: {
        hookEventName: "SessionStart" as const,
        additionalContext: preamble,
      },
    };
  };

  const onSessionEnd: HookCallback = async (input) => {
    await endSession(sdk, convIdOf(input as { session_id?: string }));
    return {};
  };

  return {
    UserPromptSubmit: [{ hooks: [onUserPromptSubmit] }],
    PreToolUse: [{ hooks: [onPreToolUse] }],
    PostToolUse: [{ hooks: [onPostToolUse] }],
    PostToolUseFailure: [{ hooks: [onPostToolUseFailure] }],
    SessionStart: [{ hooks: [onSessionStart] }],
    Stop: [{ hooks: [onStop] }],
    SessionEnd: [{ hooks: [onSessionEnd] }],
  };
}
