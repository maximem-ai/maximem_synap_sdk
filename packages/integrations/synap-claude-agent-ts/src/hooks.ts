// createSynapHooks — TypeScript mirror of the Python create_synap_hooks.
//
// Installs a UserPromptSubmit hook that fetches Synap context for each
// prompt and injects it via hookSpecificOutput.additionalContext. Optionally
// records the user's prompt to Synap conversation history so future turns
// recall it.
//
// It also reports the turn to the live gRPC stream, along with each tool call,
// each tool result, the assistant's reply and the end of the session. See
// "What reaches Synap, and by which path" below.
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
// **Stream first, REST only as a fallback, never both.** The server persists a
// turn it receives on the stream by itself, so calling `record_message` after
// a successful stream report writes it twice. A user who never calls
// `sdk.instance.listen()` sees behaviour byte-identical to before: every
// helper here is a no-op without a stream.

import type {
  HookCallback,
  HookCallbackMatcher,
  PostToolUseHookInput,
  PostToolUseFailureHookInput,
  PreToolUseHookInput,
  StopHookInput,
  UserPromptSubmitHookInput,
} from "@anthropic-ai/claude-agent-sdk";

import {
  endSession,
  reportToolCall,
  reportToolResult,
  reportTurn,
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
   * When true (default), record conversation turns to Synap. Disable for
   * injection-only semantics.
   *
   * ⚠ This now gates the ASSISTANT turn as well as the user's prompt, because
   * both are persisted server-side and it would be strange for one to be
   * recorded and the other not. The assistant turn only ever goes out over the
   * stream: there was no REST write for it before and adding one would start
   * persisting data an existing caller never asked for.
   *
   * Tool calls and tool results are not gated by this. They are
   * anticipation-only, never persisted, and impossible to send without a
   * stream the caller opened on purpose.
   */
  recordUserPrompts?: boolean;
}

/** Every hook event this integration installs. */
type SynapHookEvent =
  | "UserPromptSubmit"
  | "PreToolUse"
  | "PostToolUse"
  | "PostToolUseFailure"
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

    if (recordUserPrompts && convId) {
      // Stream first. The server persists a turn it receives on the stream
      // itself, so a record_message after a successful report writes it twice.
      const sent = await reportTurn(sdk, {
        role: "user",
        content: prompt,
        conversationId: convId,
        userId,
        customerId,
      });
      if (!sent) {
        try {
          await sdk.conversation.record_message({
            conversation_id: convId,
            role: "user",
            content: prompt,
            user_id: userId,
            customer_id: customerId,
          });
        } catch (err) {
          // Never throw — write-side in a hook still honors callback contract.
          // eslint-disable-next-line no-console
          console.error(
            `synap_claude_agent.UserPromptSubmit: record_message failed convId=${convId}`,
            err,
          );
        }
      }
    }

    if (!formatted) {
      return {};
    }

    return {
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit" as const,
        additionalContext: contextPreamble.replace("{body}", formatted),
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
    if (!recordUserPrompts || !content.trim()) {
      return {};
    }
    await reportTurn(sdk, {
      role: "assistant",
      content,
      conversationId: convIdOf(i),
      userId,
      customerId,
    });
    return {};
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
    Stop: [{ hooks: [onStop] }],
    SessionEnd: [{ hooks: [onSessionEnd] }],
  };
}
