// createSynapHooks — TypeScript mirror of the Python create_synap_hooks.
//
// Installs a UserPromptSubmit hook that fetches Synap context for each
// prompt and injects it via hookSpecificOutput.additionalContext. Optionally
// records the user's prompt to Synap conversation history so future turns
// recall it.
//
// The hook NEVER throws: SDK failures log and fall through to {} (no
// context injected, no block on the agent). Context providers must not
// crash the agent loop.

import type {
  HookCallback,
  HookCallbackMatcher,
  UserPromptSubmitHookInput,
} from "@anthropic-ai/claude-agent-sdk";

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
   * When true (default), record the user's prompt to
   * sdk.conversation.record_message. Disable for injection-only semantics.
   */
  recordUserPrompts?: boolean;
}

const DEFAULT_CONTEXT_PREAMBLE =
  "<synap_memory>\n" +
  "Relevant context from the user's long-term memory:\n\n" +
  "{body}\n" +
  "</synap_memory>";

export function createSynapHooks(
  options: CreateSynapHooksOptions,
): Partial<Record<"UserPromptSubmit", HookCallbackMatcher[]>> {
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

  return {
    UserPromptSubmit: [
      {
        hooks: [onUserPromptSubmit],
      },
    ],
  };
}
