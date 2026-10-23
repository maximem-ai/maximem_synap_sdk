// createSynapShortTermHook — TypeScript mirror of the Python
// create_synap_st_hook. Installs a UserPromptSubmit hook that fetches
// Synap *short-term* context and injects it via
// hookSpecificOutput.additionalContext.
//
// What it injects is Synap's COMPACTION (the summary of the turns the
// conversation has outgrown, plus earlier turns history search found for the
// prompt), never the recent turns: Claude Code's own transcript already holds
// those, and `formatted_context` carries them too, so injecting it sent every
// recent turn twice. Before the first compaction there is nothing to add and
// the hook returns {}. A given compaction is injected once per hook: each
// injection is stored in the transcript with the prompt, so repeating it every
// prompt would grow the transcript by the same block each turn.
//
// Composes with createSynapHooks (LT). To get both LT + ST, register
// both hooks under UserPromptSubmit:
//
//     const stHooks = createSynapShortTermHook({ sdk, conversationId });
//     const ltHooks = createSynapHooks({ sdk, userId, conversationId });
//     const options = {
//       hooks: {
//         UserPromptSubmit: [
//           ...(ltHooks.UserPromptSubmit ?? []),
//           ...(stHooks.UserPromptSubmit ?? []),
//         ],
//       },
//     };
//
// Quality contract (matches the Python LangGraph adapter template):
//
//   - conversationId is required + explicit (no per-prompt session_id fallback;
//     ST is conversation-scoped, not prompt-scoped).
//   - The hook NEVER throws on SDK failure by default (onError="fallback"):
//     logs and returns {} so the agent proceeds with no extra context.
//     onError="raise" available for tests/strict environments.
//   - Empty ST is a no-op — returns {} rather than injecting blank context.

import type {
  HookCallback,
  HookCallbackMatcher,
  UserPromptSubmitHookInput,
} from "@anthropic-ai/claude-agent-sdk";

import { compactedHistoryFrom } from "./stream-events.js";
import type { SynapSdkLike } from "./types.js";

const SUPPORTED_STYLES = ["structured", "narrative", "bullet_points"] as const;
export type SynapShortTermStyle = (typeof SUPPORTED_STYLES)[number];

export type SynapShortTermOnError = "fallback" | "raise";

export interface CreateSynapShortTermHookOptions {
  sdk: SynapSdkLike;
  /** Synap conversation ID. Required + explicit. */
  conversationId: string;
  /** Defaults to 'narrative'. */
  style?: SynapShortTermStyle;
  /**
   * Extra wrapping tags around Synap's `<synap_conversation_history>` block.
   * None by default.
   */
  preambleOpen?: string | null;
  preambleClose?: string | null;
  /** Defaults to 'fallback' — hook never throws on SDK failure. */
  onError?: SynapShortTermOnError;
}

/**
 * Return a `hooks` dict that injects Synap short-term context into the
 * Claude Agent SDK's UserPromptSubmit pipeline.
 */
export function createSynapShortTermHook(
  options: CreateSynapShortTermHookOptions,
): Partial<Record<"UserPromptSubmit", HookCallbackMatcher[]>> {
  const { sdk, conversationId } = options;
  if (!sdk) {
    throw new Error("createSynapShortTermHook requires a non-null sdk");
  }
  if (!conversationId || !conversationId.trim()) {
    throw new Error(
      "createSynapShortTermHook requires a non-empty conversationId",
    );
  }
  const style: SynapShortTermStyle = options.style ?? "narrative";
  if (!SUPPORTED_STYLES.includes(style)) {
    throw new Error(
      `createSynapShortTermHook: unsupported style=${JSON.stringify(style)}; expected one of ${SUPPORTED_STYLES.join(", ")}`,
    );
  }
  const onError: SynapShortTermOnError = options.onError ?? "fallback";

  if (!sdk.conversation?.context?.get_context_for_prompt) {
    throw new Error(
      "createSynapShortTermHook: sdk.conversation.context.get_context_for_prompt is missing. " +
        "Upgrade your Synap SDK to a version that exposes it, or pass a duck-typed mock for tests.",
    );
  }

  // The compaction last injected: a different one (or a changed excerpt set)
  // is new information, the same one is already in the transcript.
  let injected = "";

  const onUserPromptSubmit: HookCallback = async (_input) => {
    let preamble = "";
    try {
      // `style` shapes `formatted_context`, which this hook no longer injects;
      // it is still sent so a caller's choice reaches the server unchanged.
      const response = await sdk.conversation.context!.get_context_for_prompt({
        conversation_id: conversationId,
        style,
      });
      preamble = compactedHistoryFrom(response)?.preamble ?? "";
    } catch (err) {
      if (onError === "raise") throw err;
      console.warn(
        "[synap-claude-agent] createSynapShortTermHook: get_context_for_prompt failed:",
        (err as Error).message ?? err,
      );
      return {};
    }

    if (!preamble || preamble === injected) return {};
    injected = preamble;

    // The default tags are Synap's own `<synap_conversation_history>`; custom
    // ones wrap them.
    const wrapped =
      options.preambleOpen && options.preambleClose
        ? `${options.preambleOpen}\n${preamble}\n${options.preambleClose}`
        : preamble;

    return {
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext: wrapped,
      },
    };
  };

  return {
    UserPromptSubmit: [{ hooks: [onUserPromptSubmit] }],
  };
}
