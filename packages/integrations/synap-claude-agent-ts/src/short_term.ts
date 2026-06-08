// createSynapShortTermHook — TypeScript mirror of the Python
// create_synap_st_hook. Installs a UserPromptSubmit hook that fetches
// Synap *short-term* context (compacted summary + recent turns per
// conversation) and injects it via hookSpecificOutput.additionalContext.
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

import type { SynapSdkLike } from "./types.js";

const SUPPORTED_STYLES = ["structured", "narrative", "bullet_points"] as const;
export type SynapShortTermStyle = (typeof SUPPORTED_STYLES)[number];

const DEFAULT_PREAMBLE_OPEN = "<synap_short_term_context>";
const DEFAULT_PREAMBLE_CLOSE = "</synap_short_term_context>";

export type SynapShortTermOnError = "fallback" | "raise";

export interface CreateSynapShortTermHookOptions {
  sdk: SynapSdkLike;
  /** Synap conversation ID. Required + explicit. */
  conversationId: string;
  /** Defaults to 'narrative'. */
  style?: SynapShortTermStyle;
  /** Wrapping tags. Pass null for both to drop the wrapper. */
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
  const preambleOpen =
    options.preambleOpen === undefined ? DEFAULT_PREAMBLE_OPEN : options.preambleOpen;
  const preambleClose =
    options.preambleClose === undefined ? DEFAULT_PREAMBLE_CLOSE : options.preambleClose;

  if (!sdk.conversation?.context?.get_context_for_prompt) {
    throw new Error(
      "createSynapShortTermHook: sdk.conversation.context.get_context_for_prompt is missing. " +
        "Upgrade your Synap SDK to a version that exposes it, or pass a duck-typed mock for tests.",
    );
  }

  const onUserPromptSubmit: HookCallback = async (_input) => {
    // _input typed as HookInput union; we don't actually need its fields.
    let stBlock = "";
    try {
      const response = await sdk.conversation.context!.get_context_for_prompt({
        conversation_id: conversationId,
        style,
      });
      if (response.available) {
        stBlock = (response.formatted_context ?? "").trim();
      }
    } catch (err) {
      if (onError === "raise") throw err;
      console.warn(
        "[synap-claude-agent] createSynapShortTermHook: get_context_for_prompt failed:",
        (err as Error).message ?? err,
      );
      return {};
    }

    if (!stBlock) return {};

    const wrapped =
      preambleOpen && preambleClose
        ? `${preambleOpen}\n${stBlock}\n${preambleClose}`
        : stBlock;

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
