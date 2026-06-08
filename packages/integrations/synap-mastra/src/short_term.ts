// Synap short-term context for Mastra agents.
//
// `SynapMemory` already pipes ST through `recall()` for users who adopt
// it as their full memory backend. This module is the **standalone**
// path: drop ST into the system prompt of any Mastra Agent without
// committing to the full SynapMemory backend.
//
// Quality contract mirrors the Python LangGraph adapter template:
//
//   - conversationId is required + explicit.
//   - SDK failures NEVER crash the agent by default (onError="fallback"):
//     logged, returns empty string. onError="raise" propagates.
//   - Empty short-term result (no compaction yet AND no recent turns)
//     is a no-op — never wipes the user's system prompt.
//   - Preamble tags are overridable; pass null for both to emit raw.

import type { SynapSdkLike } from "./types.js";

const SUPPORTED_STYLES = ["structured", "narrative", "bullet_points"] as const;
export type SynapShortTermStyle = (typeof SUPPORTED_STYLES)[number];

const DEFAULT_PREAMBLE_OPEN = "<synap_short_term_context>";
const DEFAULT_PREAMBLE_CLOSE = "</synap_short_term_context>";

export type SynapShortTermOnError = "fallback" | "raise";

export interface SynapShortTermResult {
  /** The ST string ready to splice into a system prompt. Empty if none. */
  formattedContext: string;
  /** Whether the server reported any ST (compaction or recent turns). */
  available: boolean;
}

export interface FetchSynapShortTermOptions {
  sdk: SynapSdkLike;
  conversationId: string;
  style?: SynapShortTermStyle;
  /** Defaults to 'fallback'. */
  onError?: SynapShortTermOnError;
}

/**
 * Fetch Synap short-term context for `conversationId`.
 *
 * Returns `{ formattedContext: '', available: false }` on no-content
 * AND on SDK failure with the default `onError: 'fallback'`.
 *
 * @throws if `conversationId` is empty or `style` is unknown.
 * @throws underlying SDK error when `onError: 'raise'`.
 */
export async function fetchSynapShortTerm(
  opts: FetchSynapShortTermOptions,
): Promise<SynapShortTermResult> {
  const { sdk, conversationId } = opts;
  if (!sdk) {
    throw new TypeError("fetchSynapShortTerm requires a non-null sdk");
  }
  if (!conversationId || !conversationId.trim()) {
    throw new TypeError(
      "fetchSynapShortTerm requires a non-empty conversationId",
    );
  }
  const style: SynapShortTermStyle = opts.style ?? "narrative";
  if (!SUPPORTED_STYLES.includes(style)) {
    throw new TypeError(
      `fetchSynapShortTerm: unsupported style=${JSON.stringify(style)}; expected one of ${SUPPORTED_STYLES.join(", ")}`,
    );
  }
  const onError: SynapShortTermOnError = opts.onError ?? "fallback";

  try {
    const response = await sdk.conversation.context.get_context_for_prompt({
      conversation_id: conversationId,
      style,
    });
    if (!response.available) {
      return { formattedContext: "", available: false };
    }
    const formatted = (response.formatted_context ?? "").trim();
    return { formattedContext: formatted, available: response.available };
  } catch (err) {
    if (onError === "raise") throw err;
    console.warn(
      "[synap-mastra] fetchSynapShortTerm failed:",
      (err as Error).message ?? err,
    );
    return { formattedContext: "", available: false };
  }
}

export interface BuildSynapShortTermSystemOptions {
  /** Caller's own system prompt; ST is prepended above. */
  system?: string;
  /** Wrapping tags. Pass `null` to drop the wrapper. */
  preambleOpen?: string | null;
  preambleClose?: string | null;
}

/**
 * Compose the ST result + caller's static system text into a single
 * string ready to pass into `Agent({ instructions: ... })` or
 * `agent.generate({ instructions: ... })`.
 *
 * Returns the empty string when both inputs are empty; the user's
 * `system` is preserved unchanged when ST is unavailable.
 */
export function buildSynapShortTermSystem(
  result: SynapShortTermResult,
  opts: BuildSynapShortTermSystemOptions = {},
): string {
  const parts: string[] = [];
  const body = (result.formattedContext ?? "").trim();
  const userSystem = (opts.system ?? "").trim();
  if (body) {
    const open =
      opts.preambleOpen === undefined ? DEFAULT_PREAMBLE_OPEN : opts.preambleOpen;
    const close =
      opts.preambleClose === undefined ? DEFAULT_PREAMBLE_CLOSE : opts.preambleClose;
    if (open && close) {
      parts.push(`${open}\n${body}\n${close}`);
    } else {
      parts.push(body);
    }
  }
  if (userSystem) parts.push(userSystem);
  return parts.join("\n\n");
}

/**
 * One-call convenience: fetch ST and return a combined system-prompt
 * string. Use this when you want the simplest possible integration in
 * a Mastra agent invocation:
 *
 *     const instructions = await synapShortTermInstructions({
 *       sdk,
 *       conversationId: "conv_abc",
 *       system: "You are a helpful assistant.",
 *     });
 *     const result = await agent.generate(message, { instructions });
 */
export async function synapShortTermInstructions(
  opts: FetchSynapShortTermOptions & BuildSynapShortTermSystemOptions,
): Promise<string> {
  const result = await fetchSynapShortTerm({
    sdk: opts.sdk,
    conversationId: opts.conversationId,
    style: opts.style,
    onError: opts.onError,
  });
  return buildSynapShortTermSystem(result, {
    system: opts.system,
    preambleOpen: opts.preambleOpen,
    preambleClose: opts.preambleClose,
  });
}
