// Synap short-term context for eve agents.
//
// eve's own durability (Vercel Workflows event-log replay) persists a single
// session's turn state; it is not cross-session recall and it does not surface
// Synap's compacted short-term summary. This module injects that summary into
// the system prompt as a first-class eve extension point.
//
// `createSynapInstructions` returns a `defineDynamic` resolver for
// `agent/instructions/`, evaluated at `turn.started`. It *augments* the system
// prompt (eve lowers each `defineInstructions` to one extra `{ role: "system" }`
// message) — it never replaces `instructions.md` or the user's turn.
//
//   // agent/instructions/synap.ts
//   import { createSynapInstructions } from "@maximem/synap-eve";
//   import { sdk } from "../lib/synap.js";
//   export default createSynapInstructions({ sdk });
//
// Error policy: reads degrade. On no content, missing conversation id, or an
// SDK failure with the default `onError: "fallback"`, the resolver returns
// `null`, which eve treats as "contribute nothing" — the turn proceeds.

import { defineDynamic } from "eve/tools";
import { defineInstructions } from "eve/instructions";

import {
  resolveConversationId,
  type EveSessionLike,
  type SynapSdkLike,
} from "./types.js";

const SUPPORTED_STYLES = ["structured", "narrative", "bullet_points"] as const;
export type SynapShortTermStyle = (typeof SUPPORTED_STYLES)[number];

export type SynapShortTermOnError = "fallback" | "raise";

const DEFAULT_PREAMBLE_OPEN = "<synap_short_term_context>";
const DEFAULT_PREAMBLE_CLOSE = "</synap_short_term_context>";

export interface SynapShortTermResult {
  /** The short-term string ready to splice into a system prompt. Empty if none. */
  formattedContext: string;
  /** Whether the server reported any short-term context (compaction or recent turns). */
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
 * Returns `{ formattedContext: '', available: false }` on no content, and also
 * on SDK failure with the default `onError: 'fallback'`.
 *
 * @throws if `conversationId` is empty or `style` is unknown.
 * @throws the underlying SDK error when `onError: 'raise'`.
 */
export async function fetchSynapShortTerm(
  opts: FetchSynapShortTermOptions,
): Promise<SynapShortTermResult> {
  const { sdk, conversationId } = opts;
  if (!sdk) throw new TypeError("fetchSynapShortTerm requires a non-null sdk");
  if (!conversationId || !conversationId.trim()) {
    throw new TypeError("fetchSynapShortTerm requires a non-empty conversationId");
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
    if (!response.available) return { formattedContext: "", available: false };
    return {
      formattedContext: (response.formatted_context ?? "").trim(),
      available: response.available,
    };
  } catch (err) {
    if (onError === "raise") throw err;
    console.warn("[synap-eve] fetchSynapShortTerm failed:", (err as Error).message ?? err);
    return { formattedContext: "", available: false };
  }
}

export interface BuildSynapShortTermMarkdownOptions {
  /** Wrapping tags. Pass `null` to drop the wrapper. */
  preambleOpen?: string | null;
  preambleClose?: string | null;
}

/**
 * Wrap a short-term result as markdown for a `defineInstructions` message.
 * Returns the empty string when there is no context — the caller emits `null`.
 */
export function buildSynapShortTermMarkdown(
  result: SynapShortTermResult,
  opts: BuildSynapShortTermMarkdownOptions = {},
): string {
  const body = (result.formattedContext ?? "").trim();
  if (!body) return "";
  const open = opts.preambleOpen === undefined ? DEFAULT_PREAMBLE_OPEN : opts.preambleOpen;
  const close = opts.preambleClose === undefined ? DEFAULT_PREAMBLE_CLOSE : opts.preambleClose;
  return open && close ? `${open}\n${body}\n${close}` : body;
}

export interface SynapInstructionsOptions extends BuildSynapShortTermMarkdownOptions {
  sdk: SynapSdkLike;
  /** Explicit conversation id. Defaults to the eve session id (`ctx.session.id`). */
  conversationId?: string;
  style?: SynapShortTermStyle;
  /** Defaults to 'fallback' — a recall failure degrades to no injected context. */
  onError?: SynapShortTermOnError;
}

/**
 * Build the per-turn Synap short-term instructions resolver for
 * `agent/instructions/`. Evaluated at `turn.started`; returns
 * `defineInstructions({ markdown })` when there is context, else `null`.
 */
export function createSynapInstructions(options: SynapInstructionsOptions) {
  const { sdk } = options;
  if (!sdk) throw new TypeError("createSynapInstructions requires a non-null sdk");

  const resolve = async (_event: unknown, ctx: EveSessionLike) => {
    const conversationId = resolveConversationId(options.conversationId, ctx);
    if (!conversationId) return null;
    const result = await fetchSynapShortTerm({
      sdk,
      conversationId,
      style: options.style,
      onError: options.onError,
    });
    const markdown = buildSynapShortTermMarkdown(result, {
      preambleOpen: options.preambleOpen,
      preambleClose: options.preambleClose,
    });
    return markdown ? defineInstructions({ markdown }) : null;
  };

  return defineDynamic({
    events: {
      "turn.started": resolve,
    },
  });
}
