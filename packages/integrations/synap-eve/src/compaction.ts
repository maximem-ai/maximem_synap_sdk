// Synap's compaction in place of eve's, at the one place eve lets it in.
//
// eve compacts when the conversation passes `compaction.thresholdPercent` of the
// model's context window (default 0.9): `compactMessages` in eve's harness
// splits the history into older and recent turns and calls
//
//     generateText({ model: compactionModel, system, prompt })
//
// on the older ones, then replaces them with one user message
// ("Summary of our conversation so far:") and one assistant message holding the
// model's text (the checkpoint), followed by the recent turns. The model is
// `defineAgent({ compaction: { model } })`, which is "an AI Gateway model id or
// any AI SDK-compatible language model", and the turn model when unset.
//
// That is a seam: a model that answers the compaction call with Synap's
// compaction instead of writing a summary. eve then stores Synap's compaction as
// its checkpoint, and every later turn sees it.
//
//   // agent/agent.ts
//   import { defineAgent } from "eve";
//   import { createSynapCompactionModel } from "@maximem/synap-eve";
//   import { sdk } from "./lib/synap.js";
//
//   const model = openai("gpt-5");
//   export default defineAgent({
//     model,
//     compaction: {
//       model: createSynapCompactionModel({ sdk, model, conversationId: "support-alice" }),
//     },
//   });
//
// What this cannot do, from eve 0.25's source:
//
//   - It does not make eve compact. eve decides when (token estimate against
//     `thresholdPercent`), and until it does its own turn-by-turn history is
//     what the model sees. Lower `thresholdPercent` to hand over sooner.
//   - eve's hooks (`agent/hooks/*.ts`, `compaction.requested` and
//     `compaction.completed` included) observe events after they are durably
//     recorded and return nothing, so they cannot supply or veto a summary, and
//     this seam is configured in `agent.ts`, not installed by `createSynapStreamHooks`.
//   - The model call carries no session id, so the Synap conversation has to be
//     named here. Pass the same `conversationId` you pass to the hooks, or a
//     function that returns it.
//
// Everything else about the model, including streaming and every other call, is
// the model you pass in. When Synap has no compaction for the conversation (or
// cannot be reached) the call goes to that model and eve's own summary is used.

import { fetchCompactedHistory } from "./stream-events.js";

/** What this needs of an AI SDK language model: its own generate call. */
interface LanguageModelLike {
  readonly specificationVersion?: string;
  doGenerate(options: { prompt: unknown }): PromiseLike<unknown>;
}

export interface SynapCompactionModelOptions<M extends object> {
  /** A Synap SDK with `conversation.context.get_context_for_prompt`. */
  sdk: unknown;
  /**
   * The model that summarizes when Synap has nothing to give. Use the model eve
   * would have used (the agent's own, when `compaction.model` is unset). The
   * result behaves exactly like this model for every call but the compaction.
   */
  model: M;
  /**
   * The Synap conversation. A model call carries no eve session id, so name it:
   * the `conversationId` you gave `createSynapStreamHooks` and
   * `createSynapInstructions`, or a function returning it. Nothing resolved
   * means eve's own summary.
   */
  conversationId: string | (() => string | undefined);
}

/** The first line of eve's compaction system prompt. */
const EVE_COMPACTION_MARKER = "CONTEXT CHECKPOINT COMPACTION";

function systemText(prompt: unknown): string {
  if (!Array.isArray(prompt)) return "";
  return prompt
    .filter((m) => (m as { role?: unknown })?.role === "system")
    .map((m) => {
      const content = (m as { content?: unknown }).content;
      return typeof content === "string" ? content : "";
    })
    .join("\n");
}

/** All the text of a prompt, for finding which turns it already contains. */
function promptText(prompt: unknown): string {
  if (!Array.isArray(prompt)) return "";
  const out: string[] = [];
  for (const m of prompt) {
    const content = (m as { content?: unknown })?.content;
    if (typeof content === "string") out.push(content);
    else if (Array.isArray(content)) {
      for (const part of content) {
        const text = (part as { text?: unknown })?.text;
        if (typeof text === "string") out.push(text);
      }
    }
  }
  return out.join("\n");
}

const norm = (text: string): string => text.split(/\s+/).filter(Boolean).join(" ");

/**
 * The checkpoint text: Synap's compaction, plus the turns after it that sit in
 * the part of the conversation eve is summarizing.
 *
 * eve keeps its own recent turns after the checkpoint, so putting every Synap
 * turn into the checkpoint would show those twice. Synap's turns are matched
 * against the transcript eve sent for summary, by the start of their text (eve
 * truncates long messages in that transcript), and only the ones found are
 * carried: they are exactly the turns the checkpoint replaces and the
 * compaction does not cover.
 */
async function checkpointFor(sdk: unknown, conversationId: string, prompt: unknown): Promise<string | null> {
  const history = await fetchCompactedHistory(sdk, conversationId);
  if (!history) return null;
  const transcript = norm(promptText(prompt));
  const covered = history.turns.filter(([, text]) => {
    const head = norm(text).slice(0, 60);
    return head !== "" && transcript.includes(head);
  });
  return [history.preamble, ...covered.map(([role, text]) => `### ${role}\n${text.trim()}`)].join("\n\n");
}

/** A generate result in the shape the model's specification version expects. */
function resultFor(version: string | undefined, text: string): Record<string, unknown> {
  if (version === "v2") {
    return {
      content: [{ type: "text", text }],
      finishReason: "stop",
      usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
      warnings: [],
    };
  }
  // v3 and v4 share this shape.
  return {
    content: [{ type: "text", text }],
    finishReason: { unified: "stop", raw: "synap-compaction" },
    usage: {
      inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 0, text: 0, reasoning: 0 },
    },
    warnings: [],
  };
}

/**
 * A language model that answers eve's compaction call with Synap's compaction.
 *
 * Returns the model you pass in, wrapped: every property and method is that
 * model's own except `doGenerate`, which intercepts eve's compaction call and
 * passes everything else through.
 */
export function createSynapCompactionModel<M extends object>(
  options: SynapCompactionModelOptions<M>,
): M {
  const { sdk, model, conversationId } = options;
  if (!sdk) throw new TypeError("createSynapCompactionModel requires a non-null sdk");
  if (!model || typeof model !== "object") {
    throw new TypeError("createSynapCompactionModel requires the model to fall back to");
  }
  const target = model as unknown as LanguageModelLike;

  const doGenerate = async (params: { prompt: unknown }): Promise<unknown> => {
    try {
      if (systemText(params?.prompt).includes(EVE_COMPACTION_MARKER)) {
        const id = typeof conversationId === "function" ? conversationId() : conversationId;
        if (id) {
          const text = await checkpointFor(sdk, id, params.prompt);
          if (text) return resultFor(target.specificationVersion, text);
        }
      }
    } catch {
      // eve's own summary still stands.
    }
    return target.doGenerate(params);
  };

  return new Proxy(model, {
    get(obj, prop) {
      if (prop === "doGenerate") return doGenerate;
      const value = Reflect.get(obj, prop, obj);
      return typeof value === "function" ? value.bind(obj) : value;
    },
  });
}
