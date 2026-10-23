// Synap's compaction in place of eve's, at the one place eve lets it in.
//
// eve compacts when the conversation passes `compaction.thresholdPercent` of the
// model's context window (default 0.9): it splits the history into older and
// recent turns and asks a model, with a system prompt that opens with
// "CONTEXT CHECKPOINT COMPACTION", to summarize the older ones (`generateText`
// in eve 0.25, `streamText` in 0.82). The answer becomes the checkpoint: one
// user message ("Summary of our conversation so far:"), one assistant message
// holding the text, then the recent turns.
//
// That is a seam: a model that answers that call with Synap's compaction
// instead of writing a summary. eve then stores Synap's compaction as its
// checkpoint, and every later turn sees it.
//
// ⚠ The wrapped model goes in `model`, the agent's own model. NOT in
// `compaction.model`. eve compiles a model object in `agent.ts` into a
// reference to the module ("the `model` export of agent.ts"), and at runtime it
// turns a reference like that back into a model by loading the module and
// taking `.model` of the definition (`loadSourceBackedRuntimeModelReference`
// in eve's `runtime/agent/resolve-model.js`, in 0.25.1 and 0.82.0 alike). So
// `compaction.model: <object>` is compiled, stored, and then replaced by the
// turn model when the compaction runs: the object is never called. Live, on
// both versions, `get_context_for_prompt` was called zero times and the stored
// checkpoint was the turn model's own reply. With no `compaction.model`, eve
// compacts with the turn model, and the turn model is this wrapper.
//
//   // agent/agent.ts
//   import { defineAgent } from "eve";
//   import { createSynapCompactionModel } from "@maximem/synap-eve";
//   import { sdk } from "./lib/synap.js";
//
//   export default defineAgent({
//     model: createSynapCompactionModel({
//       sdk, model: openai("gpt-5"), conversationId: "support-alice",
//     }),
//     modelContextWindowTokens: 400_000, // a model object has no gateway metadata to read
//     compaction: { thresholdPercent: 0.6 },
//   });
//
// Needs the model as an AI SDK model object (`openai("gpt-5")`,
// `gateway("openai/gpt-5")`), not a "provider/model" string: a string cannot be
// wrapped. Leave `compaction.model` unset.
//
// What this cannot do, from eve's source:
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
// Everything else about the model, every other call included, is the model you
// pass in. When Synap has no compaction for the conversation (or cannot be
// reached) the compaction call goes to that model and eve's own summary is used.

import { fetchCompactedHistory } from "./stream-events.js";

/** What this needs of an AI SDK language model: its own generate call. */
interface LanguageModelLike {
  readonly specificationVersion?: string;
  doGenerate(options: { prompt: unknown }): PromiseLike<unknown>;
  doStream?(options: { prompt: unknown }): PromiseLike<unknown>;
}

export interface SynapCompactionModelOptions<M extends object> {
  /** A Synap SDK with `conversation.context.get_context_for_prompt`. */
  sdk: unknown;
  /**
   * The agent's model: the one that answers every turn, and that summarizes
   * when Synap has nothing to give. The result behaves exactly like this model
   * for every call but the compaction, and goes in `defineAgent({ model })`.
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

/** The finish reason and usage, in the shape the model's specification version expects. */
function endOf(version: string | undefined): { finishReason: unknown; usage: Record<string, unknown> } {
  if (version === "v2") {
    return { finishReason: "stop", usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 } };
  }
  // v3 and v4 share this shape.
  return {
    finishReason: { unified: "stop", raw: "synap-compaction" },
    usage: {
      inputTokens: { total: 0, noCache: 0, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 0, text: 0, reasoning: 0 },
    },
  };
}

/** A generate result carrying `text`. */
function resultFor(version: string | undefined, text: string): Record<string, unknown> {
  return { content: [{ type: "text", text }], ...endOf(version), warnings: [] };
}

/** A stream result carrying `text`, for the `streamText` eve 0.82 summarizes with. */
function streamFor(version: string | undefined, text: string): { stream: ReadableStream<unknown> } {
  const parts: unknown[] = [
    { type: "stream-start", warnings: [] },
    { type: "text-start", id: "synap-compaction" },
    { type: "text-delta", id: "synap-compaction", delta: text },
    { type: "text-end", id: "synap-compaction" },
    { type: "finish", ...endOf(version) },
  ];
  return {
    stream: new ReadableStream({
      start(controller) {
        for (const part of parts) controller.enqueue(part);
        controller.close();
      },
    }),
  };
}

/**
 * A language model that answers eve's compaction call with Synap's compaction.
 *
 * Returns the model you pass in, wrapped: every property and method is that
 * model's own except `doGenerate` and `doStream`, which intercept eve's
 * compaction call (eve 0.25 summarizes with `generateText`, 0.82 with
 * `streamText`) and pass everything else through. Use it as the agent's
 * `model`; see the note at the top of this file for why not `compaction.model`.
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

  /** Synap's checkpoint when this is eve's compaction call and Synap has one. */
  const synapCheckpoint = async (params: { prompt: unknown }): Promise<string | null> => {
    try {
      if (systemText(params?.prompt).includes(EVE_COMPACTION_MARKER)) {
        const id = typeof conversationId === "function" ? conversationId() : conversationId;
        if (id) return await checkpointFor(sdk, id, params.prompt);
      }
    } catch {
      // eve's own summary still stands.
    }
    return null;
  };

  const doGenerate = async (params: { prompt: unknown }): Promise<unknown> => {
    const text = await synapCheckpoint(params);
    return text !== null ? resultFor(target.specificationVersion, text) : target.doGenerate(params);
  };

  const doStream = async (params: { prompt: unknown }): Promise<unknown> => {
    const text = await synapCheckpoint(params);
    if (text !== null) return streamFor(target.specificationVersion, text);
    if (typeof target.doStream !== "function") throw new TypeError("the wrapped model has no doStream");
    return target.doStream(params);
  };

  return new Proxy(model, {
    get(obj, prop) {
      if (prop === "doGenerate") return doGenerate;
      if (prop === "doStream") return doStream;
      const value = Reflect.get(obj, prop, obj);
      return typeof value === "function" ? value.bind(obj) : value;
    },
  });
}
