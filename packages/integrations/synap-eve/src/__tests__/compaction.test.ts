// Synap's compaction at eve's compaction seam.
//
// The unit half calls the wrapped model the way eve's `generateText` does. The
// other half runs eve's OWN `compactMessages` (loaded from the installed eve
// package, not copied) with the wrapped model, so what is asserted is the
// checkpoint eve would store, and whether the underlying model was asked at all.

import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

import { MockLanguageModelV3 } from "ai/test";
import { describe, expect, it, vi } from "vitest";

import { createSynapCompactionModel } from "../compaction.js";

const COMPACTED = {
  formatted_context: "",
  available: true,
  context_block: "## Conversation so far (compacted at 6)\nSYNAP_SUMMARY",
  history_block: null,
  recent_messages: [
    { role: "user", content: "second question about the invoice" },
    { role: "assistant", content: "second answer about the invoice" },
    { role: "user", content: "a turn eve still holds as recent" },
  ],
  recent_message_count: 3,
  total_message_count: 9,
};

function synapSdk(response: unknown = COMPACTED) {
  return {
    conversation: { context: { get_context_for_prompt: vi.fn(async () => response) } },
  };
}

function underlying() {
  return new MockLanguageModelV3({
    doGenerate: async () => ({
      content: [{ type: "text", text: "EVE_OWN_SUMMARY" }],
      finishReason: { unified: "stop", raw: "stop" },
      usage: {
        inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 1, text: 1, reasoning: 0 },
      },
      warnings: [],
    }),
  });
}

// eve's compaction system prompt, read from the installed package so a change
// to its wording fails here instead of silently disabling the seam.
const evePkg = createRequire(import.meta.url).resolve("eve/package.json");
const eveDist = new URL("./dist/src/", pathToFileURL(evePkg));
const evePrompt = await import(/* @vite-ignore */ new URL("harness/compaction-prompt.js", eveDist).href);
const eveCompaction = await import(/* @vite-ignore */ new URL("harness/compaction.js", eveDist).href);

const compactionCall = (text = "### user\nhello") => ({
  prompt: [
    { role: "system" as const, content: evePrompt.createCompactionPrompt({ messages: [], previousCheckpoint: undefined }).system },
    { role: "user" as const, content: [{ type: "text" as const, text }] },
  ],
});

describe("the wrapped model", () => {
  it("answers eve's compaction call with Synap's compaction", async () => {
    const model = underlying();
    const wrapped = createSynapCompactionModel({ sdk: synapSdk(), model, conversationId: "conv-1" });
    const out = await wrapped.doGenerate(compactionCall("second question about the invoice") as never);
    const text = (out.content[0] as { text: string }).text;
    expect(text).toContain("<synap_conversation_history>");
    expect(text).toContain("SYNAP_SUMMARY");
    expect(model.doGenerateCalls).toHaveLength(0);
  });

  it("carries the post-compaction turns eve is summarizing, and not the ones it keeps", async () => {
    const wrapped = createSynapCompactionModel({ sdk: synapSdk(), model: underlying(), conversationId: "conv-1" });
    const out = await wrapped.doGenerate(compactionCall("### user\nsecond question about the invoice\n### assistant\nsecond answer about the invoice") as never);
    const text = (out.content[0] as { text: string }).text;
    expect(text).toContain("### user\nsecond question about the invoice");
    expect(text).toContain("### assistant\nsecond answer about the invoice");
    expect(text).not.toContain("a turn eve still holds as recent");
  });

  it("reads the conversation from a function too", async () => {
    const sdk = synapSdk();
    const wrapped = createSynapCompactionModel({ sdk, model: underlying(), conversationId: () => "conv-fn" });
    await wrapped.doGenerate(compactionCall() as never);
    expect(sdk.conversation.context.get_context_for_prompt).toHaveBeenCalledWith({ conversation_id: "conv-fn" });
  });

  it("passes the call to the underlying model when Synap has no compaction", async () => {
    const model = underlying();
    const wrapped = createSynapCompactionModel({
      sdk: synapSdk({ ...COMPACTED, context_block: null }), model, conversationId: "conv-1",
    });
    const out = await wrapped.doGenerate(compactionCall() as never);
    expect((out.content[0] as { text: string }).text).toBe("EVE_OWN_SUMMARY");
    expect(model.doGenerateCalls).toHaveLength(1);
  });

  it("passes the call through when the conversation cannot be named or Synap fails", async () => {
    const model = underlying();
    const noId = createSynapCompactionModel({ sdk: synapSdk(), model, conversationId: () => undefined });
    expect(((await noId.doGenerate(compactionCall() as never)).content[0] as { text: string }).text).toBe("EVE_OWN_SUMMARY");
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const failing = { conversation: { context: { get_context_for_prompt: async () => { throw new Error("500"); } } } };
    const down = createSynapCompactionModel({ sdk: failing, model, conversationId: "conv-1" });
    expect(((await down.doGenerate(compactionCall() as never)).content[0] as { text: string }).text).toBe("EVE_OWN_SUMMARY");
    vi.restoreAllMocks();
  });

  it("leaves every other call to the underlying model", async () => {
    const sdk = synapSdk();
    const model = underlying();
    const wrapped = createSynapCompactionModel({ sdk, model, conversationId: "conv-1" });
    const out = await wrapped.doGenerate({ prompt: [
      { role: "system", content: "be nice" }, { role: "user", content: [{ type: "text", text: "hi" }] },
    ] } as never);
    expect((out.content[0] as { text: string }).text).toBe("EVE_OWN_SUMMARY");
    expect(sdk.conversation.context.get_context_for_prompt).not.toHaveBeenCalled();
  });

  it("is the same model otherwise", () => {
    const model = underlying();
    const wrapped = createSynapCompactionModel({ sdk: synapSdk(), model, conversationId: "c" });
    expect(wrapped.modelId).toBe(model.modelId);
    expect(wrapped.provider).toBe(model.provider);
    expect(wrapped.specificationVersion).toBe("v3");
  });

  it("answers in the v2 shape for a v2 model", async () => {
    const v2 = { specificationVersion: "v2", provider: "p", modelId: "m", doGenerate: vi.fn() };
    const wrapped = createSynapCompactionModel({ sdk: synapSdk(), model: v2, conversationId: "c" });
    const out = await wrapped.doGenerate(compactionCall() as never) as unknown as { finishReason: unknown };
    expect(out.finishReason).toBe("stop");
    expect(v2.doGenerate).not.toHaveBeenCalled();
  });

  it("refuses to be built without an sdk or a model", () => {
    expect(() => createSynapCompactionModel({ sdk: null, model: underlying(), conversationId: "c" })).toThrow(TypeError);
    expect(() => createSynapCompactionModel({ sdk: synapSdk(), model: undefined as never, conversationId: "c" })).toThrow(TypeError);
  });
});

describe("eve's own compactMessages with the wrapped model", () => {
  const history = Array.from({ length: 6 }, (_, i) => ({
    role: i % 2 === 0 ? "user" : "assistant",
    content: i === 0 ? "first question" : i === 1 ? "first answer" : i === 2 ? "second question about the invoice"
      : i === 3 ? "second answer about the invoice" : i === 4 ? "a turn eve still holds as recent" : "its reply",
  }));
  const config = { threshold: 100_000, recentWindowSize: 2 };

  it("stores Synap's compaction as the checkpoint, and never calls the summarizing model", async () => {
    const model = underlying();
    const wrapped = createSynapCompactionModel({ sdk: synapSdk(), model, conversationId: "conv-1" });
    const out = await eveCompaction.compactMessages(history, wrapped, config, undefined, undefined, undefined, undefined);
    expect(model.doGenerateCalls).toHaveLength(0);
    expect(out[0]).toEqual({ role: "user", content: evePrompt.COMPACTION_CHECKPOINT_MARKER });
    expect(out[1].role).toBe("assistant");
    expect(out[1].content).toContain("SYNAP_SUMMARY");
    expect(out[1].content).toContain("### user\nsecond question about the invoice");
    // eve's recent window follows the checkpoint, exactly once.
    const all = JSON.stringify(out);
    expect(all.split("a turn eve still holds as recent").length - 1).toBe(1);
  });

  it("is eve's own summary, from the same call, when Synap has none", async () => {
    const model = underlying();
    const wrapped = createSynapCompactionModel({
      sdk: synapSdk({ ...COMPACTED, context_block: null }), model, conversationId: "conv-1",
    });
    const out = await eveCompaction.compactMessages(history, wrapped, config, undefined, undefined, undefined, undefined);
    expect(model.doGenerateCalls).toHaveLength(1);
    expect(out[1].content).toBe("EVE_OWN_SUMMARY");
  });
});
