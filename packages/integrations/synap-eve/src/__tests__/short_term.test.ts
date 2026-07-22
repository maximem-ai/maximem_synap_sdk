import { describe, it, expect } from "vitest";

import {
  fetchSynapShortTerm,
  buildSynapShortTermMarkdown,
  createSynapInstructions,
} from "../short_term.js";
import { makeSdk, makeFailingSdk, makeCtx, promptContext } from "./helpers.js";

describe("fetchSynapShortTerm", () => {
  it("returns available context (happy path)", async () => {
    const sdk = makeSdk();
    const res = await fetchSynapShortTerm({ sdk, conversationId: "conv-1" });
    expect(res.available).toBe(true);
    expect(res.formattedContext).toContain("senior engineer");
  });

  it("returns unavailable when the server reports none", async () => {
    const sdk = makeSdk({ promptCtx: promptContext(false) });
    const res = await fetchSynapShortTerm({ sdk, conversationId: "conv-1" });
    expect(res).toEqual({ formattedContext: "", available: false });
  });

  it("degrades to unavailable on SDK failure with onError=fallback (default)", async () => {
    const sdk = makeFailingSdk();
    const res = await fetchSynapShortTerm({ sdk, conversationId: "conv-1" });
    expect(res).toEqual({ formattedContext: "", available: false });
  });

  it("propagates the SDK error with onError=raise", async () => {
    const sdk = makeFailingSdk();
    await expect(
      fetchSynapShortTerm({ sdk, conversationId: "conv-1", onError: "raise" }),
    ).rejects.toThrow();
  });

  it("rejects an empty conversationId", async () => {
    const sdk = makeSdk();
    await expect(fetchSynapShortTerm({ sdk, conversationId: "" })).rejects.toBeInstanceOf(TypeError);
  });

  it("rejects an unsupported style", async () => {
    const sdk = makeSdk();
    await expect(
      // @ts-expect-error — intentional bad style
      fetchSynapShortTerm({ sdk, conversationId: "c", style: "poetic" }),
    ).rejects.toBeInstanceOf(TypeError);
  });
});

describe("buildSynapShortTermMarkdown", () => {
  it("wraps body in the default preamble tags", () => {
    const md = buildSynapShortTermMarkdown({ formattedContext: "hello", available: true });
    expect(md).toBe("<synap_short_term_context>\nhello\n</synap_short_term_context>");
  });

  it("emits raw body when tags are nulled", () => {
    const md = buildSynapShortTermMarkdown(
      { formattedContext: "hello", available: true },
      { preambleOpen: null, preambleClose: null },
    );
    expect(md).toBe("hello");
  });

  it("returns empty string on empty context", () => {
    expect(buildSynapShortTermMarkdown({ formattedContext: "", available: false })).toBe("");
  });
});

describe("createSynapInstructions (dynamic resolver)", () => {
  // The resolver is the `turn.started` handler eve invokes; we exercise it
  // directly with a duck-typed ctx.
  function getResolver(def: unknown): (event: unknown, ctx: unknown) => Promise<unknown> {
    const events = (def as { events: Record<string, unknown> }).events;
    return events["turn.started"] as never;
  }

  it("returns a defineInstructions payload with the recalled markdown (happy path)", async () => {
    const sdk = makeSdk();
    const resolver = getResolver(createSynapInstructions({ sdk }));
    const out = (await resolver({}, makeCtx({ sessionId: "conv-7" }))) as { markdown?: string } | null;
    expect(out).not.toBeNull();
    expect(out?.markdown).toContain("synap_short_term_context");
    expect(sdk.conversation.context.get_context_for_prompt).toHaveBeenCalledWith(
      expect.objectContaining({ conversation_id: "conv-7" }),
    );
  });

  it("returns null (no-op) when there is no conversation id", async () => {
    const sdk = makeSdk();
    const resolver = getResolver(createSynapInstructions({ sdk }));
    const out = await resolver({}, { session: { id: "", auth: null } });
    expect(out).toBeNull();
    expect(sdk.conversation.context.get_context_for_prompt).not.toHaveBeenCalled();
  });

  it("returns null (no-op) when the server reports no context", async () => {
    const sdk = makeSdk({ promptCtx: promptContext(false) });
    const resolver = getResolver(createSynapInstructions({ sdk }));
    const out = await resolver({}, makeCtx());
    expect(out).toBeNull();
  });

  it("returns null (no-op) on SDK failure — read-degrade, turn proceeds", async () => {
    const sdk = makeFailingSdk();
    const resolver = getResolver(createSynapInstructions({ sdk }));
    const out = await resolver({}, makeCtx());
    expect(out).toBeNull();
  });
});
