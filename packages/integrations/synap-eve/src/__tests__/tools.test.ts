import { describe, it, expect } from "vitest";

import { createSynapSearchTool, createSynapStoreTool } from "../tools.js";
import { SynapIntegrationError } from "../types.js";
import { makeSdk, makeFailingSdk, makeCtx, EMPTY } from "./helpers.js";

describe("createSynapSearchTool", () => {
  it("returns formatted context on a hit (happy path)", async () => {
    const sdk = makeSdk();
    const tool = createSynapSearchTool({ sdk });
    const out = await tool.execute({ query: "what do we know about alice?" }, makeCtx() as never);
    expect(out.available).toBe(true);
    expect(out.formattedContext).toContain("Senior engineer");
    expect(sdk.fetch).toHaveBeenCalledOnce();
  });

  it("auto-scopes user_id + conversation_id from the eve session", async () => {
    const sdk = makeSdk();
    const tool = createSynapSearchTool({ sdk });
    await tool.execute({ query: "x" }, makeCtx({ sessionId: "conv-9", principalId: "bob" }) as never);
    expect(sdk.fetch).toHaveBeenCalledWith(
      expect.objectContaining({ user_id: "bob", conversation_id: "conv-9" }),
    );
  });

  it("explicit userId overrides the eve principal", async () => {
    const sdk = makeSdk();
    const tool = createSynapSearchTool({ sdk, userId: "explicit" });
    await tool.execute({ query: "x" }, makeCtx({ principalId: "bob" }) as never);
    expect(sdk.fetch).toHaveBeenCalledWith(expect.objectContaining({ user_id: "explicit" }));
  });

  it("degrades to unavailable when no user scope is resolvable", async () => {
    const sdk = makeSdk();
    const tool = createSynapSearchTool({ sdk });
    const out = await tool.execute({ query: "x" }, makeCtx({ principalId: null }) as never);
    expect(out).toEqual({ formattedContext: "", available: false });
    expect(sdk.fetch).not.toHaveBeenCalled();
  });

  it("does NOT fall back to the session initiator (no cross-scope leak)", async () => {
    const sdk = makeSdk();
    const tool = createSynapSearchTool({ sdk });
    // current principal is null (delegated turn); initiator is an admin who
    // started the session — must not be used as the memory scope.
    const out = await tool.execute(
      { query: "x" },
      makeCtx({ principalId: null, initiatorId: "admin" }) as never,
    );
    expect(out).toEqual({ formattedContext: "", available: false });
    expect(sdk.fetch).not.toHaveBeenCalled();
  });

  it("forwards customerId scoping to the SDK", async () => {
    const sdk = makeSdk();
    const tool = createSynapSearchTool({ sdk, customerId: "acme" });
    await tool.execute({ query: "x" }, makeCtx() as never);
    expect(sdk.fetch).toHaveBeenCalledWith(expect.objectContaining({ customer_id: "acme" }));
  });

  it("degrades to unavailable on empty context", async () => {
    const sdk = makeSdk({ fetchResponse: EMPTY() });
    const tool = createSynapSearchTool({ sdk });
    const out = await tool.execute({ query: "x" }, makeCtx() as never);
    expect(out).toEqual({ formattedContext: "", available: false });
  });

  it("degrades (does NOT throw) on SDK failure — reads swallow errors", async () => {
    const sdk = makeFailingSdk();
    const tool = createSynapSearchTool({ sdk });
    const out = await tool.execute({ query: "x" }, makeCtx() as never);
    expect(out).toEqual({ formattedContext: "", available: false });
  });
});

describe("createSynapStoreTool", () => {
  it("records a memory and returns the ingestion id (happy path)", async () => {
    const sdk = makeSdk({ ingestionId: "ing-42" });
    const tool = createSynapStoreTool({ sdk });
    const out = await tool.execute({ content: "Alice prefers email" }, makeCtx() as never);
    expect(out).toEqual({ recorded: true, ingestionId: "ing-42" });
    expect(sdk.memories.create).toHaveBeenCalledWith(
      expect.objectContaining({ document: "Alice prefers email", user_id: "alice" }),
    );
  });

  it("stamps source=eve into metadata (caller metadata wins)", async () => {
    const sdk = makeSdk();
    const tool = createSynapStoreTool({ sdk });
    await tool.execute({ content: "note", metadata: { tag: "x" } }, makeCtx() as never);
    expect(sdk.memories.create).toHaveBeenCalledWith(
      expect.objectContaining({ metadata: { source: "eve", tag: "x" } }),
    );
  });

  it("raises SynapIntegrationError on SDK failure — writes never drop", async () => {
    const sdk = makeFailingSdk();
    const tool = createSynapStoreTool({ sdk });
    await expect(tool.execute({ content: "note" }, makeCtx() as never)).rejects.toBeInstanceOf(
      SynapIntegrationError,
    );
  });

  it("raises SynapIntegrationError on missing content", async () => {
    const sdk = makeSdk();
    const tool = createSynapStoreTool({ sdk });
    await expect(tool.execute({ content: "  " }, makeCtx() as never)).rejects.toBeInstanceOf(
      SynapIntegrationError,
    );
    expect(sdk.memories.create).not.toHaveBeenCalled();
  });

  it("raises SynapIntegrationError when no user scope is resolvable", async () => {
    const sdk = makeSdk();
    const tool = createSynapStoreTool({ sdk });
    await expect(
      tool.execute({ content: "note" }, makeCtx({ principalId: null }) as never),
    ).rejects.toBeInstanceOf(SynapIntegrationError);
    expect(sdk.memories.create).not.toHaveBeenCalled();
  });
});
