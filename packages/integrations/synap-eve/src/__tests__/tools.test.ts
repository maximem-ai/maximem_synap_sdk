import { describe, it, expect, vi } from "vitest";

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

describe("scopePath (a deeper scope ladder)", () => {
  const PATH = { customer: "acme", team: "eng", user: "alice" };

  it("search sends it to sdk.fetch as scope_path", async () => {
    const sdk = makeSdk();
    const tool = createSynapSearchTool({ sdk, scopePath: PATH });
    await tool.execute({ query: "x" }, makeCtx() as never);
    expect(sdk.fetch).toHaveBeenCalledWith(expect.objectContaining({ scope_path: PATH }));
  });

  it("store sends it to memories.create as scope (not scope_path)", async () => {
    const sdk = makeSdk();
    const tool = createSynapStoreTool({ sdk, scopePath: PATH });
    await tool.execute({ content: "note" }, makeCtx() as never);
    const sent = (sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0]![0];
    expect(sent.scope).toEqual(PATH);
    expect(Object.keys(sent)).not.toContain("scope_path");
  });

  it("without one, both calls carry exactly the keys they always did", async () => {
    const sdk = makeSdk();
    await createSynapSearchTool({ sdk }).execute({ query: "x" }, makeCtx() as never);
    await createSynapStoreTool({ sdk }).execute({ content: "note" }, makeCtx() as never);
    expect(sdk.fetch).toHaveBeenCalledWith({
      conversation_id: "sess-eve-1", user_id: "alice", customer_id: null,
      search_query: ["x"], max_results: 10, mode: "accurate",
      include_conversation_context: false,
    });
    expect(sdk.memories.create).toHaveBeenCalledWith({
      document: "note", user_id: "alice", customer_id: null, metadata: { source: "eve" },
    });
    // An empty path is the same as none.
    const sdk2 = makeSdk();
    await createSynapSearchTool({ sdk: sdk2, scopePath: {} }).execute({ query: "x" }, makeCtx() as never);
    expect(Object.keys((sdk2.fetch as ReturnType<typeof vi.fn>).mock.calls[0]![0])).not.toContain("scope_path");
  });

  it("is closed over: not in either tool's input schema, and a model-supplied one is ignored", async () => {
    const sdk = makeSdk();
    const search = createSynapSearchTool({ sdk, scopePath: PATH });
    const store = createSynapStoreTool({ sdk, scopePath: PATH });
    for (const tool of [search, store]) {
      const shape = Object.keys((tool as unknown as { inputSchema: { shape: object } }).inputSchema.shape);
      expect(shape.filter((k) => /scope/i.test(k))).toEqual([]);
    }
    const evil = { team: "finance", user: "bob" };
    await search.execute({ query: "x", scope_path: evil, scopePath: evil, scope: evil } as never, makeCtx() as never);
    await store.execute({ content: "n", scope_path: evil, scopePath: evil, scope: evil } as never, makeCtx() as never);
    expect((sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0]![0].scope_path).toEqual(PATH);
    expect((sdk.memories.create as ReturnType<typeof vi.fn>).mock.calls[0]![0].scope).toEqual(PATH);
  });

  it("is also ignored when the tool has none configured", async () => {
    const sdk = makeSdk();
    const evil = { team: "finance" };
    await createSynapSearchTool({ sdk }).execute({ query: "x", scope_path: evil } as never, makeCtx() as never);
    expect(Object.keys((sdk.fetch as ReturnType<typeof vi.fn>).mock.calls[0]![0])).not.toContain("scope_path");
  });
});
