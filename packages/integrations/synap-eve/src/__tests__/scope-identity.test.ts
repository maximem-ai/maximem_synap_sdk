// A path fixed when a tool or hook is built must not be reachable by a different
// user. The eve principal varies per call; the server lets the path's `user` and
// `customer` win over the ids sent beside it, so a mismatch would read or write
// another identity's memory. These tests assert on what reached the fake SDK:
// on a refusal, nothing may.

import { describe, expect, it, vi, afterEach } from "vitest";

import { createSynapSearchTool, createSynapStoreTool } from "../tools.js";
import { createSynapStreamHooks, type SynapTurnSdkLike } from "../stream.js";
import { SynapScopeError } from "../index.js";
import { fingerprint } from "../scope-guard.js";
import { makeSdk, makeCtx } from "./helpers.js";

const PATH = { customer: "acme", user: "alice" };

afterEach(() => vi.restoreAllMocks());

describe("tools: the principal must be the path's user", () => {
  it("search with another principal throws and the SDK is never called", async () => {
    const sdk = makeSdk();
    const tool = createSynapSearchTool({ sdk, customerId: "acme", scopePath: PATH });
    await expect(
      tool.execute({ query: "q" } as never, makeCtx({ principalId: "bob" }) as never),
    ).rejects.toBeInstanceOf(SynapScopeError);
    expect(sdk.fetch).not.toHaveBeenCalled();
  });

  it("store with another principal throws and the SDK is never called", async () => {
    const sdk = makeSdk();
    const tool = createSynapStoreTool({ sdk, customerId: "acme", scopePath: PATH });
    await expect(
      tool.execute({ content: "n" } as never, makeCtx({ principalId: "bob" }) as never),
    ).rejects.toBeInstanceOf(SynapScopeError);
    expect(sdk.memories.create).not.toHaveBeenCalled();
  });

  it("the refusal names no raw id", async () => {
    const tool = createSynapSearchTool({ sdk: makeSdk(), scopePath: PATH });
    const err = await Promise.resolve(
      tool.execute({ query: "q" } as never, makeCtx({ principalId: "bob-secret" }) as never),
    ).then(() => new Error("did not throw"), (e: unknown) => e as Error);
    expect(String(err.message)).toContain("synap_search");
    expect(String(err.message)).toContain(fingerprint("bob-secret"));
    for (const raw of ["bob-secret", "alice", "acme"]) {
      expect(String(err.message)).not.toContain(raw);
    }
  });

  it("the path's own user passes and the call carries the path", async () => {
    const sdk = makeSdk();
    await createSynapSearchTool({ sdk, scopePath: PATH })
      .execute({ query: "q" } as never, makeCtx({ principalId: "alice" }) as never);
    expect(sdk.fetch).toHaveBeenCalledWith(expect.objectContaining({ user_id: "alice", scope_path: PATH }));
  });

  it("a path with no `user` key (the multi-user case) accepts any principal", async () => {
    const sdk = makeSdk();
    const tool = createSynapSearchTool({ sdk, scopePath: { customer: "acme", team: "eng" } });
    await tool.execute({ query: "q" } as never, makeCtx({ principalId: "bob" }) as never);
    await tool.execute({ query: "q" } as never, makeCtx({ principalId: "carol" }) as never);
    expect(sdk.fetch).toHaveBeenCalledTimes(2);
  });

  it("customerId changed after build is still checked per call (search and store)", async () => {
    const sdk = makeSdk();
    const opts = { sdk, customerId: "acme", scopePath: PATH };
    const search = createSynapSearchTool(opts);
    const store = createSynapStoreTool(opts);
    opts.customerId = "globex";
    await expect(
      search.execute({ query: "q" } as never, makeCtx({ principalId: "alice" }) as never),
    ).rejects.toBeInstanceOf(SynapScopeError);
    await expect(
      store.execute({ content: "n" } as never, makeCtx({ principalId: "alice" }) as never),
    ).rejects.toBeInstanceOf(SynapScopeError);
    expect(sdk.fetch).not.toHaveBeenCalled();
    expect(sdk.memories.create).not.toHaveBeenCalled();
  });

  it("an explicit customerId that disagrees with the path is refused when the tool is built", () => {
    expect(() => createSynapSearchTool({ sdk: makeSdk(), customerId: "globex", scopePath: PATH }))
      .toThrow(SynapScopeError);
    expect(() => createSynapStoreTool({ sdk: makeSdk(), customerId: "globex", scopePath: PATH }))
      .toThrow(SynapScopeError);
  });

  it("an explicit userId that disagrees with the path is refused when the tool is built", () => {
    expect(() => createSynapSearchTool({ sdk: makeSdk(), userId: "bob", scopePath: PATH }))
      .toThrow(SynapScopeError);
  });

  it("B2C: customer == user passes with no customerId", async () => {
    const sdk = makeSdk();
    const tool = createSynapSearchTool({ sdk, scopePath: { customer: "alice", user: "alice" } });
    await tool.execute({ query: "q" } as never, makeCtx({ principalId: "alice" }) as never);
    expect(sdk.fetch).toHaveBeenCalledOnce();
  });

  it("without a path nothing is checked: any principal, any customerId", async () => {
    const sdk = makeSdk();
    await createSynapSearchTool({ sdk, customerId: "globex" })
      .execute({ query: "q" } as never, makeCtx({ principalId: "bob" }) as never);
    expect(sdk.fetch).toHaveBeenCalledWith(expect.objectContaining({ user_id: "bob", customer_id: "globex" }));
  });
});

describe("tools: an SDK without get_scope_ladder is refused when a path is set", () => {
  const OLD = () => {
    const sdk = makeSdk();
    delete (sdk as { get_scope_ladder?: unknown }).get_scope_ladder;
    return sdk;
  };

  it("both tools and the hooks throw at construction, naming 0.7.1", () => {
    expect(() => createSynapSearchTool({ sdk: OLD(), scopePath: PATH })).toThrow(/>= 0\.7\.1/);
    expect(() => createSynapStoreTool({ sdk: OLD(), scopePath: PATH })).toThrow(/>= 0\.7\.1/);
    expect(() => createSynapStreamHooks({ sdk: OLD() as unknown as SynapTurnSdkLike, scopePath: PATH }))
      .toThrow(/>= 0\.7\.1/);
  });

  it("without a path an old SDK still works", async () => {
    const sdk = OLD();
    await createSynapSearchTool({ sdk }).execute({ query: "q" } as never, makeCtx() as never);
    expect(sdk.fetch).toHaveBeenCalledOnce();
    expect(() => createSynapStreamHooks({ sdk: OLD() as unknown as SynapTurnSdkLike })).not.toThrow();
  });
});

describe("stream hooks: a mismatched principal reports nothing and does not break the run", () => {
  function streamSdk() {
    const calls: string[] = [];
    const rec = (m: string) => vi.fn(async () => { calls.push(m); });
    const sdk = {
      get_scope_ladder: vi.fn(),
      instance: {
        isListening: () => true,
        send_message: rec("send_message"),
        record_tool_call: rec("record_tool_call"),
        record_tool_result: rec("record_tool_result"),
        record_thinking: rec("record_thinking"),
        end_session: rec("end_session"),
      },
      conversation: { record_message: rec("record_message") },
    } as unknown as SynapTurnSdkLike;
    return { sdk, calls };
  }
  const ctx = (principalId: string) => makeCtx({ principalId });
  const userMsg = { type: "message.received", data: { message: "hi", sequence: 1, turnId: "t1" } };
  const doneMsg = { type: "message.completed", data: { message: "yo", sequence: 2, stepIndex: 0, turnId: "t1", finishReason: "stop" } };
  const events = (h: ReturnType<typeof createSynapStreamHooks>) =>
    h.events as unknown as Record<string, (e: unknown, c: unknown) => Promise<void>>;

  it("bob under alice's path: no stream event, no REST turn, no throw", async () => {
    const { sdk, calls } = streamSdk();
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const hooks = createSynapStreamHooks({ sdk, scopePath: PATH, customerId: "acme" });
    await events(hooks)["message.received"]!(userMsg, ctx("bob-secret"));
    await events(hooks)["message.completed"]!(doneMsg, ctx("bob-secret"));
    await events(hooks)["session.completed"]!({ type: "session.completed" }, ctx("bob-secret"));
    expect(calls).toEqual([]);
    // Logged at error level, once for the pair, fingerprints only.
    expect(err).toHaveBeenCalledTimes(1);
    const logged = String(err.mock.calls[0]![0]);
    expect(logged).toContain(fingerprint("bob-secret"));
    for (const raw of ["bob-secret", "alice", "acme"]) expect(logged).not.toContain(raw);
  });

  it("alice under alice's path still reports, with the path on the turn", async () => {
    const { sdk, calls } = streamSdk();
    const hooks = createSynapStreamHooks({ sdk, scopePath: PATH, customerId: "acme" });
    await events(hooks)["message.received"]!(userMsg, ctx("alice"));
    expect(calls).toEqual(["send_message"]);
  });

  it("an explicit customerId that disagrees with the path is refused when the hooks are built", () => {
    expect(() => createSynapStreamHooks({ sdk: streamSdk().sdk, scopePath: PATH, customerId: "globex" }))
      .toThrow(SynapScopeError);
  });

  it("without a path any principal reports", async () => {
    const { sdk, calls } = streamSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await events(hooks)["message.received"]!(userMsg, ctx("bob"));
    expect(calls).toEqual(["send_message"]);
  });
});
