// The default path: no `listen()` from the caller, and the stream is used.
//
// `stream.test.ts` hands the hooks an SDK that has a stream or cannot make one.
// This hands them the common case, a stream that is closed until somebody calls
// `listen()` (as the real SDK is), and asserts the first event opens it and
// everything after travels on it.

import { describe, expect, it, vi } from "vitest";

import { createSynapStreamHooks } from "../stream.js";
import { setAutoListen } from "../stream-events.js";
import type { SynapTurnSdkLike } from "../stream.js";

interface Call { method: string; options: unknown }

function closedSdk() {
  const calls: Call[] = [];
  const state = { listening: false, listens: 0 };
  const push = (method: string) => async (options: unknown) => { calls.push({ method, options }); };
  const sdk = {
    instance: {
      get is_listening() { return state.listening; },
      async listen() { state.listens += 1; state.listening = true; },
      send_message: push("send_message"),
      record_tool_call: push("record_tool_call"),
      record_tool_result: push("record_tool_result"),
      record_thinking: push("record_thinking"),
      end_session: push("end_session"),
    },
    conversation: { record_message: vi.fn(async () => ({})) },
  };
  return { sdk: sdk as unknown as SynapTurnSdkLike, calls, state, rest: sdk.conversation.record_message };
}

const ctx = {
  session: { id: "sess-1", auth: { current: { principalId: "alice" }, initiator: null }, turn: {} },
};

type Handler = (event: unknown, hookCtx: unknown) => Promise<void>;
const handler = (hooks: ReturnType<typeof createSynapStreamHooks>, name: string) =>
  (hooks.events as unknown as Record<string, Handler>)[name]!;

const userMessage = (message: string) => ({ type: "message.received", data: { message, sequence: 1, turnId: "t1" } });
const assistantMessage = (message: string) => ({
  type: "message.completed", data: { message, sequence: 2, stepIndex: 0, turnId: "t1", finishReason: "stop" },
});

describe("no listen() from the caller", () => {
  it("the first turn opens the stream and goes on it, not over REST", async () => {
    const { sdk, calls, state, rest } = closedSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.received")(userMessage("where is my order"), ctx);
    expect(state.listens).toBe(1);
    expect(calls.map((c) => c.method)).toEqual(["send_message"]);
    expect(calls[0]?.options).toMatchObject({
      role: "user", event_type: "user_message", conversation_id: "sess-1", user_id: "alice",
    });
    expect(rest).not.toHaveBeenCalled();
  });

  it("the rest of the run rides the same stream", async () => {
    const { sdk, calls, state, rest } = closedSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.received")(userMessage("hi"), ctx);
    await handler(hooks, "reasoning.completed")(
      { type: "reasoning.completed", data: { reasoning: "thinking", sequence: 3, stepIndex: 0, turnId: "t1" } }, ctx);
    await handler(hooks, "actions.requested")({
      type: "actions.requested",
      data: { actions: [{ callId: "c1", input: {}, kind: "tool-call", toolName: "lookup" }], sequence: 4, stepIndex: 0, turnId: "t1" },
    }, ctx);
    await handler(hooks, "action.result")({
      type: "action.result",
      data: { result: { callId: "c1", kind: "tool-result", output: "ok", toolName: "lookup" }, sequence: 5, stepIndex: 0, status: "completed", turnId: "t1" },
    }, ctx);
    await handler(hooks, "message.completed")(assistantMessage("it shipped"), ctx);
    await handler(hooks, "session.completed")({ type: "session.completed" }, ctx);
    expect(state.listens).toBe(1);
    expect(calls.map((c) => c.method)).toEqual([
      "send_message", "record_thinking", "record_tool_call", "record_tool_result", "send_message", "end_session",
    ]);
    expect(rest).not.toHaveBeenCalled();
  });

  it("a first event that is not a turn (reasoning) also opens it", async () => {
    const { sdk, state } = closedSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "reasoning.completed")(
      { type: "reasoning.completed", data: { reasoning: "thinking", sequence: 3, stepIndex: 0, turnId: "t1" } }, ctx);
    expect(state.listens).toBe(1);
  });

  it("session.completed alone does not open a stream", async () => {
    const { sdk, state } = closedSdk();
    await handler(createSynapStreamHooks({ sdk }), "session.completed")({ type: "session.completed" }, ctx);
    expect(state.listens).toBe(0);
  });

  it("setAutoListen(sdk, false): turns go over REST, stream-only events go nowhere", async () => {
    const { sdk, calls, state, rest } = closedSdk();
    setAutoListen(sdk, false);
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.received")(userMessage("hi"), ctx);
    await handler(hooks, "message.completed")(assistantMessage("hello"), ctx);
    await handler(hooks, "reasoning.completed")(
      { type: "reasoning.completed", data: { reasoning: "thinking", sequence: 3, stepIndex: 0, turnId: "t1" } }, ctx);
    expect(state.listens).toBe(0);
    expect(calls).toEqual([]);
    expect(rest).toHaveBeenCalledTimes(2);
  });

  it("a stream that cannot open falls back to REST for the turn", async () => {
    const { sdk, state, rest } = closedSdk();
    (sdk as unknown as { instance: { listen: () => Promise<void> } }).instance.listen = async () => {
      state.listens += 1;
      throw new Error("connection refused");
    };
    vi.spyOn(console, "warn").mockImplementation(() => {});
    await handler(createSynapStreamHooks({ sdk }), "message.received")(userMessage("hi"), ctx);
    expect(rest).toHaveBeenCalledOnce();
    vi.restoreAllMocks();
  });
});
