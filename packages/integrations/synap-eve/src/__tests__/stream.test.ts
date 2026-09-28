// What Synap receives when an eve agent runs with a live stream open.
//
// Every assertion is on what reached the fake SDK. A test that asserts a
// handler "did not throw" passes on a hook that reports nothing at all, and
// reporting nothing is the exact bug this work exists to fix.
//
// The handlers are pulled off the returned `HookDefinition` and called with the
// event shapes eve's own protocol types declare — `message.received` carries a
// flattened `message`, `actions.requested` carries a discriminated union of
// action kinds, and so on.

import { describe, expect, it, vi } from "vitest";

import { createSynapStreamHooks } from "../stream.js";
import type { SynapStreamSdkLike } from "../stream-events.js";

interface StreamCall {
  method: string;
  options: unknown;
}

const SESSION = "sess-eve-1";
const TURN = "turn-1";

function streamingSdk(opts: { listening?: boolean; throws?: boolean } = {}) {
  const calls: StreamCall[] = [];
  const record = (method: string) =>
    vi.fn(async (options: unknown) => {
      if (opts.throws === true) throw new Error("the stream is having a bad day");
      calls.push({ method, options });
    });
  const instance = {
    isListening: () => opts.listening !== false,
    send_message: record("send_message"),
    record_tool_call: record("record_tool_call"),
    record_tool_result: record("record_tool_result"),
    record_thinking: record("record_thinking"),
    end_session: record("end_session"),
  };
  return { sdk: { instance } as SynapStreamSdkLike, calls, instance };
}

function sentTo(calls: StreamCall[], method: string): Record<string, unknown> | undefined {
  return calls.find((c) => c.method === method)?.options as Record<string, unknown> | undefined;
}

function methodsIn(calls: StreamCall[]): string[] {
  return calls.map((c) => c.method);
}

/** A minimal eve `HookContext`: an authenticated session by default. */
function ctx(opts: { sessionId?: string; principalId?: string | null } = {}) {
  const { sessionId = SESSION, principalId = "alice" } = opts;
  return {
    session: {
      id: sessionId,
      auth: {
        current: principalId ? { principalId } : null,
        initiator: null,
      },
      turn: {},
    },
    agent: { name: "support" },
    channel: {},
    getSandbox: async () => ({}),
    getSkill: () => ({}),
  };
}

type Handler = (event: unknown, hookCtx: unknown) => void | Promise<void>;

/** The handler a hook definition registered for `name`. */
function handler(
  hooks: ReturnType<typeof createSynapStreamHooks>,
  name: string,
): Handler {
  const events = hooks.events as unknown as Record<string, Handler | undefined>;
  const found = events?.[name];
  if (!found) throw new Error(`no handler registered for ${name}`);
  return found;
}

// ── event fixtures, shaped the way eve's protocol types declare them ─────────

function userMessage(message: string, sequence = 1) {
  return { type: "message.received", data: { message, sequence, turnId: TURN } };
}

function assistantMessage(
  message: string | null,
  opts: { sequence?: number; finishReason?: string } = {},
) {
  return {
    type: "message.completed",
    data: {
      finishReason: opts.finishReason ?? "stop",
      message,
      sequence: opts.sequence ?? 2,
      stepIndex: 0,
      turnId: TURN,
    },
  };
}

function reasoning(text: string, opts: { sequence?: number; stepIndex?: number } = {}) {
  return {
    type: "reasoning.completed",
    data: {
      reasoning: text,
      sequence: opts.sequence ?? 3,
      stepIndex: opts.stepIndex ?? 0,
      turnId: TURN,
    },
  };
}

function actionsRequested(actions: unknown[], sequence = 4) {
  return {
    type: "actions.requested",
    data: { actions, sequence, stepIndex: 0, turnId: TURN },
  };
}

function actionResult(result: unknown, opts: { status?: string; sequence?: number } = {}) {
  return {
    type: "action.result",
    data: {
      result,
      sequence: opts.sequence ?? 5,
      stepIndex: 0,
      status: opts.status ?? "completed",
      turnId: TURN,
    },
  };
}

const TOOL_CALL = {
  callId: "call_abc",
  input: { orderId: "A-1" },
  kind: "tool-call",
  toolName: "lookup_order",
};

const TOOL_RESULT = {
  callId: "call_abc",
  kind: "tool-result",
  output: { status: "shipped" },
  toolName: "lookup_order",
};

// ── the hook surface ─────────────────────────────────────────────────────────

describe("the hook definition", () => {
  it("subscribes to exactly the events Synap has a use for", () => {
    const { sdk } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    expect(Object.keys(hooks.events ?? {}).sort()).toEqual([
      "action.result",
      "actions.requested",
      "message.completed",
      "message.received",
      "reasoning.completed",
      "session.completed",
    ]);
  });

  it("refuses to be built without an sdk", () => {
    expect(() =>
      createSynapStreamHooks({ sdk: undefined as unknown as SynapStreamSdkLike }),
    ).toThrow(TypeError);
  });
});

// ── silent without a stream ──────────────────────────────────────────────────

describe("silent for a caller who never opened a stream", () => {
  it("reports nothing on any handler when isListening says no", async () => {
    const { sdk, calls } = streamingSdk({ listening: false });
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.received")(userMessage("hi"), ctx());
    await handler(hooks, "message.completed")(assistantMessage("hello"), ctx());
    await handler(hooks, "reasoning.completed")(reasoning("thinking"), ctx());
    await handler(hooks, "actions.requested")(actionsRequested([TOOL_CALL]), ctx());
    await handler(hooks, "action.result")(actionResult(TOOL_RESULT), ctx());
    await handler(hooks, "session.completed")({ type: "session.completed" }, ctx());
    expect(calls).toEqual([]);
  });

  it("reports nothing when the sdk never started a stream at all", async () => {
    // An SDK with the methods but no `listen()` running: `isListening` is
    // absent entirely, which must read as "no stream" and not as a truthy
    // function object.
    const calls: StreamCall[] = [];
    const sdk = {
      instance: {
        send_message: vi.fn(async (options: unknown) => {
          calls.push({ method: "send_message", options });
        }),
      },
    } as SynapStreamSdkLike;
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.received")(userMessage("hi"), ctx());
    expect(calls).toEqual([]);
  });
});

// ── the five events ──────────────────────────────────────────────────────────

describe("the turn", () => {
  it("reports message.received as user_message scoped to the eve session", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.received")(userMessage("where is my order"), ctx());
    expect(sentTo(calls, "send_message")).toMatchObject({
      role: "user",
      event_type: "user_message",
      content: "where is my order",
      conversation_id: SESSION,
      user_id: "alice",
    });
  });

  it("reports message.completed as assistant_message, the event anticipation wakes on", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.completed")(assistantMessage("it ships Friday"), ctx());
    expect(sentTo(calls, "send_message")).toMatchObject({
      role: "assistant",
      event_type: "assistant_message",
      content: "it ships Friday",
    });
  });

  it("reports the reply a model gives before it calls a tool", async () => {
    // eve emits message.completed whenever an assistant step finishes with
    // visible text. Text the user saw is a turn, whatever came after it.
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.completed")(
      assistantMessage("let me look that up", { finishReason: "tool-calls" }),
      ctx(),
    );
    expect(sentTo(calls, "send_message")).toMatchObject({
      event_type: "assistant_message",
      content: "let me look that up",
    });
  });

  it("a completed message with no text is not a turn", async () => {
    // data.message is `string | null` on eve's own type.
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.completed")(assistantMessage(null), ctx());
    expect(calls).toEqual([]);
  });

  it("an empty user message is not a turn", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.received")(userMessage("   "), ctx());
    expect(calls).toEqual([]);
  });
});

describe("the middle of the turn, which has no other home", () => {
  it("reports reasoning.completed with eve's own step index", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "reasoning.completed")(
      reasoning("I should check the order table", { stepIndex: 2 }),
      ctx(),
    );
    expect(sentTo(calls, "record_thinking")).toMatchObject({
      content: "I should check the order table",
      thought_type: "model_thought",
      step_index: 2,
      conversation_id: SESSION,
    });
  });

  it("keeps step 0, which is the first step and not a missing one", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "reasoning.completed")(
      reasoning("first thought", { stepIndex: 0 }),
      ctx(),
    );
    expect(sentTo(calls, "record_thinking")).toHaveProperty("step_index", 0);
  });

  it("reports a tool call with eve's own call id", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "actions.requested")(actionsRequested([TOOL_CALL]), ctx());
    expect(sentTo(calls, "record_tool_call")).toMatchObject({
      tool_name: "lookup_order",
      tool_args: { orderId: "A-1" },
      tool_call_id: "call_abc",
      conversation_id: SESSION,
    });
  });

  it("reports every action in one event, not just the first", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "actions.requested")(
      actionsRequested([
        TOOL_CALL,
        { callId: "call_2", input: {}, kind: "tool-call", toolName: "send_email" },
      ]),
      ctx(),
    );
    expect(
      calls.map((c) => (c.options as Record<string, unknown>)["tool_name"]),
    ).toEqual(["lookup_order", "send_email"]);
  });

  it("reports a subagent call rather than dropping it", async () => {
    // An agent that delegates everything would otherwise report no actions.
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "actions.requested")(
      actionsRequested([
        {
          callId: "call_sub",
          description: "research",
          input: { task: "find the order" },
          kind: "subagent-call",
          name: "researcher_tool",
          nodeId: "n1",
          subagentName: "researcher",
        },
      ]),
      ctx(),
    );
    expect(sentTo(calls, "record_tool_call")).toMatchObject({
      tool_name: "researcher",
      tool_call_id: "call_sub",
    });
  });

  it("reports a load-skill action under a name rather than an empty one", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "actions.requested")(
      actionsRequested([{ callId: "call_sk", input: { name: "refunds" }, kind: "load-skill" }]),
      ctx(),
    );
    expect(sentTo(calls, "record_tool_call")).toMatchObject({
      tool_name: "load-skill",
      tool_call_id: "call_sk",
    });
  });

  it("reports a tool result under the same call id, so the two pair", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "actions.requested")(actionsRequested([TOOL_CALL]), ctx());
    await handler(hooks, "action.result")(actionResult(TOOL_RESULT), ctx());
    expect(sentTo(calls, "record_tool_call")?.["tool_call_id"]).toBe("call_abc");
    expect(sentTo(calls, "record_tool_result")).toMatchObject({
      tool_call_id: "call_abc",
      tool_name: "lookup_order",
    });
  });

  it("sends the tool's own output, not eve's envelope around it", async () => {
    // Shipping the RuntimeActionResult itself would send eve's bookkeeping
    // (callId, kind) as if the tool had returned it.
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "action.result")(actionResult(TOOL_RESULT), ctx());
    expect(sentTo(calls, "record_tool_result")?.["result"]).toEqual({ status: "shipped" });
  });

  it("reports a subagent result under the subagent's name", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "action.result")(
      actionResult({
        callId: "call_sub",
        kind: "subagent-result",
        output: "the order shipped",
        subagentName: "researcher",
      }),
      ctx(),
    );
    expect(sentTo(calls, "record_tool_result")).toMatchObject({
      tool_name: "researcher",
      tool_call_id: "call_sub",
      result: "the order shipped",
    });
  });

  it("reports a tool that returned nothing, because that is still an answer", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "action.result")(
      actionResult({ callId: "c9", kind: "tool-result", output: "", toolName: "search" }),
      ctx(),
    );
    expect(sentTo(calls, "record_tool_result")).toHaveProperty("result", "");
  });

  it("reports a failed action's output too", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "action.result")(
      actionResult(
        {
          callId: "c9", isError: true, kind: "tool-result",
          output: "order service timed out", toolName: "lookup_order",
        },
        { status: "failed" },
      ),
      ctx(),
    );
    expect(sentTo(calls, "record_tool_result")).toMatchObject({
      result: "order service timed out",
    });
  });

  it("honours reportThoughts: false", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk, reportThoughts: false });
    await handler(hooks, "reasoning.completed")(reasoning("private"), ctx());
    expect(calls).toEqual([]);
  });

  it("honours reportToolResults: false but still reports the call", async () => {
    // A tool result is the caller's own customer data. The call itself is not.
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk, reportToolResults: false });
    await handler(hooks, "actions.requested")(actionsRequested([TOOL_CALL]), ctx());
    await handler(hooks, "action.result")(actionResult(TOOL_RESULT), ctx());
    expect(methodsIn(calls)).toEqual(["record_tool_call"]);
  });
});

describe("the end of the session", () => {
  it("closes the Synap session, passing the id positionally", async () => {
    // end_session is the one method on the namespace that does not take an
    // options object. Passing one closes nothing.
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "session.completed")({ type: "session.completed" }, ctx());
    expect(calls).toEqual([{ method: "end_session", options: SESSION }]);
  });

  it("honours endSessionOnComplete: false for a shared conversation", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({
      sdk, conversationId: "conv-shared", endSessionOnComplete: false,
    });
    await handler(hooks, "session.completed")({ type: "session.completed" }, ctx());
    expect(calls).toEqual([]);
  });
});

// ── identity ─────────────────────────────────────────────────────────────────

describe("identity", () => {
  it("reports nothing on an unauthenticated channel with no explicit userId", async () => {
    // Without a user the server cannot say whose event this is, and a misfiled
    // event is worse than no event.
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.received")(userMessage("hi"), ctx({ principalId: null }));
    expect(calls).toEqual([]);
  });

  it("an explicit userId works where the channel has no principal", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk, userId: "bob" });
    await handler(hooks, "message.received")(userMessage("hi"), ctx({ principalId: null }));
    expect(sentTo(calls, "send_message")).toMatchObject({ user_id: "bob" });
  });

  it("an explicit conversationId overrides the eve session id", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk, conversationId: "conv-xyz" });
    await handler(hooks, "message.received")(userMessage("hi"), ctx());
    expect(sentTo(calls, "send_message")).toMatchObject({ conversation_id: "conv-xyz" });
  });

  it("drops an empty customer_id rather than sending it", async () => {
    // A B2C instance refuses a call carrying a customer_id at all.
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.received")(userMessage("hi"), ctx());
    expect(sentTo(calls, "send_message")).not.toHaveProperty("customer_id");
  });

  it("carries the customer_id on a B2B instance", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk, customerId: "cust-9" });
    await handler(hooks, "message.received")(userMessage("hi"), ctx());
    expect(sentTo(calls, "send_message")).toMatchObject({ customer_id: "cust-9" });
  });
});

// ── durability ───────────────────────────────────────────────────────────────

describe("an event that arrives twice is reported once", () => {
  it("de-duplicates a replayed user message on its turn and sequence", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.received")(userMessage("hi", 1), ctx());
    await handler(hooks, "message.received")(userMessage("hi", 1), ctx());
    expect(calls).toHaveLength(1);
  });

  it("still reports a second, different user message in the same turn", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.received")(userMessage("hi", 1), ctx());
    await handler(hooks, "message.received")(userMessage("and one more thing", 7), ctx());
    expect(calls).toHaveLength(2);
  });

  it("de-duplicates a re-delivered action on its call id", async () => {
    // eve says calls may arrive incrementally, so a consumer correlates by
    // call id rather than assuming one event carries them all.
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "actions.requested")(actionsRequested([TOOL_CALL]), ctx());
    await handler(hooks, "actions.requested")(actionsRequested([TOOL_CALL], 6), ctx());
    expect(calls).toHaveLength(1);
  });

  it("does not suppress an event whose send failed, so the retry still goes out", async () => {
    // Remembering a key before the send succeeded would lose the event twice
    // over: once to the failure, once to the de-duplication.
    const calls: StreamCall[] = [];
    let failNext = true;
    const sdk: SynapStreamSdkLike = {
      instance: {
        isListening: () => true,
        send_message: vi.fn(async (options: Record<string, unknown>) => {
          if (failNext) {
            failNext = false;
            throw new Error("stream hiccup");
          }
          calls.push({ method: "send_message", options });
        }),
      },
    };
    const hooks = createSynapStreamHooks({ sdk });
    await handler(hooks, "message.received")(userMessage("hi", 1), ctx());
    await handler(hooks, "message.received")(userMessage("hi", 1), ctx());
    expect(calls).toHaveLength(1);
  });
});

describe("never breaks the run", () => {
  it("a stream that is failing does not stop the rest of the turn", async () => {
    // Every send throws. Nothing reaches the caller, and the second action is
    // still attempted rather than lost to the first one's failure.
    const { sdk, instance } = streamingSdk({ throws: true });
    const hooks = createSynapStreamHooks({ sdk });
    await expect(
      handler(hooks, "actions.requested")(
        actionsRequested([
          TOOL_CALL,
          { callId: "call_2", input: {}, kind: "tool-call", toolName: "send_email" },
        ]),
        ctx(),
      ),
    ).resolves.toBeUndefined();
    expect(instance.record_tool_call).toHaveBeenCalledTimes(2);
  });

  it("an event it cannot read is swallowed, not raised", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await expect(
      handler(hooks, "actions.requested")({ type: "actions.requested" }, ctx()),
    ).resolves.toBeUndefined();
    expect(calls).toEqual([]);
  });

  it("a context with no session reports nothing", async () => {
    const { sdk, calls } = streamingSdk();
    const hooks = createSynapStreamHooks({ sdk });
    await expect(
      handler(hooks, "message.received")(userMessage("hi"), {}),
    ).resolves.toBeUndefined();
    expect(calls).toEqual([]);
  });
});
