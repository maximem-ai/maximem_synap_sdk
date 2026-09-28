/**
 * Report what an agent is doing, when there is a stream to report it on.
 *
 * The TypeScript mirror of `synap_integrations_common/stream_events.py`. Every
 * integration hooks a different framework, and they all have the same job once
 * a hook fires: turn it into a Synap stream event. A copy of that per
 * integration is a copy per integration of getting the role wrong, forgetting
 * an id, or letting an exception escape into somebody's agent loop.
 *
 * **Two rules, and they are the whole design.**
 *
 * *Silent when there is no stream.* These calls only work while
 * `sdk.instance.listen()` is running. Most callers never start one, and for
 * them every function here is a no-op. An integration must behave exactly as
 * it did before for a user who has not opted into streaming.
 *
 * *Never throws.* A hook runs inside the framework's own loop. A telemetry
 * call that throws there takes down the agent run, and no context is worth
 * that. Every function swallows everything.
 *
 * ---
 *
 * ⚠ **This file is vendored, not imported.** The three TypeScript integrations
 * (`claude-agent-ts`, `mastra`, `eve`) deliberately carry no runtime
 * dependencies: they duck-type the SDK so a user brings their own version of
 * it. A shared package would mean a new public npm name that has to be
 * published before any of the three can be. So this file is byte-identical in
 * all three, and `stream-events-is-vendored.test.ts` fails the build if the
 * copies drift. Change one, copy it to the other two.
 *
 * ⚠ **Why a helper at all, when `vercel-adk` hand-rolls it.** Hand-rolling is
 * what left `strands-agents` calling `send_message({event_type: 'tool_call'})`,
 * which has no field for a `tool_call_id`, so its tool results had nothing to
 * pair with. The typed methods set the id and set `role` and `event_type`
 * together, which is the pair that was silently wrong in our own published
 * example.
 */

/**
 * The slice of the SDK this file touches. Duck-typed for the same reason the
 * rest of these packages are: a user passes the real `@maximem/synap-js-sdk`,
 * a test passes a stub, and neither is imported here.
 *
 * Everything is optional. An older SDK with no `instance` namespace must make
 * `streamIsActive` answer no, not throw.
 */
export interface SynapStreamSdkLike {
  instance?: {
    isListening?: () => boolean;
    send_message?: (options: Record<string, unknown>) => Promise<unknown>;
    record_tool_call?: (options: Record<string, unknown>) => Promise<unknown>;
    record_tool_result?: (options: Record<string, unknown>) => Promise<unknown>;
    record_thinking?: (options: Record<string, unknown>) => Promise<unknown>;
    end_session?: (conversationId: string) => Promise<unknown>;
  };
}

/** Ids every report can carry. Absent ones are dropped, never sent as ''. */
export interface StreamScope {
  conversationId?: string;
  userId?: string;
  customerId?: string;
}

/**
 * Whether there is a stream to report on.
 *
 * ⚠ `isListening` is a METHOD on the JS SDK and a PROPERTY (`is_listening`) on
 * the Python one. Reading it without calling it gives a function object, which
 * is truthy, so every report would be attempted against a closed stream.
 */
export function streamIsActive(sdk: unknown): boolean {
  try {
    const fn = (sdk as SynapStreamSdkLike | null)?.instance?.isListening;
    return typeof fn === 'function' ? fn.call((sdk as SynapStreamSdkLike).instance) === true : false;
  } catch {
    return false;
  }
}

/**
 * Report a user or assistant turn. Resolves to whether it was sent.
 *
 * `assistant_message` is the event the anticipation agent acts on: it is the
 * moment a turn ends and the next one can be predicted. An integration that
 * reports everything else and not this one gets no anticipation at all.
 */
export async function reportTurn(
  sdk: unknown,
  args: StreamScope & {
    role: string;
    content: string;
    metadata?: Record<string, string>;
  },
): Promise<boolean> {
  const isUser = args.role === 'user';
  return send(sdk, 'send_message', {
    content: args.content,
    role: isUser ? 'user' : 'assistant',
    event_type: isUser ? 'user_message' : 'assistant_message',
    metadata: args.metadata ?? {},
    ...scope(args),
  });
}

/** Report a tool the agent is invoking. */
export async function reportToolCall(
  sdk: unknown,
  args: StreamScope & {
    toolName: string;
    /** Only a plain object travels. See `asArgsObject`. */
    toolArgs?: unknown;
    toolCallId?: string;
  },
): Promise<boolean> {
  return send(sdk, 'record_tool_call', {
    tool_name: args.toolName,
    tool_args: asArgsObject(args.toolArgs),
    tool_call_id: args.toolCallId,
    ...scope(args),
  });
}

/**
 * Report what a tool returned.
 *
 * ⚠ A tool result is usually the caller's customer data. It is an anticipation
 * hint and never becomes a long-term memory, but it does leave their process.
 * An integration passes through what the framework gives it and does not go
 * looking for more.
 */
export async function reportToolResult(
  sdk: unknown,
  args: StreamScope & {
    result: unknown;
    toolName?: string;
    toolCallId?: string;
  },
): Promise<boolean> {
  return send(sdk, 'record_tool_result', {
    result: args.result,
    tool_name: args.toolName,
    tool_call_id: args.toolCallId,
    ...scope(args),
  });
}

/**
 * Report one reasoning step, when the framework exposes one.
 *
 * Most providers return a summary rather than raw reasoning, and some return
 * nothing. An integration reports what it is given; a turn with no reasoning is
 * an ordinary turn, not a gap to fill.
 */
export async function reportReasoning(
  sdk: unknown,
  args: StreamScope & {
    content: string;
    stepIndex?: number;
    thoughtType?: string;
  },
): Promise<boolean> {
  if (!args.content) return false;
  return send(sdk, 'record_thinking', {
    content: args.content,
    step_index: args.stepIndex,
    thought_type: args.thoughtType,
    ...scope(args),
  });
}

/** Close the session for a conversation the integration knows has ended. */
export async function endSession(sdk: unknown, conversationId: string): Promise<boolean> {
  if (!conversationId) return false;
  return sendPositional(sdk, 'end_session', conversationId);
}

// --------------------------------------------------------------- internals

function scope(args: StreamScope): Record<string, unknown> {
  return {
    conversation_id: args.conversationId,
    user_id: args.userId,
    customer_id: args.customerId,
  };
}

/**
 * The SDK forwards `tool_args` only when it is a plain object, and drops
 * anything else on the floor without a word.
 *
 * Frameworks disagree about what a tool's arguments are: LangChain hands over a
 * dict, LiveKit hands over a JSON **string**, and Semantic Kernel hands over a
 * `Mapping` that is not a plain object. Passing those through unchanged sends a
 * tool call with no arguments at all, which looks like a tool that takes none.
 */
function asArgsObject(value: unknown): Record<string, unknown> | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'string') {
    try {
      const parsed: unknown = JSON.parse(value);
      return isPlainObject(parsed) ? parsed : { input: value };
    } catch {
      return { input: value };
    }
  }
  // ⚠ Before `isPlainObject`. A Map is `typeof 'object'` and is not an array,
  // so the plain-object branch matches it and it reaches the SDK as `{}`: a
  // tool call that claims the tool took no arguments.
  if (value instanceof Map) return Object.fromEntries(value) as Record<string, unknown>;
  if (isPlainObject(value)) return value;
  return { input: value };
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/**
 * Fields that are dropped when empty rather than sent as `''`.
 *
 * A B2C instance refuses a call carrying a `customer_id` at all, so sending an
 * empty one turns an ordinary turn into a refusal.
 *
 * ⚠ This is a list and not "drop every empty value" on purpose. `result` and
 * `content` are the payload, not an id: a tool that legitimately returns `''`
 * or `null` has said something, and dropping it would turn a real answer into
 * a malformed call. Python cannot hit this because it passes both of those
 * positionally, so a blanket rule here would be a silent divergence between
 * the two SDKs about what an empty tool result means.
 */
const DROP_WHEN_EMPTY = new Set([
  'conversation_id', 'user_id', 'customer_id',
  'tool_call_id', 'tool_name', 'thought_type', 'step_index',
]);

/**
 * One call site for every report, so the two rules hold in one place.
 */
async function send(
  sdk: unknown,
  method: 'send_message' | 'record_tool_call' | 'record_tool_result' | 'record_thinking',
  options: Record<string, unknown>,
): Promise<boolean> {
  if (!streamIsActive(sdk)) return false;
  const cleaned: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(options)) {
    if (v === undefined) continue;
    if (DROP_WHEN_EMPTY.has(k) && (v === null || v === '')) continue;
    cleaned[k] = v;
  }
  try {
    const fn = (sdk as SynapStreamSdkLike).instance?.[method];
    if (typeof fn !== 'function') return false;
    await fn.call((sdk as SynapStreamSdkLike).instance, cleaned);
    return true;
  } catch {
    // A hook must never break the agent run.
    return false;
  }
}

async function sendPositional(
  sdk: unknown,
  method: 'end_session',
  arg: string,
): Promise<boolean> {
  if (!streamIsActive(sdk)) return false;
  try {
    const fn = (sdk as SynapStreamSdkLike).instance?.[method];
    if (typeof fn !== 'function') return false;
    await fn.call((sdk as SynapStreamSdkLike).instance, arg);
    return true;
  } catch {
    return false;
  }
}
