/**
 * Report what an agent is doing on the gRPC stream, opening it when needed.
 *
 * The TypeScript mirror of `synap_integrations_common/stream_events.py` and
 * `history.py`. Every integration hooks a different framework, and they all have
 * the same job once a hook fires: turn it into a Synap stream event. A copy of
 * that per integration is a copy per integration of getting the role wrong,
 * forgetting an id, or letting an exception escape into somebody's agent loop.
 *
 * **Three rules, and they are the whole design.**
 *
 * *The stream is the default path.* The first report on an SDK with no stream
 * opens one (`sdk.instance.listen()`), so a caller who never heard of
 * `listen()` still gets turns, tool calls and reasoning to the server, and
 * anticipation bundles back. `SYNAP_AUTO_LISTEN=false` or
 * `setAutoListen(sdk, false)` turns that off.
 *
 * *REST when the stream cannot carry it.* A stream that will not open is not
 * used. Turns then go over REST (`recordTurn`); events that only exist on the
 * stream are dropped. A failed open is retried after a minute, not on every
 * call.
 *
 * *Never throws.* A hook runs inside the framework's own loop. A telemetry
 * call that throws there takes down the agent run, and no context is worth
 * that. Every function swallows everything.
 *
 * The second half of this file (`fetchCompactedHistory`, `spliceHistory`) puts
 * Synap's compaction where the framework's own history would go.
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
    /** A getter on the real SDK (`NAMESPACE.is_listening`), like Python's. */
    is_listening?: boolean;
    /** Not on the published SDK's `instance`; kept for stubs that model it as a method. */
    isListening?: () => boolean;
    listen?: (options?: Record<string, unknown>) => Promise<unknown>;
    send_message?: (options: Record<string, unknown>) => Promise<unknown>;
    record_tool_call?: (options: Record<string, unknown>) => Promise<unknown>;
    record_tool_result?: (options: Record<string, unknown>) => Promise<unknown>;
    record_thinking?: (options: Record<string, unknown>) => Promise<unknown>;
    end_session?: (conversationId: string) => Promise<unknown>;
  };
  conversation?: {
    record_message?: (options: Record<string, unknown>) => Promise<unknown>;
    context?: {
      get_context_for_prompt?: (options: Record<string, unknown>) => Promise<unknown>;
    };
  };
}

/** Ids every report can carry. Absent ones are dropped, never sent as ''. */
export interface StreamScope {
  conversationId?: string;
  userId?: string;
  customerId?: string;
}

const RETRY_AFTER_MS = 60_000;
const OPEN_TIMEOUT_MS = 10_000;

interface AutoStreamState {
  /** `undefined`: follow `SYNAP_AUTO_LISTEN`. */
  enabled?: boolean;
  /** One open at a time, however many reports arrive together. */
  opening?: Promise<boolean>;
  failedAt?: number;
  warned: boolean;
}

const states = new WeakMap<object, AutoStreamState>();

function stateOf(sdk: object): AutoStreamState {
  let state = states.get(sdk);
  if (state === undefined) {
    state = { warned: false };
    states.set(sdk, state);
  }
  return state;
}

/**
 * Turn opening the stream on first report on or off for one SDK.
 *
 * Overrides `SYNAP_AUTO_LISTEN` for that SDK. A stream the caller opened
 * themselves is used either way.
 */
export function setAutoListen(sdk: unknown, enabled: boolean): void {
  if (sdk !== null && typeof sdk === 'object') stateOf(sdk).enabled = enabled;
}

function autoListenEnabled(sdk: object): boolean {
  const set = stateOf(sdk).enabled;
  if (set !== undefined) return set;
  // `process` does not exist on Edge and in Workers; there the stream cannot
  // open anyway, but reading it must not be what throws.
  const raw = typeof process !== 'undefined' ? process.env?.['SYNAP_AUTO_LISTEN'] : undefined;
  return !['0', 'false', 'no', 'off'].includes((raw ?? 'true').trim().toLowerCase());
}

/**
 * Whether there is a stream to report on. Does not open one.
 *
 * ⚠ The real SDK exposes `instance.is_listening` as a GETTER, the same name as
 * Python's. The method `isListening()` exists only on its private `_internal`
 * object, so reading the method here answered "not listening" on every real
 * SDK and left the stream permanently unused. Both are read: the getter first,
 * then the method for a stub that models it that way. Reading a method without
 * calling it gives a function object, which is truthy, so a method is always
 * called.
 */
export function streamIsActive(sdk: unknown): boolean {
  try {
    const instance = (sdk as SynapStreamSdkLike | null)?.instance;
    if (instance === undefined || instance === null) return false;
    const flag = instance.is_listening;
    if (typeof flag === 'boolean') return flag;
    const fn = instance.isListening;
    return typeof fn === 'function' ? fn.call(instance) === true : false;
  } catch {
    return false;
  }
}

/** Rejects after `ms`, and does not keep a plain script alive while waiting. */
function timeoutAfter(ms: number): { promise: Promise<never>; cancel: () => void } {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const promise = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`no answer within ${ms / 1000}s`)), ms);
    (timer as { unref?: () => void }).unref?.();
  });
  return { promise, cancel: () => clearTimeout(timer) };
}

// ⚠ A stream this module opened for the caller must not be what keeps their
// script alive. grpc-js refs the HTTP/2 session for as long as a call is open,
// and the stream is one call that never ends, so a plain `node script.js` that
// has finished its turns would hang forever. The SDK owns the grpc client and
// does not unref it, so the session is unref'd here: `node:http2.connect` is
// wrapped, and a session whose authority is the Synap host gets a `ref()` that
// does nothing. Sessions to any other authority keep theirs. The wrapper stays
// installed so a reconnect is covered too. `synap-vercel-adk` does the same.
const unrefedAuthorities = new Set<string>();
let connectWrapped = false;

async function unrefSessionsTo(authority: string): Promise<void> {
  unrefedAuthorities.add(authority);
  if (connectWrapped) return;
  try {
    // A dynamic import: Edge and Workers have no `node:http2`, and a static one
    // fails the build there.
    const mod = (await import(/* @vite-ignore */ 'node:http2')) as { default?: unknown };
    const http2 = (mod.default ?? mod) as { connect: (...args: unknown[]) => { ref: () => unknown } };
    const real = http2.connect;
    http2.connect = (...args: unknown[]) => {
      const session = real.apply(http2, args);
      if (unrefedAuthorities.has(String(args[0] ?? '').replace(/^[a-z]+:\/\//, ''))) {
        session.ref = function ref(this: unknown) { return this; };
      }
      return session;
    };
    connectWrapped = true;
  } catch {
    // No http2 here, so no session to keep a process alive either.
  }
}

/** Where `instance.listen()` connects with no options, as the SDK resolves it. */
function defaultStreamAuthority(): string {
  const env = typeof process !== 'undefined' ? process.env : undefined;
  const host = env?.['SYNAP_GRPC_HOST'] || 'synap-cloud-prod.maximem.ai';
  const port = Number(env?.['SYNAP_GRPC_PORT']);
  return `${host}:${Number.isFinite(port) && port > 0 ? port : 443}`;
}

async function openStream(sdk: object, state: AutoStreamState): Promise<boolean> {
  const instance = (sdk as SynapStreamSdkLike).instance;
  const timeout = timeoutAfter(OPEN_TIMEOUT_MS);
  try {
    await unrefSessionsTo(defaultStreamAuthority());
    const opened = instance!.listen!.call(instance);
    // If the timeout wins, `listen()` may still settle later; its rejection
    // must not surface as an unhandled one.
    opened.catch(() => {});
    await Promise.race([opened, timeout.promise]);
  } catch (error) {
    // `ListeningAlreadyActiveError` lands here when the caller opened it first.
    if (streamIsActive(sdk)) return true;
    state.failedAt = Date.now();
    if (!state.warned) {
      state.warned = true;
      console.warn(
        `synap: could not open the gRPC stream (${(error as Error)?.message ?? String(error)}); `
        + 'turns go over REST until it opens. SYNAP_AUTO_LISTEN=false stops trying.',
      );
    }
    return false;
  } finally {
    timeout.cancel();
  }
  if (!streamIsActive(sdk)) {
    state.failedAt = Date.now();
    return false;
  }
  state.failedAt = undefined;
  return true;
}

/**
 * Open the stream if there is none. Resolves to whether there is one to use.
 *
 * Never throws. Never waits more than ten seconds, and after a failure does not
 * try again for a minute: until then every turn goes over REST. Concurrent
 * calls share one open.
 */
export async function ensureStream(sdk: unknown): Promise<boolean> {
  try {
    if (sdk === null || typeof sdk !== 'object') return false;
    if (streamIsActive(sdk)) return true;
    const state = stateOf(sdk);
    if (state.opening !== undefined) return await state.opening;
    if (!autoListenEnabled(sdk)) return false;
    if (typeof (sdk as SynapStreamSdkLike).instance?.listen !== 'function') return false;
    if (state.failedAt !== undefined && Date.now() - state.failedAt < RETRY_AFTER_MS) return false;
    state.opening = openStream(sdk, state).finally(() => { state.opening = undefined; });
    return await state.opening;
  } catch {
    // A hook must never break the agent run.
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

/**
 * Write a user or assistant turn: on the stream (opened if need be), over REST
 * when the stream cannot carry it. Resolves to whether it was written.
 *
 * A turn is the conversation itself, not telemetry about it: without it the
 * server has nothing to compact and the SDK nothing to search. Stream first,
 * REST second, never both: the server persists a streamed turn from its event,
 * so a REST write on top would store and extract it twice. Never throws.
 */
export async function recordTurn(
  sdk: unknown,
  args: StreamScope & {
    role: string;
    content: string;
    metadata?: Record<string, string>;
  },
): Promise<boolean> {
  if (typeof args.content !== 'string' || !args.content.trim()) return false;
  if (await reportTurn(sdk, args)) return true;
  // The REST write needs a conversation; the stream would have minted one.
  if (!args.conversationId) return false;
  try {
    const conversation = (sdk as SynapStreamSdkLike | null)?.conversation;
    const record = conversation?.record_message;
    if (typeof record !== 'function') return false;
    const payload: Record<string, unknown> = {
      conversation_id: args.conversationId,
      role: args.role === 'user' ? 'user' : 'assistant',
      content: args.content,
    };
    if (args.userId) payload['user_id'] = args.userId;
    // Refused on a B2C instance, so an empty one is left out, not sent as "".
    if (args.customerId) payload['customer_id'] = args.customerId;
    if (args.metadata && Object.keys(args.metadata).length > 0) payload['metadata'] = args.metadata;
    await record.call(conversation, payload);
    return true;
  } catch (error) {
    // A hook must never break the agent run, but a lost turn is not silent.
    console.warn(
      `synap: turn write failed (conversation_id=${args.conversationId}): `
      + `${(error as Error)?.message ?? String(error)}`,
    );
    return false;
  }
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
  // With no stream, the result is still kept in the SDK's local turn archive
  // (never sent), so history search can find it after a compaction; with a
  // stream, the SDK's record_tool_result archives it itself.
  if (args.conversationId && !(await ensureStream(sdk))) {
    try {
      const archive = (sdk as { archive_tool_result?: (c: string, r: unknown, o: { tool_name?: string }) => void } | null)
        ?.archive_tool_result;
      if (typeof archive === 'function') archive.call(sdk, args.conversationId, args.result, { tool_name: args.toolName });
    } catch {
      // A hook must never break the agent run.
    }
  }
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

/**
 * Close the session for a conversation the integration knows has ended.
 *
 * Never opens a stream: there is no session to end on a new one.
 */
export async function endSession(sdk: unknown, conversationId: string): Promise<boolean> {
  if (!conversationId) return false;
  return sendPositional(sdk, 'end_session', conversationId);
}

// --------------------------------------------------------------- history
//
// The mirror of `synap_integrations_common/history.py`. Every framework keeps
// the conversation as a list of messages and has some way to shrink it when it
// grows. An integration that only adds Synap context beside that list leaves
// the framework's compaction in charge, and sends the recent turns twice: once
// in the list and once inside Synap's block. `spliceHistory` replaces the
// framework's earlier turns with Synap's compaction instead. The framework's
// own compaction stays installed: it sees a short history and does not fire,
// and it is the fallback when Synap has nothing.

/** What the SDK's compaction block starts with. */
const COMPACTION_PREFIX = '## Conversation so far (compacted';
const HISTORY_OPEN = '<synap_conversation_history>';
const HISTORY_CLOSE = '</synap_conversation_history>';

/** The conversation before the current turn, as Synap holds it. */
export interface CompactedHistory {
  /**
   * The compaction block and any earlier turns history search found, wrapped
   * in `<synap_conversation_history>` tags.
   */
  preamble: string;
  /** The turns after the compaction, oldest first. */
  turns: ReadonlyArray<readonly [role: 'user' | 'assistant', text: string]>;
  compactionId?: string | null;
}

/**
 * Whether a message is the history `spliceHistory` put there, not a turn.
 *
 * An integration that writes messages back to Synap skips these, or the
 * compaction would be recorded as something somebody said.
 */
export function isSynapHistory(text: unknown): boolean {
  if (typeof text !== 'string') return false;
  const stripped = text.trimStart();
  return stripped.startsWith(HISTORY_OPEN) || stripped.startsWith(COMPACTION_PREFIX);
}

/**
 * The compacted conversation, or `null` when there is nothing to replace.
 *
 * `null` when the conversation has no compaction yet (the framework's list is
 * still short and is the better record), when the SDK predates compaction
 * blocks, and on any error. Never throws.
 */
export async function fetchCompactedHistory(
  sdk: unknown,
  conversationId: string,
): Promise<CompactedHistory | null> {
  if (!conversationId) return null;
  try {
    const context = (sdk as SynapStreamSdkLike | null)?.conversation?.context;
    const get = context?.get_context_for_prompt;
    if (typeof get !== 'function') return null;
    return compactedHistoryFrom(await get.call(context, { conversation_id: conversationId }));
  } catch (error) {
    // The framework's own history still stands.
    console.warn(
      `synap: compacted history unavailable (conversation_id=${conversationId}): `
      + `${(error as Error)?.message ?? String(error)}`,
    );
    return null;
  }
}

/**
 * `fetchCompactedHistory`'s parsing of a `get_context_for_prompt` response, for
 * a caller that has to make the call itself (to choose a style, or to let an
 * error propagate). `null` when the response carries no compaction block.
 */
export function compactedHistoryFrom(response: unknown): CompactedHistory | null {
  const ctx = (response ?? null) as Record<string, unknown> | null;
  const block = ctx?.['context_block'];
  if (typeof block !== 'string' || !block.trim()) return null;
  const parts = [block.trim()];
  const historyBlock = ctx?.['history_block'];
  if (typeof historyBlock === 'string' && historyBlock.trim()) parts.push(historyBlock.trim());
  const turns: Array<readonly ['user' | 'assistant', string]> = [];
  const recent = ctx?.['recent_messages'];
  for (const rm of Array.isArray(recent) ? recent : []) {
    const role = (rm as { role?: unknown } | null)?.role;
    const content = (rm as { content?: unknown } | null)?.content;
    if ((role === 'user' || role === 'assistant') && typeof content === 'string' && content.trim()) {
      turns.push([role, content]);
    }
  }
  const compactionId = ctx?.['compaction_id'];
  return {
    preamble: `${HISTORY_OPEN}\n${parts.join('\n\n')}\n${HISTORY_CLOSE}`,
    turns,
    compactionId: typeof compactionId === 'string' ? compactionId : null,
  };
}

function norm(text: unknown): string {
  return String(text ?? '').split(/\s+/).filter(Boolean).join(' ');
}

/**
 * Replace every turn before the current one with Synap's history.
 *
 * `roleOf` returns `"system"`, `"user"`, `"assistant"` or anything else (a tool
 * message, say). `makeMessage(role, text)` builds a message of the framework's
 * type; it is called with `"system"` once, for the preamble, and with `"user"`
 * or `"assistant"` for turns.
 *
 * The result, in order:
 *
 * 1. the framework's own system messages from before the current turn;
 * 2. the preamble, as one system message;
 * 3. Synap's turns after the compaction;
 * 4. any earlier turns the framework has and Synap does not yet (a write still
 *    in flight), rebuilt as plain text;
 * 5. the current turn untouched: the last user message and everything after it,
 *    so a tool call and its result stay paired.
 *
 * Tool messages from earlier turns are dropped with the turns they belonged
 * to: the compaction covers them, and history search finds their results.
 */
export function spliceHistory<M>(
  messages: readonly M[],
  history: CompactedHistory,
  fns: {
    roleOf: (message: M) => string;
    textOf: (message: M) => string;
    makeMessage: (role: string, text: string) => M;
  },
): M[] {
  const { roleOf, textOf, makeMessage } = fns;
  let currentStart = messages.length;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (roleOf(messages[i]!) === 'user') {
      currentStart = i;
      break;
    }
  }
  const earlier = messages.slice(0, currentStart);
  const current = messages.slice(currentStart);

  const systems = earlier.filter((m) => roleOf(m) === 'system' && !isSynapHistory(textOf(m)));
  const frameworkTurns: Array<[string, string]> = earlier
    .filter((m) => {
      const role = roleOf(m);
      return (role === 'user' || role === 'assistant') && norm(textOf(m)) !== '';
    })
    .map((m) => [roleOf(m), textOf(m)]);

  const turns: Array<[string, string]> = history.turns.map(([r, t]) => [r, t]);
  // The current user message is usually recorded before the model is called,
  // so Synap already has it, and maybe a reply from an earlier lap of this
  // turn. Both belong to the current turn, which the framework still holds.
  if (current.length > 0 && roleOf(current[0]!) === 'user') {
    const now = norm(textOf(current[0]!));
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      if (turns[i]![0] === 'user' && norm(turns[i]![1]) === now) {
        turns.length = i;
        break;
      }
    }
  }

  // Turns the framework has after the last one Synap has.
  let pending: Array<[string, string]> = [];
  if (turns.length > 0) {
    const [lastRole, lastTextRaw] = turns[turns.length - 1]!;
    const lastText = norm(lastTextRaw);
    for (let i = frameworkTurns.length - 1; i >= 0; i -= 1) {
      const [role, text] = frameworkTurns[i]!;
      if (role === lastRole && norm(text) === lastText) {
        pending = frameworkTurns.slice(i + 1);
        break;
      }
    }
  }

  return [
    ...systems,
    makeMessage('system', history.preamble),
    ...turns.map(([role, text]) => makeMessage(role, text)),
    ...pending.map(([role, text]) => makeMessage(role, String(text))),
    ...current,
  ];
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
 * One call site for every report, so the rules hold in one place.
 */
async function send(
  sdk: unknown,
  method: 'send_message' | 'record_tool_call' | 'record_tool_result' | 'record_thinking',
  options: Record<string, unknown>,
): Promise<boolean> {
  if (!(await ensureStream(sdk))) return false;
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
