// Report an eve turn on the Synap stream.
//
// This package gave an eve agent two tools and a short-term-context resolver.
// Synap saw whatever the model chose to search for and never saw the run
// itself: not the user's turn, not the reply, not the tools, not the
// reasoning. Anticipation had nothing to work with, so every fetch was a cold
// retrieval.
//
// `createSynapStreamHooks` fills that in. eve's `agent/hooks/` extension point
// subscribes to the runtime's own stream events, which fire *after* eve has
// accepted and durably recorded each one, and they carry every event Synap
// wants:
//
//   =========================  =============================================
//   eve stream event            what it is
//   =========================  =============================================
//   `message.received`          the user's turn
//   `message.completed`         one completed assistant message
//   `reasoning.completed`       one completed reasoning block
//   `actions.requested`         the tools (and subagents) the model called
//   `action.result`             what those returned
//   `session.completed`         the session is over
//   =========================  =============================================
//
// Every action and its result carry eve's own `callId`, so a call and its
// result travel with a shared `tool_call_id` and the anticipation agent can
// tell they belong together.
//
//   // agent/hooks/synap.ts
//   import { createSynapStreamHooks } from "@maximem/synap-eve";
//   import { sdk } from "../lib/synap.js";
//   export default createSynapStreamHooks({ sdk });
//
// ⚠ **Stream only, and that is deliberate.** Unlike the other integrations
// there is no REST fallback here, because this package has never written
// conversation turns: `SynapSdkLike` has no `conversation.record_message` and
// an eve agent's turns have never reached Synap as conversation. Adding a REST
// write would change what happens for someone who has not opted into
// streaming, which is the one thing this work must not do. Without an active
// `sdk.instance.listen()` every handler below is a no-op.
//
// The second rule is the same one everywhere: *never throws*. A hook runs
// inside eve's own accepted-event path, and an exception from telemetry there
// is not worth any amount of context.

import { defineHook, type HookContext, type HookEvent } from "eve/hooks";

import {
  endSession,
  reportReasoning,
  reportToolCall,
  reportToolResult,
  reportTurn,
  streamIsActive,
  type SynapStreamSdkLike,
} from "./stream-events.js";
import {
  resolveConversationId,
  resolveUserId,
  type EveSessionLike,
} from "./types.js";

/** One action the model asked for, as `actions.requested` carries it. */
type EveActionRequest = HookEvent<"actions.requested">["data"]["actions"][number];
/** One action's outcome, as `action.result` carries it. */
type EveActionResult = HookEvent<"action.result">["data"]["result"];

export interface SynapStreamHooksOptions {
  /**
   * Configured Synap SDK. Only its `instance` namespace is used, so the full
   * SDK, or anything carrying that namespace, is accepted. Without an active
   * `listen()` stream every handler is a no-op.
   */
  sdk: SynapStreamSdkLike;
  /**
   * Explicit Synap user scope. Required on unauthenticated channels, where
   * `ctx.session.auth.current` is `null`.
   *
   * This *overrides* the eve principal — it is not a fallback. The factory is
   * called once per `agent/hooks/*.ts` module, so setting it on an
   * authenticated multi-user agent pins every session to the same scope.
   */
  userId?: string;
  /** B2B instances only, where it is required. NOT accepted on a B2C
   * instance: the server rejects a call carrying one with HTTP 400. */
  customerId?: string;
  /** Explicit conversation id. Defaults to the eve session id. */
  conversationId?: string;
  /**
   * Whether tool results are reported. On by default, because a result is what
   * tells anticipation how the turn is actually going.
   *
   * ⚠ A tool result is usually your own customer's data. It is an anticipation
   * hint and never becomes a long-term memory, but it does leave your process.
   * Turn this off if that is not something you want to send.
   */
  reportToolResults?: boolean;
  /**
   * Whether `reasoning.completed` is reported as a reasoning step. On by
   * default. Most providers return a summary rather than raw reasoning and
   * some return nothing; a turn without reasoning is an ordinary turn.
   */
  reportThoughts?: boolean;
  /**
   * Whether `session.completed` closes the Synap session. On by default: an
   * eve session is the conversation, and when it ends the conversation has.
   *
   * ⚠ Turn this off if you pass an explicit `conversationId` that several eve
   * sessions share, where one session ending does not mean the conversation
   * has.
   */
  endSessionOnComplete?: boolean;
}

/**
 * How many reported events this remembers, so a replayed event is not reported
 * twice. Bounded because one hook module serves every session in the process.
 */
const REMEMBERED_EVENTS = 512;

/** A bounded set with oldest-out eviction. */
class RecentKeys {
  private readonly keys = new Set<string>();
  private readonly order: string[] = [];

  has(key: string): boolean {
    return this.keys.has(key);
  }

  add(key: string): void {
    if (this.keys.has(key)) return;
    this.keys.add(key);
    this.order.push(key);
    if (this.order.length > REMEMBERED_EVENTS) {
      const oldest = this.order.shift();
      if (oldest !== undefined) this.keys.delete(oldest);
    }
  }
}

interface Scope {
  conversationId: string;
  userId: string;
  customerId: string;
}

/**
 * Build the Synap stream hooks for `agent/hooks/`.
 *
 * Returns an eve `HookDefinition`. Export it as the default export of a file
 * under `agent/hooks/` and eve subscribes it to the runtime stream.
 */
export function createSynapStreamHooks(options: SynapStreamHooksOptions) {
  if (!options?.sdk) {
    throw new TypeError("createSynapStreamHooks requires a non-null sdk");
  }
  const sdk = options.sdk;
  const sent = new RecentKeys();

  /**
   * The ids every event carries, or `undefined` when this turn cannot be
   * scoped. Without a user and a conversation the server cannot say whose
   * event this is, so nothing is reported rather than something misfiled.
   */
  function scope(ctx: HookContext): Scope | undefined {
    const session = ctx as unknown as EveSessionLike;
    const userId = resolveUserId(options.userId, session);
    const conversationId = resolveConversationId(options.conversationId, session);
    if (!userId || !conversationId) return undefined;
    return { conversationId, userId, customerId: options.customerId ?? "" };
  }

  /**
   * Run one handler's body. Wrapped whole: `stream-events` already swallows
   * what it does, but reading the event's shape is ours, and a throw here
   * lands in eve's accepted-event path.
   *
   * The `streamIsActive` check is first so that a caller with no stream — which
   * is every caller who has not opted in — does no work at all.
   */
  async function guard(
    ctx: HookContext,
    body: (scoped: Scope) => Promise<void>,
  ): Promise<void> {
    try {
      if (!streamIsActive(sdk)) return;
      const scoped = scope(ctx);
      if (!scoped) return;
      await body(scoped);
    } catch {
      // A hook must never break the run.
    }
  }

  /** Whether this is the first time we are reporting `key`. */
  function first(key: string): boolean {
    return !sent.has(key);
  }

  return defineHook({
    events: {
      "message.received": (event, ctx) =>
        guard(ctx, async (scoped) => {
          const content = (event.data.message ?? "").trim();
          if (!content) return;
          const key = `user:${event.data.turnId}:${event.data.sequence}`;
          if (!first(key)) return;
          if (await reportTurn(sdk, { ...scoped, role: "user", content })) {
            sent.add(key);
          }
        }),

      "reasoning.completed": (event, ctx) =>
        guard(ctx, async (scoped) => {
          if (options.reportThoughts === false) return;
          const content = (event.data.reasoning ?? "").trim();
          if (!content) return;
          const key = `think:${event.data.turnId}:${event.data.sequence}`;
          if (!first(key)) return;
          const ok = await reportReasoning(sdk, {
            ...scoped,
            content,
            // eve's own index for the model call this reasoning belongs to.
            // Step 0 is the first step, not a missing one.
            stepIndex: event.data.stepIndex,
            thoughtType: "model_thought",
          });
          if (ok) sent.add(key);
        }),

      "actions.requested": (event, ctx) =>
        guard(ctx, async (scoped) => {
          for (const action of event.data.actions ?? []) {
            const key = `call:${action.callId}`;
            if (!first(key)) continue;
            const ok = await reportToolCall(sdk, {
              ...scoped,
              toolName: actionName(action),
              toolArgs: action.input,
              // eve's own id, and the same value comes back on the result.
              // Nothing else pairs the two.
              toolCallId: action.callId,
            });
            if (ok) sent.add(key);
          }
        }),

      "action.result": (event, ctx) =>
        guard(ctx, async (scoped) => {
          if (options.reportToolResults === false) return;
          const result = event.data.result;
          if (!result) return;
          const key = `result:${result.callId}`;
          if (!first(key)) return;
          const ok = await reportToolResult(sdk, {
            ...scoped,
            // `output` and not the whole result object: the envelope is eve's
            // bookkeeping, and what the tool actually said is inside it.
            result: result.output,
            toolName: resultName(result),
            toolCallId: result.callId,
          });
          if (ok) sent.add(key);
        }),

      "message.completed": (event, ctx) =>
        guard(ctx, async (scoped) => {
          const content = (event.data.message ?? "").trim();
          if (!content) return;
          // One turn can complete more than one assistant message: eve emits
          // this whenever an assistant step finishes with visible text, which
          // includes the reply a model gives before it calls a tool. Each is
          // text the user saw, so each is a turn, and the last of them is the
          // `assistant_message` that tells anticipation the turn has ended.
          const key = `assistant:${event.data.turnId}:${event.data.sequence}`;
          if (!first(key)) return;
          if (await reportTurn(sdk, { ...scoped, role: "assistant", content })) {
            sent.add(key);
          }
        }),

      "session.completed": (_event, ctx) =>
        guard(ctx, async (scoped) => {
          if (options.endSessionOnComplete === false) return;
          const key = `end:${scoped.conversationId}`;
          if (!first(key)) return;
          if (await endSession(sdk, scoped.conversationId)) sent.add(key);
        }),
    },
  });
}

// --------------------------------------------------------------- internals

/**
 * The name to report one requested action under.
 *
 * A `tool-call` is one action kind. A subagent or a remote agent is a tool
 * from the model's side — it called something by name and is waiting for an
 * answer — so it is reported as one rather than dropped, which would leave an
 * agent that delegates everything reporting no actions at all.
 */
function actionName(action: EveActionRequest): string {
  switch (action.kind) {
    case "tool-call":
      return action.toolName;
    case "subagent-call":
      return action.subagentName || action.name;
    case "remote-agent-call":
      return action.remoteAgentName || action.name;
    case "load-skill":
      return "load-skill";
    default:
      return "tool";
  }
}

/** The name to report one action result under. Pairs with `actionName`. */
function resultName(result: EveActionResult): string {
  switch (result.kind) {
    case "tool-result":
      return result.toolName;
    case "subagent-result":
      return result.subagentName;
    case "load-skill-result":
      return result.name ?? "load-skill";
    default:
      return "tool";
  }
}
