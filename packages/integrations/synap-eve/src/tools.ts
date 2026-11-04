// Synap tools for eve agents.
//
// Each factory returns an eve `ToolDefinition` (branded by `defineTool`) meant
// to be the default export of a file under `agent/tools/`, where the filename
// becomes the model-facing tool name:
//
//   // agent/tools/synap_search.ts
//   import { createSynapSearchTool } from "@maximem/synap-eve";
//   import { sdk } from "../lib/synap.js";
//   export default createSynapSearchTool({ sdk });
//
// Identity is auto-scoped from the eve session (`ctx.session.auth.current.principalId`),
// with an explicit `userId` override for unauthenticated channels.
//
// Error policy: reads (search) degrade to an empty result; writes (store) raise
// `SynapIntegrationError` so ingestion outages are observable to the agent.

import { defineTool } from "eve/tools";
import { z } from "zod";

import {
  SynapIntegrationError,
  resolveUserId,
  resolveConversationId,
  type EveSessionLike,
  type SynapSdkLike,
} from "./types.js";
import { assertSdkSupportsScopePath, ensureIdsMatchPath } from "./scope-guard.js";

export interface SynapToolOptions {
  /** Configured Synap SDK. */
  sdk: SynapSdkLike;
  /**
   * Explicit Synap user scope. Required on unauthenticated channels, where
   * `ctx.session.auth.current` is `null`.
   *
   * This *overrides* the eve principal — it is not a fallback. Because the
   * factories are called once per `agent/tools/*.ts` module, setting it on an
   * authenticated multi-user agent pins every session to the same scope. Leave
   * it unset wherever the session carries a real principal.
   */
  userId?: string;
  /** B2B instances only, where it is required. NOT accepted on a B2C
   * instance: the server rejects a call carrying one with HTTP 400. */
  customerId?: string;
  /** Explicit conversation id. Defaults to the eve session id. */
  conversationId?: string;
  /** Synap fetch mode ("accurate" default, or "fast"). */
  mode?: string;
  /**
   * The level of your account's scope ladder to read and write at, for an
   * account whose ladder has a level between customer and user, for example
   * `{ customer: "acme", team: "eng", user: "u1" }`. Fixed when the tool is
   * built: it is not a parameter the model sees, and a `scope_path` in the
   * model's arguments is ignored. Unset on a default ladder, where the calls
   * are exactly what they were. Reads send it as `scope_path`, the store
   * tool as `scope`.
   */
  scopePath?: Record<string, string>;
}

/** `{ [key]: path }` when a path was given and has entries, else `{}`, so an unchanged call stays unchanged. */
function withPath<K extends string>(
  key: K,
  path: Record<string, string> | undefined,
): { [P in K]?: Record<string, string> } {
  return path && Object.keys(path).length > 0
    ? ({ [key]: { ...path } } as { [P in K]?: Record<string, string> })
    : {};
}

const searchInput = z.object({
  query: z.string().describe("What to look up in the user's memory"),
  maxResults: z
    .number()
    .int()
    .positive()
    .optional()
    .describe("Maximum number of results (default 10)"),
});

const storeInput = z.object({
  content: z.string().describe("The fact, preference, or note to persist"),
  metadata: z
    .record(z.string(), z.unknown())
    .optional()
    .describe("Optional metadata to attach"),
});

/**
 * Read tool: search the user's Synap memory for context relevant to a query.
 * Degrades to `{ available: false }` on missing identity or SDK failure — a
 * recall miss never aborts the turn.
 */
export function createSynapSearchTool(options: SynapToolOptions) {
  const { sdk } = options;
  if (!sdk) throw new TypeError("createSynapSearchTool requires a non-null sdk");
  assertSdkSupportsScopePath(sdk, options.scopePath, "createSynapSearchTool");
  ensureIdsMatchPath(options.scopePath, {
    userId: options.userId,
    customerId: options.customerId,
    where: "createSynapSearchTool",
  });

  return defineTool({
    description:
      "Search the user's long-term Synap memory (facts, preferences, " +
      "episodes, emotions, temporal events) for context relevant to a " +
      "query. Use when you need background about the user that is not in " +
      "the current conversation.",
    inputSchema: searchInput,
    outputSchema: z.object({
      formattedContext: z.string(),
      available: z.boolean(),
    }),
    async execute(input, ctx) {
      const query = (input?.query ?? "").trim();
      const maxResults = input?.maxResults ?? 10;
      const userId = resolveUserId(options.userId, ctx as EveSessionLike);
      if (!query || !userId) {
        return { formattedContext: "", available: false };
      }
      // The principal varies per call but the path was fixed at build time, and
      // the server lets the path's `user` / `customer` win over the ids below.
      // Refuse (throw, not degrade) rather than read another identity's memory.
      ensureIdsMatchPath(options.scopePath, {
        userId,
        customerId: options.customerId,
        where: "synap_search",
      });
      try {
        const response = await sdk.fetch({
          conversation_id: resolveConversationId(options.conversationId, ctx as EveSessionLike) ?? null,
          user_id: userId,
          customer_id: options.customerId || null,
          search_query: [query],
          max_results: maxResults,
          mode: options.mode ?? "accurate",
          include_conversation_context: false,
          ...withPath("scope_path", options.scopePath),
        });
        const text = (response.formatted_context ?? "").trim();
        return { formattedContext: text, available: !!text };
      } catch (err) {
        console.error(`[synap-eve] synap_search failed userId=${userId}:`, err);
        return { formattedContext: "", available: false };
      }
    },
  });
}

/**
 * Write tool: persist an explicit fact/preference/note to Synap for future
 * recall. Raises `SynapIntegrationError` on failure (writes never silently
 * drop) — eve surfaces the error result to the model.
 */
export function createSynapStoreTool(options: SynapToolOptions) {
  const { sdk } = options;
  if (!sdk) throw new TypeError("createSynapStoreTool requires a non-null sdk");
  assertSdkSupportsScopePath(sdk, options.scopePath, "createSynapStoreTool");
  ensureIdsMatchPath(options.scopePath, {
    userId: options.userId,
    customerId: options.customerId,
    where: "createSynapStoreTool",
  });

  return defineTool({
    description:
      "Persist an explicit fact, preference, or note to the user's Synap " +
      "memory for future recall. Call this when the user shares something " +
      "worth remembering across sessions.",
    inputSchema: storeInput,
    outputSchema: z.object({
      recorded: z.boolean(),
      ingestionId: z.string().optional(),
    }),
    async execute(input, ctx) {
      const content = (input?.content ?? "").trim();
      const userId = resolveUserId(options.userId, ctx as EveSessionLike);
      if (!content) {
        throw new SynapIntegrationError("synap_store", "missing content");
      }
      if (!userId) {
        throw new SynapIntegrationError(
          "synap_store",
          "no user scope (authenticate the channel or pass userId)",
        );
      }
      // See synap_search: fail closed when the principal is not the path's user.
      ensureIdsMatchPath(options.scopePath, {
        userId,
        customerId: options.customerId,
        where: "synap_store",
      });
      const metadata: Record<string, unknown> = { source: "eve", ...(input?.metadata ?? {}) };
      try {
        const result = await sdk.memories.create({
          document: content,
          user_id: userId,
          customer_id: options.customerId || null,
          metadata,
          ...withPath("scope", options.scopePath),
        });
        return { recorded: true, ingestionId: result.ingestion_id ?? "" };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`[synap-eve] synap_store failed userId=${userId}:`, err);
        throw new SynapIntegrationError("synap_store", msg, err);
      }
    },
  });
}
