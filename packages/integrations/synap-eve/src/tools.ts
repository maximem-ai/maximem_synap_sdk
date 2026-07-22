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
// Identity is auto-scoped from the eve session (`ctx.session.auth.principalId`),
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

export interface SynapToolOptions {
  /** Configured Synap SDK. */
  sdk: SynapSdkLike;
  /** Explicit Synap user scope. Overrides the eve principal; required on unauthenticated channels. */
  userId?: string;
  /** Optional customer/org scope. */
  customerId?: string;
  /** Explicit conversation id. Defaults to the eve session id. */
  conversationId?: string;
  /** Synap fetch mode ("accurate" default, or "fast"). */
  mode?: string;
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
      try {
        const response = await sdk.fetch({
          conversation_id: resolveConversationId(options.conversationId, ctx as EveSessionLike) ?? null,
          user_id: userId,
          customer_id: options.customerId || null,
          search_query: [query],
          max_results: maxResults,
          mode: options.mode ?? "accurate",
          include_conversation_context: false,
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
      const metadata: Record<string, unknown> = { source: "eve", ...(input?.metadata ?? {}) };
      try {
        const result = await sdk.memories.create({
          document: content,
          user_id: userId,
          customer_id: options.customerId || null,
          metadata,
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
