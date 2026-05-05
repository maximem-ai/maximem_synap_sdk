// Synap Mastra Tools — `synapSearchTool` and `synapStoreTool`.
//
// Register in the Agent's tools map so the agent can explicitly read/write
// Synap memory via tool calls. Complements (and can be used independently
// of) SynapMemory.

import { createTool } from "@mastra/core/tools";
import { z } from "zod";

import type { SynapIdentityOptions } from "./types.js";

export interface SynapToolsOptions extends SynapIdentityOptions {
  /**
   * Optional conversation id. When set, `synapStoreTool` records writes
   * against it; when unset, writes create free-standing memories without
   * a conversation anchor.
   */
  conversationId?: string;
}

export function synapSearchTool(options: SynapToolsOptions) {
  const { sdk, userId, customerId = "", conversationId, mode = "accurate" } = options;
  if (!sdk) throw new Error("synapSearchTool requires a non-null sdk");
  if (!userId) throw new Error("synapSearchTool requires a non-empty userId");

  return createTool({
    id: "synap_search",
    description:
      "Search the user's Synap memory (facts, preferences, episodes, " +
      "emotions, temporal events) for context relevant to a query. Use " +
      "when you need background about the user that isn't in the " +
      "current conversation.",
    inputSchema: z.object({
      query: z.string().describe("Search query"),
      maxResults: z
        .number()
        .int()
        .positive()
        .optional()
        .describe("Maximum number of results (default 10)"),
    }),
    outputSchema: z.object({
      formattedContext: z.string(),
      available: z.boolean(),
    }),
    execute: async (input: { query?: string; maxResults?: number }) => {
      const query = typeof input?.query === "string" ? input.query : "";
      const maxResults = typeof input?.maxResults === "number" ? input.maxResults : 10;
      if (!query) {
        return { formattedContext: "", available: false };
      }
      try {
        const response = await sdk.fetch({
          conversation_id: conversationId ?? null,
          user_id: userId,
          customer_id: customerId || null,
          search_query: [query],
          max_results: maxResults,
          mode,
          include_conversation_context: false,
        });
        const text = (response.formatted_context ?? "").trim();
        return { formattedContext: text, available: !!text };
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error(`synap_search: sdk.fetch failed userId=${userId}`, err);
        return { formattedContext: "", available: false };
      }
    },
  });
}

export function synapStoreTool(options: SynapToolsOptions) {
  const { sdk, userId, customerId = "" } = options;
  if (!sdk) throw new Error("synapStoreTool requires a non-null sdk");
  if (!userId) throw new Error("synapStoreTool requires a non-empty userId");

  return createTool({
    id: "synap_store",
    description:
      "Persist an explicit fact, preference, or note to the user's " +
      "Synap memory for future recall. Call this when the user shares " +
      "something worth remembering across sessions.",
    inputSchema: z.object({
      content: z.string().describe("The fact or note to persist"),
      metadata: z
        .record(z.string(), z.unknown())
        .optional()
        .describe("Optional metadata to attach"),
    }),
    outputSchema: z.object({
      recorded: z.boolean(),
      ingestionId: z.string().optional(),
      error: z.string().optional(),
    }),
    execute: async (input: { content?: string; metadata?: Record<string, unknown> }) => {
      const content = typeof input?.content === "string" ? input.content : "";
      if (!content.trim()) {
        return { recorded: false, error: "missing content" };
      }
      const metadata: Record<string, unknown> = { ...(input?.metadata ?? {}) };
      if (metadata.source === undefined) {
        metadata.source = "mastra";
      }
      try {
        const result = await sdk.memories.create({
          document: content,
          user_id: userId,
          customer_id: customerId || null,
          metadata,
        });
        return { recorded: true, ingestionId: result.ingestion_id ?? "" };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        // eslint-disable-next-line no-console
        console.error(
          `synap_store: sdk.memories.create failed userId=${userId}`,
          err,
        );
        // Write-side: surface the error to the agent so ingestion outages
        // are observable (matches Python synap_remember's isError path).
        throw new Error(`synap_store: ${msg}`);
      }
    },
  });
}
