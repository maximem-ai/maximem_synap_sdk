// createSynapMcpServer — TypeScript mirror of the Python equivalent.
//
// Builds an in-process MCP server with two tools that let the Claude agent
// read/write Synap memory explicitly:
//   - synap_search(query, max_results?) → formatted context
//   - synap_remember(content, metadata?) → memories.create
//
// `buildSynapTools` is exposed so smoke tests can invoke handlers directly
// without spinning up the MCP server runtime.

import {
  createSdkMcpServer,
  tool,
  type McpSdkServerConfigWithInstance,
  type SdkMcpToolDefinition,
} from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";

import type { SynapIdentityOptions } from "./types.js";

export interface CreateSynapMcpServerOptions extends SynapIdentityOptions {
  /** Server name (default: "synap"). Affects the mcp__<name>__<tool> prefix. */
  name?: string;
  /** Server version. */
  version?: string;
  /** Synap fetch mode ("accurate" default, or "fast"). */
  mode?: string;
}

interface ToolContext {
  sdk: CreateSynapMcpServerOptions["sdk"];
  userId: string;
  customerId: string;
  conversationId?: string;
  mode: string;
}

/**
 * Build the raw tool definitions. Exposed for smoke tests and advanced
 * integrations that want to compose with additional tools before wrapping
 * in an MCP server.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function buildSynapTools(ctx: ToolContext): SdkMcpToolDefinition<any>[] {
  const searchSchema = {
    query: z.string().describe("Search query"),
    max_results: z
      .number()
      .int()
      .positive()
      .optional()
      .describe("Maximum number of results (default: 10)"),
  };

  const rememberSchema = {
    content: z.string().describe("Fact or note to persist"),
    metadata: z
      .record(z.string(), z.unknown())
      .optional()
      .describe("Optional metadata to attach"),
  };

  const synapSearch = tool(
    "synap_search",
    "Search the user's Synap memory (facts, preferences, episodes, " +
      "emotions, temporal events) for context relevant to a query. Use " +
      "this when you need background about the user that isn't in the " +
      "current conversation.",
    searchSchema,
    async (args) => {
      const query = args.query;
      const maxResults = args.max_results ?? 10;
      if (!query) {
        return {
          content: [
            { type: "text" as const, text: "synap_search: missing `query` argument." },
          ],
          isError: true,
        };
      }

      try {
        const response = await ctx.sdk.fetch({
          conversation_id: ctx.conversationId ?? null,
          user_id: ctx.userId,
          customer_id: ctx.customerId || null,
          search_query: [query],
          max_results: maxResults,
          mode: ctx.mode,
          include_conversation_context: false,
        });
        const text = (response.formatted_context ?? "").trim();
        return {
          content: [
            {
              type: "text" as const,
              text: text || "synap_search: no relevant context.",
            },
          ],
        };
      } catch (err) {
        const name = err instanceof Error ? err.constructor.name : "Error";
        // eslint-disable-next-line no-console
        console.error(
          `synap_search: sdk.fetch failed userId=${ctx.userId}`,
          err,
        );
        return {
          content: [
            {
              type: "text" as const,
              text: `synap_search: no context available (${name}).`,
            },
          ],
        };
      }
    },
  );

  const synapRemember = tool(
    "synap_remember",
    "Persist an explicit fact, preference, or note to the user's " +
      "Synap memory for future recall. Call this when the user shares " +
      "something worth remembering across sessions.",
    rememberSchema,
    async (args) => {
      const content = args.content;
      if (!content || !content.trim()) {
        return {
          content: [
            { type: "text" as const, text: "synap_remember: missing `content` argument." },
          ],
          isError: true,
        };
      }
      const metadata: Record<string, unknown> = { ...(args.metadata ?? {}) };
      if (metadata.source === undefined) {
        metadata.source = "claude_agent_sdk";
      }

      try {
        const result = await ctx.sdk.memories.create({
          document: content,
          user_id: ctx.userId,
          customer_id: ctx.customerId || null,
          metadata,
        });
        return {
          content: [
            {
              type: "text" as const,
              text: `synap_remember: recorded (ingestion_id=${result.ingestion_id ?? ""}).`,
            },
          ],
        };
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error(
          `synap_remember: sdk.memories.create failed userId=${ctx.userId}`,
          err,
        );
        return {
          content: [
            {
              type: "text" as const,
              text: `synap_remember: ingestion failed (${err instanceof Error ? err.message : String(err)}).`,
            },
          ],
          isError: true,
        };
      }
    },
  );

  return [synapSearch, synapRemember];
}

export function createSynapMcpServer(
  options: CreateSynapMcpServerOptions,
): McpSdkServerConfigWithInstance {
  const {
    sdk,
    userId,
    customerId = "",
    conversationId,
    name = "synap",
    version = "0.1.0",
    mode = "accurate",
  } = options;

  if (!sdk) {
    throw new Error("createSynapMcpServer requires a non-null sdk");
  }
  if (!userId) {
    throw new Error("createSynapMcpServer requires a non-empty userId");
  }

  const tools = buildSynapTools({
    sdk,
    userId,
    customerId,
    conversationId,
    mode,
  });

  return createSdkMcpServer({ name, version, tools });
}
