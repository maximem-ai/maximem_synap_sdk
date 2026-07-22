// Shared mock helpers for synap-eve tests. No shared `synap-integrations-common`
// exists for TS packages, so the mock/failing SDK and the eve-context stub live
// here (mirrors synap-mastra's helpers, trimmed to the surface synap-eve uses).

import { vi } from "vitest";
import type {
  SynapSdkLike,
  SynapFetchResponseLike,
  SynapPromptContext,
  SynapMemoryCreateResult,
  EveSessionLike,
} from "../types.js";

// ── Synap response factories ──────────────────────────────────────────────────

export function RICH(): SynapFetchResponseLike {
  return {
    formatted_context:
      "## User Context\n### Facts\n- Senior engineer at Acme\n- Timezone PT\n",
    facts: [{ id: "f1", content: "User is a senior engineer at Acme", confidence: 0.92 }],
    preferences: [{ id: "p1", content: "Prefers email over Slack", strength: 0.9 }],
    episodes: [],
    emotions: [],
    temporal_events: [],
  };
}

export function EMPTY(): SynapFetchResponseLike {
  return { formatted_context: "", facts: [], preferences: [], episodes: [], emotions: [], temporal_events: [] };
}

export function promptContext(available = true): SynapPromptContext {
  return {
    formatted_context: available ? "User is a senior engineer. Prefers email updates." : "",
    available,
    recent_messages: [],
    recent_message_count: 0,
    total_message_count: 0,
  };
}

// ── SDK mocks ─────────────────────────────────────────────────────────────────

export interface MakeSdkOptions {
  fetchResponse?: SynapFetchResponseLike;
  promptCtx?: SynapPromptContext;
  ingestionId?: string;
}

export function makeSdk(opts: MakeSdkOptions = {}): SynapSdkLike {
  const fetchResponse = opts.fetchResponse ?? RICH();
  const ctx = opts.promptCtx ?? promptContext();
  const ingestionId = opts.ingestionId ?? "ing-test-001";
  return {
    fetch: vi.fn().mockResolvedValue(fetchResponse),
    conversation: {
      context: { get_context_for_prompt: vi.fn().mockResolvedValue(ctx) },
    },
    memories: {
      create: vi.fn().mockResolvedValue({ ingestion_id: ingestionId } satisfies SynapMemoryCreateResult),
    },
  };
}

export function makeFailingSdk(err?: Error): SynapSdkLike {
  const e = err ?? new Error("simulated-sdk-failure");
  const sdk = makeSdk();
  (sdk.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(e);
  (sdk.conversation.context.get_context_for_prompt as ReturnType<typeof vi.fn>).mockRejectedValue(e);
  (sdk.memories.create as ReturnType<typeof vi.fn>).mockRejectedValue(e);
  return sdk;
}

// ── eve context stub ──────────────────────────────────────────────────────────

/** Minimal eve `ctx` satisfying `EveSessionLike` — authenticated by default. */
export function makeCtx(
  opts: { sessionId?: string; principalId?: string | null; initiatorId?: string | null } = {},
): EveSessionLike {
  const { sessionId = "sess-eve-1", principalId = "alice", initiatorId = null } = opts;
  return {
    session: {
      id: sessionId,
      auth: {
        current: principalId ? { principalId } : null,
        initiator: initiatorId ? { principalId: initiatorId } : null,
      },
    },
  };
}
