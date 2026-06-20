/**
 * Minimal Synap example — TypeScript, no framework.
 *
 * The JS SDK is a thin wrapper that spawns the Python SDK as a subprocess, so the
 * host needs Python 3.11+ on PATH in addition to Node 18+. It does NOT run on Edge
 * Runtime, Cloudflare Workers, Bun, Deno Deploy, or Node-only Lambda runtimes.
 *
 * Run after:
 *   npm install @maximem/synap-js-sdk
 *   export SYNAP_API_KEY=synap_...
 *
 * The instance is resolved from the API key — no instance id needed.
 */

import { createClient } from "@maximem/synap-js-sdk";
import { v5 as uuidv5 } from "uuid";

const NAMESPACE_URL = "6ba7b811-9dad-11d1-80b4-00c04fd430c8";

async function main() {
  const sdk = createClient({ apiKey: process.env.SYNAP_API_KEY! });
  await sdk.init(); // note: init(), not initialize()

  try {
    const userId = "alice";
    const customerId = "acme"; // required by addMemory; on B2C, pass the same value as userId
    // conversation_id must be a UUID — derive deterministically from any session string
    const convId = uuidv5("session-2026-05-04", NAMESPACE_URL);

    // 1. Ingest a turn. The JS write path takes a `messages` array, not a `document` string.
    await sdk.addMemory({
      userId,
      customerId,
      conversationId: convId,
      messages: [
        { role: "user", content: "I prefer concise bullet-point summaries." },
        { role: "assistant", content: "Got it — I'll keep responses tight." },
      ],
      mode: "long-range",
    });
    console.log("ingest accepted (async pipeline; extraction is eventual)");

    // Real apps don't sleep here — drive retrieval from webhooks.
    await new Promise((r) => setTimeout(r, 3000));

    // 2. Fetch context. We ingested at USER scope, so we read at user scope to match.
    const context = await sdk.fetchUserContext({
      userId,
      searchQuery: ["communication preferences"],
      maxResults: 5,
      mode: "fast",
    });

    console.log(
      `\nFound ${context.facts.length} facts, ${context.preferences.length} preferences`
    );
    for (const p of context.preferences) {
      // JS normalizes preference relevance to `strength` (facts use `confidence`).
      console.log(`  preference: ${p.content} (strength=${p.strength})`);
    }
  } finally {
    await sdk.shutdown();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

// ---------------------------------------------------------------------------
// Namespaced API (mirrors the Python SDK 1:1), available from
// @maximem/synap-js-sdk 0.3.0. The flat methods used above (addMemory,
// fetchUserContext) still work; the namespaced surface is added alongside them
// and accepts camelCase OR snake_case argument keys:
//
//   await sdk.conversation.record_message({ conversationId, role, content, userId, customerId });
//   await sdk.memories.create({ document, userId, customerId });
//   const ctx = await sdk.user.context.fetch({ userId, customerId, searchQuery: ["..."] });
//   const promptCtx = await sdk.conversation.context.get_context_for_prompt({ conversationId });
//
// Also: sdk.fetch(...), sdk.customer.context.fetch(...), sdk.client.context.fetch(...).
// Namespaced calls return the raw (snake_case) response shape that the framework
// integrations (@maximem/synap-mastra, @maximem/synap-claude-agent) consume.
// ---------------------------------------------------------------------------
//
// Accurate as of @maximem/synap-js-sdk 0.3.0 — verified 2026-06-20. Docs: https://docs.maximem.ai
