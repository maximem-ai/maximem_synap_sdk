/**
 * Minimal Synap example — TypeScript, no framework.
 *
 * Run after:
 *   npm install @maximem/synap
 *   export SYNAP_INSTANCE_ID=inst_...
 *   export SYNAP_API_KEY=synap_...
 */

import { MaximemSynapSDK } from "@maximem/synap";
import { v5 as uuidv5 } from "uuid";

const NAMESPACE_URL = "6ba7b811-9dad-11d1-80b4-00c04fd430c8";

async function main() {
  const sdk = new MaximemSynapSDK({
    instanceId: process.env.SYNAP_INSTANCE_ID!,
    apiKey: process.env.SYNAP_API_KEY!,
  });
  await sdk.initialize();

  try {
    const userId = "alice";
    const customerId = "acme";
    // conversation_id must be a UUID — derive deterministically from any session string
    const convId = uuidv5("session-2026-05-04", NAMESPACE_URL);

    // 1. Ingest a turn
    const ingest = await sdk.memories.create({
      document:
        "User: I prefer concise bullet-point summaries.\n" +
        "Assistant: Got it — I'll keep responses tight.",
      documentType: "ai-chat-conversation",
      userId,
      customerId,
      mode: "long-range",
    });
    console.log(`ingestion_id=${ingest.ingestionId} status=${ingest.status}`);

    // Real apps don't sleep here.
    await new Promise((r) => setTimeout(r, 3000));

    // 2. Fetch context
    const context = await sdk.conversation.context.fetch({
      conversationId: convId,
      searchQuery: ["communication preferences"],
      maxResults: 5,
      mode: "fast",
    });

    console.log(
      `\nFound ${context.facts.length} facts, ${context.preferences.length} preferences`
    );
    for (const p of context.preferences) {
      console.log(`  preference: ${p.content} (conf=${p.confidence.toFixed(2)})`);
    }
  } finally {
    await sdk.shutdown();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
