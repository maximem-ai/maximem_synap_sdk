/**
 * Minimal Synap example — TypeScript, no framework.
 *
 * Run after:
 *   npm install @maximem/synap-js-sdk
 *   export SYNAP_API_KEY=synap_...
 */

import { createClient } from "@maximem/synap-js-sdk";

async function main() {
  const sdk = createClient({
    apiKey: process.env.SYNAP_API_KEY!,   // instance resolved from the key
  });
  await sdk.init();

  try {
    const userId = "alice";
    const customerId = "acme";
    // conversation_id must be a valid UUID
    const convId = crypto.randomUUID();

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
