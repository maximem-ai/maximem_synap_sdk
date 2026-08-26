"""
Minimal Synap example — Python, no framework.

Run after:
    pip install maximem-synap
    export SYNAP_API_KEY=synap_...
    export SYNAP_INSTANCE_ID=inst_...      # optional; shown with the key in the dashboard

The instance is resolved from the API key on initialize(); SYNAP_INSTANCE_ID is
only needed to pin a specific instance.
"""

import asyncio
from maximem_synap import MaximemSynapSDK


async def main():
    sdk = MaximemSynapSDK()         # reads SYNAP_API_KEY from the environment
    await sdk.initialize()

    try:
        user_id = "alice"

        # 1. Ingest a turn (user-scoped: we pass user_id).
        # This is the B2C shape: user_id and nothing else. On a B2B instance
        # (whoami reports user_context_isolation="strict") add customer_id="acme"
        # to both calls below. On B2C a customer_id is rejected with HTTP 400, and
        # the SDK raises client-side from 0.4.7.
        ingest = await sdk.memories.create(
            document=(
                "User: I prefer concise bullet-point summaries.\n"
                "Assistant: Got it — I'll keep responses tight."
            ),
            document_type="ai-chat-conversation",
            user_id=user_id,
            mode="long-range",
        )
        print(f"ingestion_id={ingest.ingestion_id}  status={ingest.status}")

        # Real apps don't sleep here. Use webhooks or fire-and-forget.
        # We sleep for the demo so retrieval has something to find.
        await asyncio.sleep(3)

        # 2. Fetch context for the next turn.
        # We wrote at USER scope, so we read at user scope (interface matches ingestion).
        context = await sdk.user.context.fetch(
            user_id=user_id,
            search_query=["communication preferences"],
            max_results=5,
            mode="fast",
        )

        print(f"\nFound {len(context.facts)} facts, "
              f"{len(context.preferences)} preferences")
        for p in context.preferences:
            # Preference relevance is `strength`; Fact relevance is `confidence`.
            print(f"  preference: {p.content} (strength={p.strength:.2f})")

        # 3. Build a system prompt with the memory
        memory_block = "\n".join(
            f"- {p.content}" for p in context.preferences
        ) or "No prior preferences known."

        system_prompt = (
            "You are a helpful assistant. Use this context about the user, "
            "but do not mention you are reading from a memory system.\n\n"
            f"## Preferences\n{memory_block}"
        )
        print(f"\n--- system prompt ---\n{system_prompt}")

    finally:
        await sdk.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

# Accurate as of maximem-synap 0.2.6 — verified 2026-06-17. Docs: https://docs.maximem.ai
