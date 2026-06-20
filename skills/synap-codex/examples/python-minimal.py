"""
Minimal Synap example — Python, no framework.

Run after:
    pip install maximem-synap
    export SYNAP_INSTANCE_ID=inst_...
    export SYNAP_API_KEY=synap_...
"""

import asyncio
from maximem_synap import MaximemSynapSDK


async def main():
    sdk = MaximemSynapSDK()         # reads env vars
    await sdk.initialize()

    try:
        user_id = "alice"
        customer_id = "acme"

        # 1. Ingest a turn (user-scoped: we pass user_id)
        ingest = await sdk.memories.create(
            document=(
                "User: I prefer concise bullet-point summaries.\n"
                "Assistant: Got it — I'll keep responses tight."
            ),
            document_type="ai-chat-conversation",
            user_id=user_id,
            customer_id=customer_id,
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
