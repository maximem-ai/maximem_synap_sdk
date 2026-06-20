#!/usr/bin/env python3
"""Maximem Synap smoke test.

Run this as the LAST step of any Synap integration. It proves the SDK can
authenticate, resolve its instance from the API key, and shut down cleanly.

    export SYNAP_API_KEY=synap_...
    python scripts/verify_synap.py

Exit code 0 = green. Non-zero = something is wrong; read the printed error
and consult reference/sdk-setup.md (error-handling section).

Optional round-trip check (writes + reads one throwaway memory) — only runs
when you opt in, so the default smoke test never mutates your instance:

    SYNAP_VERIFY_ROUNDTRIP=1 python scripts/verify_synap.py
"""

import asyncio
import os
import uuid

from maximem_synap import MaximemSynapSDK


async def verify() -> None:
    if not os.environ.get("SYNAP_API_KEY"):
        print("[ERROR] SYNAP_API_KEY is not set. Run: export SYNAP_API_KEY=synap_...")
        raise SystemExit(1)

    # No instance_id needed — it is resolved from the API key on initialize().
    sdk = MaximemSynapSDK()
    try:
        await sdk.initialize()
        print("[OK] SDK initialized")
        print(f"[OK] Connected to instance: {sdk.instance_id}")

        if os.environ.get("SYNAP_VERIFY_ROUNDTRIP"):
            await _roundtrip(sdk)

        await sdk.shutdown()
        print("[OK] SDK shut down cleanly")
    except Exception as e:  # noqa: BLE001 - surface any failure to the developer
        print(f"[ERROR] {type(e).__name__}: {e}")
        # Best-effort cleanup; ignore secondary errors during teardown.
        try:
            await sdk.shutdown()
        except Exception:
            pass
        raise SystemExit(1)


async def _roundtrip(sdk: MaximemSynapSDK) -> None:
    """Ingest a known fact, then fetch it back at user scope."""
    user_id = f"verify-{uuid.uuid4().hex[:8]}"
    await sdk.memories.create(
        document="User: My favorite color is teal.\nAssistant: Got it.",
        document_type="ai-chat-conversation",
        user_id=user_id,
    )
    print("[OK] Ingest accepted (async pipeline; extraction is eventual)")

    # Extraction is asynchronous — give the pipeline a moment. In production,
    # drive retrieval from webhooks instead of sleeping.
    await asyncio.sleep(5)

    ctx = await sdk.user.context.fetch(user_id=user_id, search_query=["favorite color"])
    total = len(ctx.facts) + len(ctx.preferences)
    print(f"[OK] Fetched user context ({total} items; may be 0 on a cold pipeline)")


if __name__ == "__main__":
    asyncio.run(verify())

# Accurate as of maximem-synap 0.2.6 — verified 2026-06-17. Docs: https://docs.maximem.ai
