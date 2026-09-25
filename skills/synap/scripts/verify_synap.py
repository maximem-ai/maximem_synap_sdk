#!/usr/bin/env python3
"""Maximem Synap smoke test.

Run this as the LAST step of any Synap integration. It proves the SDK can
authenticate, resolve its instance from the API key, open the live gRPC stream,
and shut down cleanly.

    export SYNAP_API_KEY=synap_...
    export SYNAP_INSTANCE_ID=inst_...      # optional; shown with the key in the dashboard
    python scripts/verify_synap.py

Exit code 0 = green. Non-zero = something is wrong; read the printed error
and consult reference/sdk-setup.md (error-handling section).

Optional round-trip check (writes + reads one throwaway memory) — only runs
when you opt in, so the default smoke test never mutates your instance:

    SYNAP_VERIFY_ROUNDTRIP=1 python scripts/verify_synap.py

On a B2B instance (whoami reports user_context_isolation="strict") the round-trip
also needs a customer id, because a user_id on its own is an error there:

    SYNAP_VERIFY_CUSTOMER_ID=acme SYNAP_VERIFY_ROUNDTRIP=1 python scripts/verify_synap.py

Leave it unset on B2C (equals_customer): a customer_id is rejected there with HTTP 400.

The stream check opens `sdk.instance.listen()` and closes it again. It sends
nothing and stores nothing. A failure there is a warning, not an error: a
sandbox with no gRPC egress cannot open a stream and that is not a problem with
the integration. Skip it with SYNAP_VERIFY_SKIP_STREAM=1.
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

        # ⚠ `initialize()` does NOT raise on a bad key. It logs
        # "whoami bootstrap failed (non-fatal, will fall back to env)" and
        # returns, leaving `instance_id` empty. This script used to print that
        # empty value as an [OK] line, so a junk key produced four green ticks
        # and exit 0 while every request 401'd. It is the gate SKILL.md tells
        # you never to skip, and it could not fail.
        if not sdk.instance_id:
            print("[ERROR] Authenticated to nothing: the API key did not resolve "
                  "an instance.")
            print("        Check SYNAP_API_KEY, and SYNAP_BASE_URL if you are "
                  "pointing at anything other than production.")
            raise SystemExit(1)
        print(f"[OK] Connected to instance: {sdk.instance_id}")

        if os.environ.get("SYNAP_VERIFY_ROUNDTRIP"):
            await _roundtrip(sdk)

        if not os.environ.get("SYNAP_VERIFY_SKIP_STREAM"):
            await _stream(sdk)

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

    # B2C sends user_id alone. B2B (user_context_isolation="strict") also needs a
    # customer_id, and a B2C instance rejects one with HTTP 400, so send it only
    # when the caller has said this is a B2B instance.
    scope = {"user_id": user_id}
    customer_id = os.environ.get("SYNAP_VERIFY_CUSTOMER_ID")
    if customer_id:
        scope["customer_id"] = customer_id

    await sdk.memories.create(
        document="User: My favorite color is teal.\nAssistant: Got it.",
        document_type="ai-chat-conversation",
        **scope,
    )
    print("[OK] Ingest accepted (async pipeline; extraction is eventual)")

    # Extraction is asynchronous — give the pipeline a moment. In production,
    # drive retrieval from webhooks instead of sleeping.
    await asyncio.sleep(5)

    ctx = await sdk.user.context.fetch(search_query=["favorite color"], **scope)
    total = len(ctx.facts) + len(ctx.preferences)
    print(f"[OK] Fetched user context ({total} items; may be 0 on a cold pipeline)")


async def _stream(sdk: MaximemSynapSDK) -> None:
    """Open the live stream and close it. Sends nothing, stores nothing.

    Checked because the stream is the operation integrations skip, and because
    when it fails it fails quietly: fetch() still works, it is just cold every
    time, and no turn ever becomes memory on its own.
    """
    try:
        await sdk.instance.listen()
    except Exception as e:  # noqa: BLE001 - a closed network is not a code bug
        print(f"[WARN] Could not open the live stream: {type(e).__name__}: {e}")
        print("       Anticipation needs it. If this host has gRPC egress on 443,")
        print("       fix this before shipping; see reference/streaming.md.")
        return

    # ⚠ `is_listening` is true for a stream the server rejected: it reflects
    # the client's own intent, not the server's answer. Verified against a live
    # instance with an invalid key, where it read True while the stream logged
    # UNAUTHENTICATED. So prove the stream instead of trusting the flag: send
    # one real event and ask the SDK whether it got out.
    if not sdk.instance.is_listening:
        print("[ERROR] listen() returned but the stream is not connected.")
        raise SystemExit(1)

    probe_conversation = str(uuid.uuid4())
    await sdk.instance.send_message(
        conversation_id=probe_conversation,
        user_id="synap-verify-probe",
        event_type="user_message",
        role="user",
        content="synap verify probe",
    )
    await sdk.instance.end_session(probe_conversation)
    await sdk.instance.stop_listening()

    # `undelivered()` is the only honest signal here: every send path returns
    # None, so "it did not raise" says nothing about whether the server got it.
    transport = getattr(sdk, "_grpc_transport", None)
    left = transport.undelivered() if hasattr(transport, "undelivered") else None
    if left and (left["queued"] or left["unacknowledged"]):
        print(f"[ERROR] The stream is open but the server confirmed nothing: "
              f"{left}.")
        print("        Anticipation will receive none of your turns. Check "
              "gRPC egress on 443 and your key's instance.")
        raise SystemExit(1)
    print("[OK] Live stream opened and the server confirmed a test event")


if __name__ == "__main__":
    asyncio.run(verify())

# Accurate as of maximem-synap 0.5.1 — verified 2026-09-25. Docs: https://docs.maximem.ai
