"""Phase 1 — verify the MAF harness mappings against a live Synap instance.

The deepagents run already answered the questions the two plans share: fetch
latency (p50 ~470 ms, flat across every mode and precision), delete semantics
(hard delete), and the small-scope recall floor. What is left is specific to
this plan, and it is the question Tier A's whole design rests on:

    Does a ``MemoryTopicRecord`` survive a round trip through Synap?

MAF's ``MemoryStore`` is a *record* store. ``write_topic(record)`` then
``get_topic(topic)`` must return that record — ``_merge_memory`` does
read-modify-write on it. Synap is not a record store: ``memories.create``
runs an extraction pipeline, and ``Memory`` carries ``content`` with no
metadata field to smuggle a payload through. So the round trip is the thing
to measure, not assume.

Four checks:

1. **Record round trip.** Write ``record.to_dict()`` as a document, wait for
   ingestion, then try to rebuild the record from ``memories.get`` and from
   ``fetch``. Reports what came back rather than pass/fail, because the shape
   of the loss is what decides the design.
2. **Prose round trip.** The same record written as prose, to see whether
   losing the JSON envelope changes what is retrievable.
3. **Index payload.** An unqueried scope ``fetch`` — the exact call
   ``get_index_text`` would make on every turn — against a scope seeded past
   the Finding A floor, to see whether it returns anything worth rendering
   into pointer lines.
4. **Tier B delete chain.** ``create`` -> ``status().memory_ids`` ->
   ``memories.delete`` -> confirm gone. Already verified once; re-run here so
   ``SynapAgentFileStore.delete`` has a result of its own to point at.

Nothing here is a unit test. It talks to a real instance and writes real
memories under a throwaway scope.

    export SYNAP_API_KEY=synap_...
    export SYNAP_INSTANCE_ID=inst_...
    python integrations/synap-microsoft-agent/bench/live_verify.py --customer acme

Add ``--keep`` to leave the seeded memories behind.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from maximem_synap import MaximemSynapSDK

# One realistic topic record, in the shape MAF's extraction pass produces.
TOPIC = "deployment workflow"
SUMMARY = "How this person ships software to production."
MEMORIES: Sequence[str] = (
    "Ships to production by tagging a release; GitHub Actions promotes the build.",
    "Never deploys by hand from a laptop.",
    "Rolls back by re-tagging the previous release rather than reverting commits.",
)

# Background corpus so the scope clears the Finding A floor (~a dozen memories)
# before check 3 measures the unqueried index fetch. Without this, check 3
# measures the floor and tells us nothing about the mapping.
BACKGROUND: Sequence[str] = (
    "Prefers ruff over black for Python formatting, and switched in 2024.",
    "Reaches for PostgreSQL first; only considers a document store when the "
    "access pattern is genuinely document-shaped.",
    "Runs integration tests against a disposable container, never a shared "
    "staging database, because shared state makes failures unreadable.",
    "In code review cares most about error handling at the boundaries; "
    "internal helpers can be terse, anything touching the network cannot.",
    "Keeps secrets in a managed vault and rotates them quarterly.",
    "Writes design docs before large changes and circulates them for a week.",
    "Uses feature flags for anything user-visible so rollout is decoupled "
    "from deploy.",
    "Runs the linter in CI as a blocking check, not an advisory one.",
    "Prefers small pull requests reviewed same-day over large batched ones.",
    "Keeps a runbook per service and treats a missing runbook as a launch "
    "blocker.",
    "Monitors p99 latency rather than averages, because averages hide the "
    "failures users actually notice.",
    "Treats flaky tests as outages and quarantines them the same day.",
)


def _record_dict() -> Dict[str, Any]:
    """The payload ``write_topic`` would submit, without importing MAF.

    Keeping this literal rather than constructing a ``MemoryTopicRecord``
    means the script runs against the shipped SDK alone — it does not need
    the pinned ``agent-framework==1.13.0`` environment.
    """
    return {
        "topic": TOPIC,
        "slug": "deployment-workflow",
        "summary": SUMMARY,
        "memories": list(MEMORIES),
        "updated_at": "2026-08-07T00:00:00+00:00",
        "session_ids": ["sess-live-verify"],
    }


def _as_prose(record: Dict[str, Any]) -> str:
    lines = [f"# {record['topic']}", "", record["summary"], ""]
    lines.extend(f"- {item}" for item in record["memories"])
    return "\n".join(lines)


async def _seed(
    sdk: MaximemSynapSDK,
    documents: Sequence[str],
    *,
    user_id: str,
    customer_id: Optional[str],
    document_type: str = "document",
) -> List[Any]:
    """Submit ``documents`` and wait for every ingestion to complete."""
    responses = []
    for document in documents:
        responses.append(
            await sdk.memories.create(
                document=document,
                user_id=user_id,
                customer_id=customer_id,
                document_type=document_type,
                mode="fast",
            )
        )
    await asyncio.gather(
        *(
            sdk.memories.wait_for_completion(r.ingestion_id, timeout_seconds=300)
            for r in responses
        )
    )
    return responses


async def check_record_round_trip(
    sdk: MaximemSynapSDK, *, user_id: str, customer_id: Optional[str]
) -> Dict[str, Any]:
    """Write a topic record as JSON; report what comes back."""
    record = _record_dict()
    payload = json.dumps(record, ensure_ascii=False)

    created = await sdk.memories.create(
        document=payload,
        user_id=user_id,
        customer_id=customer_id,
        document_type="document",
        mode="fast",
    )
    status = await sdk.memories.wait_for_completion(
        created.ingestion_id, timeout_seconds=300
    )

    contents: List[str] = []
    for memory_id in status.memory_ids:
        memory = await sdk.memories.get(memory_id)
        contents.append(memory.content)

    # Can the JSON be parsed back out of any single returned memory?
    parseable = False
    for content in contents:
        try:
            reparsed = json.loads(content)
        except (ValueError, TypeError):
            continue
        if isinstance(reparsed, dict) and reparsed.get("topic"):
            parseable = True
            break

    joined = "\n".join(contents)
    return {
        "submitted_chars": len(payload),
        "memories_created": len(status.memory_ids),
        "returned_chars": len(joined),
        "json_parseable": parseable,
        "memories_present": sum(1 for m in MEMORIES if _fuzzy_in(m, joined)),
        "memories_total": len(MEMORIES),
        "sample": contents[:3],
    }


async def check_prose_round_trip(
    sdk: MaximemSynapSDK, *, user_id: str, customer_id: Optional[str]
) -> Dict[str, Any]:
    """Write the same record as prose; report what comes back."""
    record = _record_dict()
    payload = _as_prose(record)

    created = await sdk.memories.create(
        document=payload,
        user_id=user_id,
        customer_id=customer_id,
        document_type="document",
        mode="fast",
    )
    status = await sdk.memories.wait_for_completion(
        created.ingestion_id, timeout_seconds=300
    )
    contents = [
        (await sdk.memories.get(memory_id)).content for memory_id in status.memory_ids
    ]
    joined = "\n".join(contents)
    return {
        "submitted_chars": len(payload),
        "memories_created": len(status.memory_ids),
        "returned_chars": len(joined),
        "memories_present": sum(1 for m in MEMORIES if _fuzzy_in(m, joined)),
        "memories_total": len(MEMORIES),
        "sample": contents[:3],
    }


def _fuzzy_in(needle: str, haystack: str) -> bool:
    """Is the *substance* of ``needle`` present in ``haystack``?

    Exact substring matching would report a false negative on any rewording,
    and the pipeline rewords by design. Score on distinctive words instead:
    every word of five characters or more, lowercased. Two thirds present is
    treated as recovered.
    """
    words = {w.lower().strip(".,;:'\"") for w in needle.split() if len(w) >= 5}
    if not words:
        return False
    hay = haystack.lower()
    hits = sum(1 for w in words if w in hay)
    return hits >= (2 * len(words)) / 3


async def check_index_payload(
    sdk: MaximemSynapSDK, *, user_id: str, customer_id: Optional[str]
) -> Dict[str, Any]:
    """The unqueried scope fetch ``get_index_text`` would make each turn."""
    timings: List[float] = []
    text = ""
    for _ in range(3):
        start = time.perf_counter()
        response = await sdk.fetch(
            user_id=user_id,
            customer_id=customer_id,
            search_query=None,
            max_results=20,
            mode="fast",
            precision_level="high",
            include_conversation_context=False,
        )
        timings.append((time.perf_counter() - start) * 1000.0)
        text = (getattr(response, "formatted_context", None) or "").strip()

    queried = await sdk.fetch(
        user_id=user_id,
        customer_id=customer_id,
        search_query=["how does this person deploy to production"],
        max_results=20,
        mode="accurate",
        precision_level="high",
        include_conversation_context=False,
    )
    queried_text = (getattr(queried, "formatted_context", None) or "").strip()

    return {
        "unqueried_chars": len(text),
        "unqueried_lines": len([ln for ln in text.splitlines() if ln.strip()]),
        "unqueried_p50_ms": round(sorted(timings)[len(timings) // 2], 1),
        "queried_chars": len(queried_text),
        "topic_recoverable_unqueried": _fuzzy_in(MEMORIES[0], text),
        "topic_recoverable_queried": _fuzzy_in(MEMORIES[0], queried_text),
        "unqueried_sample": text[:400],
    }


async def check_delete_chain(
    sdk: MaximemSynapSDK, *, user_id: str, customer_id: Optional[str]
) -> Dict[str, Any]:
    """``write`` -> ``status().memory_ids`` -> ``delete`` -> confirm gone.

    This is exactly what ``SynapAgentFileStore.delete`` does. ``delete`` is
    abstract on ``AgentFileStore`` and returns ``bool``, so the chain has to
    hold or the method has to lie.
    """
    created = await sdk.memories.create(
        document=(
            "Temporary note written only so it can be deleted again: the "
            "quarterly planning meeting moved to the first Tuesday."
        ),
        user_id=user_id,
        customer_id=customer_id,
        document_type="document",
        mode="fast",
    )
    status = await sdk.memories.wait_for_completion(
        created.ingestion_id, timeout_seconds=300
    )

    deleted = 0
    for memory_id in status.memory_ids:
        await sdk.memories.delete(memory_id)
        deleted += 1

    gone = 0
    for memory_id in status.memory_ids:
        try:
            await sdk.memories.get(memory_id)
        except Exception:  # noqa: BLE001 — absence is the expected outcome
            gone += 1

    return {
        "memories_created": len(status.memory_ids),
        "deleted": deleted,
        "confirmed_gone": gone,
    }


def _render(results: Dict[str, Any]) -> str:
    out: List[str] = ["", "=" * 72, "MAF harness — Phase 1 live results", "=" * 72, ""]
    for name, payload in results.items():
        out.append(f"## {name}")
        for key, value in payload.items():
            if key.startswith("sample") or key.endswith("_sample"):
                continue
            out.append(f"  {key}: {value}")
        sample = payload.get("sample") or payload.get("unqueried_sample")
        if sample:
            out.append("  sample:")
            items = sample if isinstance(sample, list) else [sample]
            for item in items:
                out.append(f"    | {str(item)[:200]}")
        out.append("")
    return "\n".join(out)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer", default=None, help="customer_id (required on B2B instances)")
    parser.add_argument("--scope", default=None, help="reuse a scope from an earlier run")
    parser.add_argument("--keep", action="store_true", help="leave seeded memories in place")
    args = parser.parse_args()

    api_key = os.environ.get("SYNAP_API_KEY", "").strip()
    if not api_key:
        print("SYNAP_API_KEY is not set", file=sys.stderr)
        return 2

    sdk = MaximemSynapSDK(
        api_key=api_key,
        instance_id=os.environ.get("SYNAP_INSTANCE_ID") or None,
    )
    await sdk.initialize()

    user_id = args.scope or f"maf-verify-{uuid.uuid4().hex[:10]}"
    print(f"scope: user_id={user_id} customer_id={args.customer}")

    results: Dict[str, Any] = {}
    try:
        if not args.scope:
            print("seeding background corpus (clears the Finding A floor)...")
            await _seed(
                sdk, BACKGROUND, user_id=user_id, customer_id=args.customer
            )

        print("1/4 record round trip (JSON)...")
        results["1. record round trip — to_dict() JSON"] = await check_record_round_trip(
            sdk, user_id=user_id, customer_id=args.customer
        )

        print("2/4 record round trip (prose)...")
        results["2. record round trip — prose"] = await check_prose_round_trip(
            sdk, user_id=user_id, customer_id=args.customer
        )

        print("3/4 index payload (unqueried scope fetch)...")
        results["3. index payload — get_index_text's call"] = await check_index_payload(
            sdk, user_id=user_id, customer_id=args.customer
        )

        print("4/4 delete chain (Tier B)...")
        results["4. delete chain — AgentFileStore.delete"] = await check_delete_chain(
            sdk, user_id=user_id, customer_id=args.customer
        )
    finally:
        print(_render(results))
        if not args.keep:
            print(f"(seeded memories left under user_id={user_id}; pass --scope to reuse)")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
