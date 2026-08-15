"""Phase 1 — verify ``SynapBackend`` against a live Synap instance.

Source recon told us what the interfaces are. This tells us how they behave:
latency on the paths that sit inside an agent turn, how big the recall block
actually gets, and whether the ingestion lifecycle is what the backend assumes.

Nothing here is a unit test. It talks to a real instance, writes real memories
under a throwaway scope, and prints a Markdown block to paste into
``integrations/DEEPAGENTS_HARNESS_INTEGRATION_PLAN.md``.

    export SYNAP_API_KEY=synap_...       # required
    export SYNAP_INSTANCE_ID=inst_...    # required unless whoami resolves it
    export SYNAP_BASE_URL=https://...    # optional, defaults to prod

    python integrations/synap-deepagents/bench/live_verify.py

Add ``--keep`` to leave the seeded memories in place, ``--scope NAME`` to reuse
a scope from a previous run instead of seeding a fresh one.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from maximem_synap import MaximemSynapSDK

from synap_deepagents import SynapBackend

# Seed corpus. Deliberately phrased so that no query below is a literal match —
# a regex `grep` would miss every one of them, which is the whole argument for
# routing `grep` at a retrieval service.
SEED_DOCUMENTS: Sequence[str] = (
    "I ship to production by tagging a release and letting the GitHub Actions "
    "pipeline promote the build. I never deploy by hand from a laptop.",
    "My preferred Python formatter is ruff. I dislike black's handling of "
    "magic trailing commas and stopped using it in 2024.",
    "For databases I reach for PostgreSQL first. I only consider a document "
    "store when the access pattern is genuinely document-shaped.",
    "I run integration tests against a disposable container rather than a "
    "shared staging database, because shared state makes failures unreadable.",
    "When reviewing code I care most about error handling on the boundaries. "
    "Internal helpers can be terse; anything that touches the network cannot.",
)

# (label, query) — phrased the way an agent would ask, not the way the seed
# documents are worded.
GREP_QUERIES: Sequence[Tuple[str, str]] = (
    ("deployment", "how does this person release software"),
    ("formatting", "what code formatter should I use here"),
    ("storage", "which database would they pick for this"),
    ("testing", "how do they like integration tests to be run"),
    ("review", "what do they look for in a code review"),
)

REPEATS = 5


def _pct(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


async def _timed(coro_factory) -> Tuple[float, Any]:
    start = time.perf_counter()
    result = await coro_factory()
    return (time.perf_counter() - start) * 1000.0, result


def _body_of(read_result: Any) -> str:
    """Pull the text out of a ``ReadResult``.

    ``FileData`` is a ``TypedDict``, so ``file_data`` is a plain dict at
    runtime and attribute access silently yields nothing.
    """
    file_data = getattr(read_result, "file_data", None)
    if isinstance(file_data, dict):
        return str(file_data.get("content") or "")
    return str(getattr(file_data, "content", "") or "")


async def seed(
    sdk: MaximemSynapSDK, user_id: str, customer_id: Optional[str]
) -> Dict[str, Any]:
    """Ingest the seed corpus and wait for it to land.

    Waiting is fine here and nowhere near an agent turn — the point is to make
    the measurements below reproducible, and to observe the ingestion lifecycle
    the backend has to assume rather than guess at it.
    """
    print(
        f"seeding {len(SEED_DOCUMENTS)} documents under "
        f"user_id={user_id} customer_id={customer_id} ..."
    )
    ingestion_ids: List[uuid.UUID] = []
    for document in SEED_DOCUMENTS:
        response = await sdk.memories.create(
            document=document,
            document_type="document",
            user_id=user_id,
            customer_id=customer_id,
            mode="fast",
        )
        ingestion_ids.append(response.ingestion_id)

    lifecycle: List[Dict[str, Any]] = []
    for ingestion_id in ingestion_ids:
        started = time.perf_counter()
        try:
            await sdk.memories.wait_for_completion(ingestion_id, timeout_seconds=300)
        except Exception as exc:  # noqa: BLE001 — a timeout is a finding, not a crash
            print(f"  ! wait_for_completion({ingestion_id}) failed: {exc}")
        elapsed = (time.perf_counter() - started) * 1000.0
        status = await sdk.memories.status(ingestion_id)
        lifecycle.append(
            {
                "ingestion_id": str(ingestion_id),
                "status": str(status.status),
                "ms": elapsed,
                "memories_created": status.memories_created,
                "memory_ids": list(status.memory_ids),
            }
        )
        print(
            f"  {status.status}: {status.memories_created} memories "
            f"in {elapsed:.0f} ms"
        )

    return {"lifecycle": lifecycle}


async def measure_fetch(
    sdk: MaximemSynapSDK, user_id: str, customer_id: Optional[str]
) -> List[Dict[str, Any]]:
    """Latency and payload size across the fetch matrix the backend exposes.

    ``mode`` and ``precision_level`` are both configurable on ``SynapBackend``
    and both land inside a turn — ``mode`` on every read, ``grep_mode`` on
    every grep. These numbers are what the defaults should be chosen from.
    """
    rows: List[Dict[str, Any]] = []
    for mode in ("fast", "accurate"):
        for precision in ("medium", "high"):
            for query in (None, GREP_QUERIES[0][1]):
                samples: List[float] = []
                sizes: List[int] = []
                for _ in range(REPEATS):
                    elapsed, response = await _timed(
                        lambda: sdk.fetch(
                            user_id=user_id,
                            customer_id=customer_id,
                            search_query=[query] if query else None,
                            max_results=20,
                            mode=mode,
                            precision_level=precision,
                            include_conversation_context=False,
                        )
                    )
                    samples.append(elapsed)
                    sizes.append(
                        len((getattr(response, "formatted_context", None) or ""))
                    )
                rows.append(
                    {
                        "mode": mode,
                        "precision": precision,
                        "query": "yes" if query else "no",
                        "p50_ms": statistics.median(samples),
                        "p95_ms": _pct(samples, 0.95),
                        "chars": int(statistics.median(sizes)),
                    }
                )
                print(
                    f"  fetch mode={mode} precision={precision} "
                    f"query={'yes' if query else 'no'}: "
                    f"p50={statistics.median(samples):.0f} ms, "
                    f"{int(statistics.median(sizes))} chars"
                )
    return rows


async def measure_max_results(
    sdk: MaximemSynapSDK, user_id: str, customer_id: Optional[str]
) -> List[Dict[str, Any]]:
    """Does a grep-shaped query stay useful as ``max_results`` grows?

    The backend defaults to 20. If 10 and 20 return the same characters, the
    corpus is the ceiling and the default is free; if 20 is much larger, the
    default is spending prompt budget the agent may not need.
    """
    rows: List[Dict[str, Any]] = []
    for max_results in (5, 10, 20, 40):
        elapsed, response = await _timed(
            lambda: sdk.fetch(
                user_id=user_id,
                customer_id=customer_id,
                search_query=[GREP_QUERIES[0][1]],
                max_results=max_results,
                mode="accurate",
                precision_level="high",
                include_conversation_context=False,
            )
        )
        text = getattr(response, "formatted_context", None) or ""
        rows.append(
            {"max_results": max_results, "ms": elapsed, "chars": len(text)}
        )
        print(f"  max_results={max_results}: {elapsed:.0f} ms, {len(text)} chars")
    return rows


async def exercise_backend(
    sdk: MaximemSynapSDK, user_id: str, customer_id: Optional[str]
) -> List[Dict[str, Any]]:
    """Drive the real backend through the operations an agent performs.

    Mocks proved the shapes. This proves the semantics: that a grep actually
    finds a differently-worded memory, that a write is readable back inside the
    same turn despite queued ingestion, and that a miss reports ``file_not_found``
    rather than a code that makes ``MemoryMiddleware`` raise.
    """
    backend = SynapBackend(sdk, user_id=user_id, customer_id=customer_id)
    checks: List[Dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str, ms: float = 0.0) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail, "ms": ms})
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")

    elapsed, read_result = await _timed(lambda: backend.aread("/AGENTS.md"))
    body = _body_of(read_result)
    record(
        "read recall file",
        bool(body),
        f"{len(body)} chars, {len(body.splitlines())} lines",
        elapsed,
    )

    for label, query in GREP_QUERIES:
        elapsed, grep_result = await _timed(lambda: backend.agrep(query))
        matches = list(getattr(grep_result, "matches", []) or [])
        # GrepMatch is a TypedDict — subscript it, do not use getattr.
        paths = sorted({(m.get("path") if isinstance(m, dict) else None) or "?" for m in matches})
        record(
            f"grep: {label}",
            bool(matches),
            f"{len(matches)} matches from {paths}",
            elapsed,
        )

    # Read-after-write. This is the one the write-through cache exists for: the
    # memory is still queued, so without the cache the agent would read back
    # nothing it just wrote and reasonably conclude the write was lost.
    marker = f"Read-after-write probe {uuid.uuid4().hex[:8]}."
    elapsed, _ = await _timed(lambda: backend.awrite("/probe.md", marker))
    _, back = await _timed(lambda: backend.aread("/probe.md"))
    back_body = _body_of(back)
    record(
        "read-after-write",
        marker in back_body,
        "cache served the pending write" if marker in back_body else "LOST",
        elapsed,
    )

    # A path we never wrote, phrased as nonsense so retrieval has nothing to
    # return. The only acceptable failure code is file_not_found.
    _, miss = await _timed(
        lambda: backend.adownload_files([f"/{uuid.uuid4().hex}-zzzz.md"])
    )
    # FileDownloadResponse.error is a plain string literal, not an object.
    codes = [getattr(r, "error", None) for r in miss]
    record(
        "miss reports file_not_found",
        all(c in (None, "file_not_found") for c in codes),
        f"codes={codes}",
    )

    return checks


async def cleanup(sdk: MaximemSynapSDK, lifecycle: Sequence[Dict[str, Any]]) -> str:
    """Delete what we seeded, and report what deletion actually does.

    This is the open question behind the missing ``delete`` on the backend:
    ``create`` hands back an ``ingestion_id``, but ``status`` hands back
    ``memory_ids`` once ingestion finishes. If that bridge is reliable, a
    deferred delete becomes possible and the plan's D-decision is worth
    revisiting.
    """
    memory_ids = [mid for row in lifecycle for mid in row.get("memory_ids", [])]
    if not memory_ids:
        return "no memory_ids were returned by status(); the ingestion→memory bridge is unavailable"

    deleted, failed = 0, 0
    for memory_id in memory_ids:
        try:
            await sdk.memories.delete(uuid.UUID(str(memory_id)))
            deleted += 1
        except Exception as exc:  # noqa: BLE001 — a failure is the finding
            failed += 1
            print(f"  ! delete({memory_id}) failed: {exc}")

    surviving = 0
    for memory_id in memory_ids:
        try:
            await sdk.memories.get(uuid.UUID(str(memory_id)))
            surviving += 1
        except Exception:  # noqa: BLE001 — a raise here means it is gone
            pass

    kind = "hard delete" if surviving == 0 else f"tombstone ({surviving} still readable)"
    return f"{deleted} deleted, {failed} failed; get() after delete suggests a {kind}"


def render(
    user_id: str,
    seed_info: Dict[str, Any],
    fetch_rows: Sequence[Dict[str, Any]],
    max_results_rows: Sequence[Dict[str, Any]],
    checks: Sequence[Dict[str, Any]],
    delete_note: str,
) -> str:
    lines: List[str] = []
    lines.append("## Phase 1 — recon RESULTS")
    lines.append("")
    lines.append(f"Scope `user_id={user_id}`, {len(SEED_DOCUMENTS)} seed documents, "
                 f"{REPEATS} samples per fetch cell.")
    lines.append("")

    lines.append("### Ingestion lifecycle")
    lines.append("")
    lines.append("| ingestion | status | wait ms | memories | memory_ids returned |")
    lines.append("|---|---|---:|---:|---|")
    for row in seed_info["lifecycle"]:
        lines.append(
            f"| `{row['ingestion_id'][:8]}` | {row['status']} | {row['ms']:.0f} | "
            f"{row['memories_created']} | {len(row['memory_ids'])} |"
        )
    lines.append("")

    lines.append("### `sdk.fetch` latency and payload")
    lines.append("")
    lines.append("| mode | precision | query | p50 ms | p95 ms | chars |")
    lines.append("|---|---|---|---:|---:|---:|")
    for row in fetch_rows:
        lines.append(
            f"| {row['mode']} | {row['precision']} | {row['query']} | "
            f"{row['p50_ms']:.0f} | {row['p95_ms']:.0f} | {row['chars']} |"
        )
    lines.append("")

    lines.append("### `max_results` sweep (grep-shaped query)")
    lines.append("")
    lines.append("| max_results | ms | chars |")
    lines.append("|---:|---:|---:|")
    for row in max_results_rows:
        lines.append(
            f"| {row['max_results']} | {row['ms']:.0f} | {row['chars']} |"
        )
    lines.append("")

    lines.append("### Backend behaviour")
    lines.append("")
    lines.append("| check | result | detail | ms |")
    lines.append("|---|---|---|---:|")
    for row in checks:
        lines.append(
            f"| {row['check']} | {'pass' if row['ok'] else '**FAIL**'} | "
            f"{row['detail']} | {row['ms']:.0f} |"
        )
    lines.append("")

    lines.append("### Delete semantics")
    lines.append("")
    lines.append(delete_note)
    lines.append("")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", help="reuse an existing user_id instead of seeding")
    parser.add_argument(
        "--customer",
        help="customer_id to scope under; B2B instances require one alongside user_id",
    )
    parser.add_argument("--keep", action="store_true", help="do not delete seeded memories")
    parser.add_argument("--out", help="write the results block to this file")
    args = parser.parse_args()

    api_key = os.environ.get("SYNAP_API_KEY", "").strip()
    if not api_key:
        print("SYNAP_API_KEY is not set. This script needs a live instance.", file=sys.stderr)
        return 2

    sdk = MaximemSynapSDK(
        api_key=api_key,
        instance_id=os.environ.get("SYNAP_INSTANCE_ID") or None,
    )
    await sdk.initialize()

    user_id = args.scope or f"da-verify-{uuid.uuid4().hex[:10]}"
    customer_id = args.customer or f"da-verify-co-{uuid.uuid4().hex[:8]}"

    seed_info: Dict[str, Any] = {"lifecycle": []}
    if not args.scope:
        seed_info = await seed(sdk, user_id, customer_id)

    print("measuring fetch ...")
    fetch_rows = await measure_fetch(sdk, user_id, customer_id)
    print("sweeping max_results ...")
    max_results_rows = await measure_max_results(sdk, user_id, customer_id)
    print("exercising the backend ...")
    checks = await exercise_backend(sdk, user_id, customer_id)

    delete_note = "skipped (--keep)"
    if not args.keep and seed_info["lifecycle"]:
        print("cleaning up ...")
        delete_note = await cleanup(sdk, seed_info["lifecycle"])

    block = render(f"{user_id} / customer_id={customer_id}", seed_info, fetch_rows, max_results_rows, checks, delete_note)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(block)
        print(f"\nwrote {args.out}")
    else:
        print("\n" + block)

    return 0 if all(row["ok"] for row in checks) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
