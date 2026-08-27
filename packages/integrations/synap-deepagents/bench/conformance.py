"""Phase 5 — run the same agent task set across three deepagents memory backends.

deepagents ships no memory benchmark, so there is no scoreboard to enter. This
is the honest minimum substitute: one agent, one model, one seed, four graded
fixtures, three backends behind the same ``/memories/`` mount.

    FilesystemBackend  — an AGENTS.md on disk, pasted whole
    StoreBackend       — LangGraph BaseStore, key lookup, pasted whole
    SynapBackend       — retrieval, and `grep` is a question

What it reports per cell: pass/fail on the fixture's assertions, wall-clock, and
the size of the memory block that reached the prompt. That last column is the
one that matters — it is where a whole-file backend runs out of room.

    export SYNAP_API_KEY=synap_...
    export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY, see --model
    python integrations/synap-deepagents/bench/conformance.py

    --arms filesystem,store             # skip the Synap arm (no key needed)
    --fixtures supersession,large-scope
    --model openai:gpt-5

This costs real model tokens. The large-scope fixture alone pastes roughly 200
memories into the prompt on two of the three arms, which is the point of it.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from deepagents import create_deep_agent
from deepagents.backends import (
    BackendProtocol,
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
    StoreBackend,
)
from langgraph.store.memory import InMemoryStore

from bench.fixtures import FIXTURES, Fixture

MEMORY_MOUNT = "/memories/"
MEMORY_FILE = "/memories/AGENTS.md"
DEFAULT_MODEL = "anthropic:claude-sonnet-5"

# Seeding knobs for the Synap arm only. Ingestion is queued and per-document
# completion ranges from ~0.5 s to ~8 s, so a 200-document fixture needs both.
BATCH_SIZE = 25
WAIT_CONCURRENCY = 25

SYSTEM_PROMPT = (
    "You answer from memory. Consult /memories/AGENTS.md, and use grep against "
    "/memories/ when you need something the file does not obviously contain. "
    "On this mount grep is a relevance search over remembered facts, not a "
    "regex over bytes — ask it a plain-language question. Answer in one short "
    "sentence. If memory contradicts itself, prefer the most recent fact."
)


def _as_markdown(memories: Sequence[str]) -> str:
    """Render a corpus the way a hand-maintained AGENTS.md would look."""
    lines = ["# Memories", ""]
    lines.extend(f"- {memory}" for memory in memories)
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Arms
# ----------------------------------------------------------------------


class Arm:
    """One backend under test, plus whatever seeding it needs."""

    name: str

    async def setup(self, fixture: Fixture) -> Dict[str, Any]:
        """Seed the corpus and return kwargs for ``create_deep_agent``."""
        raise NotImplementedError

    async def teardown(self) -> None:
        return None


class FilesystemArm(Arm):
    name = "filesystem"

    def __init__(self) -> None:
        self._tmp: Optional[tempfile.TemporaryDirectory[str]] = None

    async def setup(self, fixture: Fixture) -> Dict[str, Any]:
        self._tmp = tempfile.TemporaryDirectory(prefix="da-bench-")
        root = Path(self._tmp.name)
        (root / "AGENTS.md").write_text(_as_markdown(fixture.memories), encoding="utf-8")
        # virtual_mode=True is load-bearing. CompositeBackend strips the route
        # prefix, so the backend is asked for "/AGENTS.md" — which in real-path
        # mode means the root of the host disk, not our temp directory.
        backend = CompositeBackend(
            default=StateBackend(),
            routes={MEMORY_MOUNT: FilesystemBackend(root_dir=str(root), virtual_mode=True)},
        )
        return {"backend": backend}

    async def teardown(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None


class StoreArm(Arm):
    name = "store"

    async def setup(self, fixture: Fixture) -> Dict[str, Any]:
        store = InMemoryStore()
        namespace = ("bench", fixture.name)
        store.put(
            namespace,
            "/AGENTS.md",
            {"content": _as_markdown(fixture.memories), "encoding": "utf-8"},
        )
        # The explicit store matters: without one, StoreBackend resolves the
        # store from the LangGraph runtime and raises outside a graph, which
        # would make --dry-run impossible to run.
        backend = CompositeBackend(
            default=StateBackend(),
            routes={
                MEMORY_MOUNT: StoreBackend(namespace=lambda _rt: namespace, store=store)
            },
        )
        return {"backend": backend, "store": store}


class SynapArm(Arm):
    name = "synap"

    def __init__(self, sdk: Any, customer_id: str) -> None:
        self.sdk = sdk
        # B2B instances reject a user_id with no customer_id, so the bench
        # always supplies both. On a B2C instance the opposite holds: a
        # customer_id is rejected with HTTP 400, so this bench cannot run
        # against one without sending user_id alone.
        self.customer_id = customer_id
        self._user_id: Optional[str] = None

    async def setup(self, fixture: Fixture) -> Dict[str, Any]:
        from synap_deepagents import SynapBackend

        # A fresh scope per fixture, so one fixture's corpus can never answer
        # another's question. This is also why the Synap arm is slow to set up:
        # ingestion is queued and we wait for it here, off the measured path.
        self._user_id = f"da-bench-{fixture.name}-{uuid.uuid4().hex[:8]}"
        print(f"    seeding {len(fixture.memories)} memories ...", flush=True)

        # batch_create + concurrent waits. The large-scope fixture is ~200
        # documents; one at a time, with a serial wait each, that is twenty
        # minutes of setup for a one-turn measurement.
        from maximem_synap.memories.models import CreateMemoryRequest

        requests = [
            CreateMemoryRequest(
                document=memory,
                document_type="document",
                user_id=self._user_id,
                customer_id=self.customer_id,
                mode="fast",
            )
            for memory in fixture.memories
        ]
        ingestion_ids = []
        for start in range(0, len(requests), BATCH_SIZE):
            batch = await self.sdk.memories.batch_create(requests[start : start + BATCH_SIZE])
            ingestion_ids.extend(r.ingestion_id for r in batch.results)

        async def _wait(ingestion_id: Any) -> None:
            try:
                await self.sdk.memories.wait_for_completion(
                    ingestion_id, timeout_seconds=600
                )
            except Exception as exc:  # noqa: BLE001 — report, do not abort the run
                print(f"    ! ingestion {ingestion_id} did not complete: {exc}")

        for start in range(0, len(ingestion_ids), WAIT_CONCURRENCY):
            await asyncio.gather(
                *(_wait(i) for i in ingestion_ids[start : start + WAIT_CONCURRENCY])
            )
        print(f"    seeded ({len(ingestion_ids)} ingestions complete)", flush=True)

        backend = CompositeBackend(
            default=StateBackend(),
            routes={
                MEMORY_MOUNT: SynapBackend(
                    self.sdk, user_id=self._user_id, customer_id=self.customer_id
                )
            },
        )
        return {"backend": backend}


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


def _body_of(read_result: Any) -> str:
    """Pull the text out of a ``ReadResult``.

    ``FileData`` is a ``TypedDict``, so ``file_data`` is a plain dict at
    runtime and attribute access silently yields nothing.
    """
    file_data = getattr(read_result, "file_data", None)
    if isinstance(file_data, dict):
        return str(file_data.get("content") or "")
    return str(getattr(file_data, "content", "") or "")


def _final_text(result: Any) -> str:
    messages = (result or {}).get("messages") or []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            joined = " ".join(p for p in parts if p).strip()
            if joined:
                return joined
    return ""


async def _memory_block_chars(backend: BackendProtocol) -> int:
    """How many characters of memory reach the prompt.

    ``MemoryMiddleware`` loads the memory sources through the backend and parks
    them in state under ``memory_contents`` — but that key is a
    ``PrivateStateAttr``, so it never appears in the returned state and reading
    it back always yields zero. Asking the backend for the same file the
    middleware asks for gives the identical bytes, one call earlier.
    """
    result = await backend.aread(MEMORY_FILE)
    return len(_body_of(result))


def _grade(fixture: Fixture, answer: str) -> tuple[bool, str]:
    lowered = answer.lower()
    missing = [s for s in fixture.expect_contains if s.lower() not in lowered]
    present = [s for s in fixture.expect_absent if s.lower() in lowered]
    if missing:
        return False, f"missing {missing}"
    if present:
        return False, f"leaked {present}"
    return True, "ok"


async def run_cell(arm: Arm, fixture: Fixture, model: str) -> Dict[str, Any]:
    memory_chars = 0
    try:
        # setup() belongs inside the try. It seeds up to 200 documents over the
        # network, and a transient failure there should cost one cell, not the
        # whole run — a DNS blip during large-scope seeding took out the first
        # full run and lost three fixtures that had already passed.
        kwargs = await arm.setup(fixture)
        memory_chars = await _memory_block_chars(kwargs["backend"])
        agent = create_deep_agent(
            model=model,
            memory=[MEMORY_FILE],
            system_prompt=SYSTEM_PROMPT,
            **kwargs,
        )
        started = time.perf_counter()
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": fixture.question}]}
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        answer = _final_text(result)
        passed, note = _grade(fixture, answer)
        return {
            "arm": arm.name,
            "fixture": fixture.name,
            "pass": passed,
            "note": note,
            "ms": elapsed,
            "memory_chars": memory_chars,
            "answer": answer[:200],
        }
    except Exception as exc:  # noqa: BLE001 — one broken cell must not end the run
        return {
            "arm": arm.name,
            "fixture": fixture.name,
            "pass": False,
            "note": f"error: {type(exc).__name__}: {exc}",
            "ms": 0.0,
            "memory_chars": 0,
            "answer": "",
        }
    finally:
        await arm.teardown()


def render(rows: Sequence[Dict[str, Any]], model: str, skipped: Sequence[str]) -> str:
    arms = sorted({row["arm"] for row in rows})
    fixtures = [f.name for f in FIXTURES if any(r["fixture"] == f.name for r in rows)]
    index = {(row["arm"], row["fixture"]): row for row in rows}

    lines: List[str] = []
    lines.append("## Phase 5 — conformance RESULTS")
    lines.append("")
    lines.append(f"Model `{model}`, one turn per cell, deterministic grading.")
    lines.append("")

    lines.append("### Pass / fail")
    lines.append("")
    lines.append("| fixture | " + " | ".join(arms) + " |")
    lines.append("|---" * (len(arms) + 1) + "|")
    for fixture in fixtures:
        cells = []
        for arm in arms:
            row = index.get((arm, fixture))
            cells.append("—" if row is None else ("pass" if row["pass"] else "**fail**"))
        lines.append(f"| {fixture} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### Latency (ms) and memory block size (chars)")
    lines.append("")
    lines.append("| fixture | " + " | ".join(f"{a} ms / chars" for a in arms) + " |")
    lines.append("|---" * (len(arms) + 1) + "|")
    for fixture in fixtures:
        cells = []
        for arm in arms:
            row = index.get((arm, fixture))
            cells.append("—" if row is None else f"{row['ms']:.0f} / {row['memory_chars']}")
        lines.append(f"| {fixture} | " + " | ".join(cells) + " |")
    lines.append("")

    failures = [row for row in rows if not row["pass"]]
    if failures:
        lines.append("### Failures")
        lines.append("")
        for row in failures:
            lines.append(f"- `{row['arm']}` / `{row['fixture']}` — {row['note']}")
            if row["answer"]:
                lines.append(f"  > {row['answer']}")
        lines.append("")

    if skipped:
        lines.append(f"**Not run:** {', '.join(skipped)}. Coverage below is partial.")
        lines.append("")

    return "\n".join(lines)


async def dry_run(arms: Sequence[Arm], fixtures: Sequence[Fixture]) -> int:
    """Prove the harness before spending model tokens on it.

    Reads ``/memories/AGENTS.md`` through each arm's mounted backend and checks
    the corpus came back. A benchmark that silently serves an empty memory file
    on one arm would report that arm as simply worse, which is the most
    expensive kind of wrong.
    """
    ok = True
    for fixture in fixtures:
        print(f"\n{fixture.name}")
        for arm in arms:
            try:
                kwargs = await arm.setup(fixture)
                backend: BackendProtocol = kwargs["backend"]
                result = await backend.aread(MEMORY_FILE)
                body = _body_of(result)
                first = fixture.memories[0][:40].lower()
                served = first in body.lower()
                print(
                    f"  {arm.name}: {len(body)} chars, "
                    f"first memory {'present' if served else 'MISSING'}"
                )
                ok = ok and served
            except Exception as exc:  # noqa: BLE001 — a broken arm is the finding
                print(f"  {arm.name}: ERROR {type(exc).__name__}: {exc}")
                ok = False
            finally:
                await arm.teardown()
    print("\ndry run:", "ok" if ok else "FAILED")
    return 0 if ok else 1


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--arms", default="filesystem,store,synap")
    parser.add_argument("--fixtures", default="")
    parser.add_argument(
        "--customer",
        help="customer_id for the Synap arm; B2B instances require one alongside user_id",
    )
    parser.add_argument("--out", help="write the results block to this file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check the arms wire up and serve the corpus, without calling a model",
    )
    args = parser.parse_args()

    wanted_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    fixtures = list(FIXTURES)
    if args.fixtures:
        wanted = {f.strip() for f in args.fixtures.split(",") if f.strip()}
        fixtures = [f for f in fixtures if f.name in wanted]
    if not fixtures:
        print("no fixtures selected", file=sys.stderr)
        return 2

    arms: List[Arm] = []
    skipped: List[str] = []
    if "filesystem" in wanted_arms:
        arms.append(FilesystemArm())
    if "store" in wanted_arms:
        arms.append(StoreArm())
    if "synap" in wanted_arms:
        api_key = os.environ.get("SYNAP_API_KEY", "").strip()
        if not api_key:
            skipped.append("the synap arm (SYNAP_API_KEY is not set)")
        else:
            from maximem_synap import MaximemSynapSDK

            sdk = MaximemSynapSDK(
                api_key=api_key,
                instance_id=os.environ.get("SYNAP_INSTANCE_ID") or None,
            )
            await sdk.initialize()
            arms.append(SynapArm(sdk, args.customer or f"da-bench-co-{uuid.uuid4().hex[:8]}"))

    if args.dry_run:
        return await dry_run(arms, fixtures)

    rows: List[Dict[str, Any]] = []
    for fixture in fixtures:
        print(f"\n{fixture.name}")
        for arm in arms:
            print(f"  {arm.name} ...", flush=True)
            row = await run_cell(arm, fixture, args.model)
            rows.append(row)
            print(
                f"    {'pass' if row['pass'] else 'FAIL'} "
                f"({row['note']}) {row['ms']:.0f} ms, "
                f"{row['memory_chars']} memory chars"
            )

    block = render(rows, args.model, skipped)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(block)
        print(f"\nwrote {args.out}")
    else:
        print("\n" + block)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
