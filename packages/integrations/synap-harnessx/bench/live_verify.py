"""Verify ``SynapExtension`` against a live Synap instance.

The unit tests prove the wiring against mocks. This proves the behaviour: that
a memory worded differently from the question is recalled into the system
prompt, that the tools work, that a turn is recorded, and that nothing Synap
rejects is being silently swallowed.

That last one is the reason this script exists. The extension never raises into
the agent run, by design, so a server that rejects every call (a wrong scope on
a B2C instance, a malformed conversation id) looks exactly like a server that
works. Here every ERROR the extension logs, and every stream report that fails,
is captured and turned into a failing check.

No model key is needed. The agent runs on a scripted provider, so what is under
test is the extension and Synap, not a model.

    export SYNAP_API_KEY=synap_...       # required
    export SYNAP_INSTANCE_ID=inst_...    # required unless whoami resolves it

    python packages/integrations/synap-harnessx/bench/live_verify.py

``--keep`` leaves the seeded memories in place. ``--scope NAME`` reuses a
previous run's user_id instead of seeding again. ``--customer ID`` sets the
customer id on a B2B instance; the default is a throwaway one, and on a B2C
instance none is sent. ``--dry-run`` runs the same script against an in-memory
fake, to check the script itself without credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import statistics
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnessx import Agent, AgentConfig, PermissionLevel, ProviderResponse, ToolCall
from harnessx.providers import LLMProvider

from synap_harnessx import SynapExtension

# Seed corpus. Each query below is worded differently from the document it
# should find, so a keyword match would miss every one of them.
SEED_DOCUMENTS: Sequence[str] = (
    "I ship to production by tagging a release and letting the GitHub Actions "
    "pipeline promote the build. I never deploy by hand from a laptop.",
    "My preferred Python formatter is ruff. I dislike black's handling of "
    "magic trailing commas and stopped using it in 2024.",
    "For databases I reach for PostgreSQL first. I only consider a document "
    "store when the access pattern is genuinely document-shaped.",
)

# (label, query, substring that must appear in the recalled block)
RECALL_QUERIES: Sequence[Tuple[str, str, str]] = (
    ("deployment", "how does this person release software", "GitHub Actions"),
    ("formatting", "what code formatter should I use here", "ruff"),
    ("storage", "which database would they pick for this", "PostgreSQL"),
)

REPEATS = 5


class ScriptedProvider(LLMProvider):
    """Replays fixed responses and keeps every system prompt it was sent."""

    def __init__(self, responses: Sequence[ProviderResponse]) -> None:
        self._responses = iter(responses)
        self.calls: List[Dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> ProviderResponse:
        self.calls.append(kwargs)
        return next(self._responses)

    async def count_tokens(self, **kwargs: Any) -> int:
        return 0

    def system(self, index: int = 0) -> str:
        system = self.calls[index]["system"]
        return system if isinstance(system, str) else str(system)


class LogCapture(logging.Handler):
    """Collects what the extension would otherwise swallow.

    ERROR from ``synap_harnessx`` is a swallowed recall or record failure. DEBUG
    from the stream-events module is a swallowed stream report. Neither reaches
    the caller, so neither can be seen without this.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.swallowed: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if record.name.startswith("synap_harnessx") and record.levelno >= logging.ERROR:
            self.swallowed.append(f"{record.name}: {message}")
        elif "synap stream report failed" in message:
            self.swallowed.append(f"{record.name}: {message}")

    def drain(self) -> List[str]:
        found, self.swallowed = self.swallowed, []
        return found


def _pct(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def make_agent(sdk: Any, user_id: str, customer_id: Optional[str], *responses: ProviderResponse, **kwargs: Any):
    provider = ScriptedProvider(responses or [ProviderResponse(text="noted")])
    ext = SynapExtension(
        sdk=sdk, user_id=user_id, customer_id=customer_id,
        store_permission=PermissionLevel.ALLOW, **kwargs,
    )
    agent = Agent(
        config=AgentConfig(model="claude-sonnet-4-6", system_prompt="BASE"),
        provider=provider,
        extensions=[ext],
    )
    return agent, provider, ext


async def seed(sdk: Any, user_id: str, customer_id: Optional[str]) -> List[Dict[str, Any]]:
    """Ingest the seed corpus and wait for it. Waiting is fine here, and
    nowhere near an agent turn: it makes the measurements reproducible."""
    print(f"seeding {len(SEED_DOCUMENTS)} documents under user_id={user_id} customer_id={customer_id} ...")
    ingestion_ids = []
    for document in SEED_DOCUMENTS:
        response = await sdk.memories.create(
            document=document, document_type="document",
            user_id=user_id, customer_id=customer_id, mode="fast",
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
        lifecycle.append({
            "ingestion_id": str(ingestion_id), "status": str(status.status), "ms": elapsed,
            "memories_created": status.memories_created, "memory_ids": list(status.memory_ids),
        })
        print(f"  {status.status}: {status.memories_created} memories in {elapsed:.0f} ms")
    return lifecycle


async def run_checks(
    sdk: Any, user_id: str, customer_id: Optional[str], capture: LogCapture
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Returns ``(checks, latency_rows, stored_ingestion_ids)``."""
    checks: List[Dict[str, Any]] = []
    stored: List[str] = []

    def record(name: str, ok: bool, detail: str, ms: float = 0.0) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail, "ms": ms})
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")

    def clean(name: str) -> None:
        """Fail when the extension swallowed an error since the last call."""
        swallowed = capture.drain()
        record(f"{name}: nothing swallowed", not swallowed,
               "no errors logged" if not swallowed else "; ".join(swallowed)[:300])

    # Recall: a question worded unlike the memory finds it, and the block lands
    # in the system prompt the model is sent.
    for label, query, expected in RECALL_QUERIES:
        agent, provider, _ = make_agent(sdk, user_id, customer_id)
        started = time.perf_counter()
        await agent.run(query)
        ms = (time.perf_counter() - started) * 1000.0
        system = provider.system()
        record(f"recall: {label}", "<synap_memory>" in system and expected in system,
               f"{expected!r} {'in' if expected in system else 'NOT in'} the system prompt "
               f"({len(system)} chars)", ms)
        clean(f"recall: {label}")

    # Latency of the one call that sits inside a turn.
    agent, _, ext = make_agent(sdk, user_id, customer_id)
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        await ext._fetch(RECALL_QUERIES[0][1])
        samples.append((time.perf_counter() - started) * 1000.0)
    latency = [{"what": "recall fetch (fast, high)", "p50": statistics.median(samples),
                "p95": _pct(samples, 0.95)}]
    clean("recall latency")

    # The scope boundary: someone else must not see this user's memories.
    stranger = f"{user_id}-stranger"
    agent, provider, _ = make_agent(sdk, user_id=stranger, customer_id=customer_id)
    await agent.run(RECALL_QUERIES[0][1])
    leaked = RECALL_QUERIES[0][2] in provider.system()
    record("scope: another user sees nothing", not leaked,
           "no leak" if not leaked else "LEAKED across users")
    clean("scope")

    # Tool: search through the real engine, as the model would call it.
    call = ToolCall("search-1", "search_memory", {"query": RECALL_QUERIES[1][1]})
    agent, _, _ = make_agent(
        sdk, user_id, customer_id,
        ProviderResponse(tool_calls=[call], stop_reason="tool_use"), ProviderResponse(text="done"),
    )
    result = await agent.tools.execute(call)
    record("tool: search_memory", not result.is_error and RECALL_QUERIES[1][2] in result.content,
           f"{len(result.content)} chars, error={result.is_error}")

    # Tool: store, wait for ingestion, then recall it. Written under a unique
    # marker so a previous run's leftovers cannot satisfy it.
    marker = f"zebra{uuid.uuid4().hex[:6]}"
    agent, _, _ = make_agent(sdk, user_id, customer_id)
    # Two sentences, on purpose: Synap's extraction creates no memories from a
    # one-sentence document, though the ingestion still reports COMPLETED.
    call = ToolCall("store-1", "store_memory",
                    {"content": f"My favourite editor theme is called {marker} and I use it everywhere. "
                                "I switched to it last spring because the contrast is easier on my eyes."})
    stored_result = await agent.tools.execute(call)
    ingestion_id = stored_result.content.rsplit("ingestion_id: ", 1)[-1].rstrip(")")
    record("tool: store_memory accepted", not stored_result.is_error and "ingestion_id" in stored_result.content,
           stored_result.content[:120])
    if not stored_result.is_error:
        try:
            await sdk.memories.wait_for_completion(uuid.UUID(ingestion_id), timeout_seconds=300)
            stored.append(ingestion_id)
            created = (await sdk.memories.status(uuid.UUID(ingestion_id))).memories_created
            record("tool: store_memory creates memories", created > 0, f"{created} memories created")
        except Exception as exc:  # noqa: BLE001
            record("tool: store_memory ingests", False, f"wait_for_completion failed: {exc}")
        else:
            agent, provider, _ = make_agent(sdk, user_id, customer_id)
            await agent.run("what editor theme do I use")
            record("tool: stored memory is recalled", marker in provider.system(),
                   f"{marker} {'found' if marker in provider.system() else 'NOT found'} in the prompt")
    clean("tools")

    # Recording, REST path: no stream is open yet.
    agent, _, ext = make_agent(sdk, user_id, customer_id)
    record("conversation id is a UUID", _is_uuid(ext._conversation_id), str(ext._conversation_id))
    await agent.run("remember that I am testing the harnessx integration")
    clean("record over REST")

    # Recording and reporting, stream path.
    try:
        await sdk.instance.listen()
        live = bool(sdk.instance.is_listening)
    except Exception as exc:  # noqa: BLE001 — no stream is a finding, not a crash
        live = False
        record("stream: listen()", False, f"{type(exc).__name__}: {exc}")
    if live:
        try:
            call = ToolCall("stream-1", "search_memory", {"query": RECALL_QUERIES[2][1]})
            agent, _, ext = make_agent(
                sdk, user_id, customer_id,
                ProviderResponse(tool_calls=[call], stop_reason="tool_use"), ProviderResponse(text="PostgreSQL."),
            )
            await agent.run("which database should I use")
            await ext.teardown()
            clean("stream (turn, tool call, tool result, reply, end_session)")
        finally:
            await sdk.instance.stop_listening()

    return checks, latency, stored


def _is_uuid(value: Any) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False


async def cleanup(sdk: Any, lifecycle: Sequence[Dict[str, Any]], stored: Sequence[str]) -> str:
    """Delete what was seeded and stored."""
    memory_ids = [m for row in lifecycle for m in row.get("memory_ids", [])]
    for ingestion_id in stored:
        try:
            status = await sdk.memories.status(uuid.UUID(ingestion_id))
            memory_ids.extend(status.memory_ids)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! status({ingestion_id}) failed: {exc}")
    if not memory_ids:
        return "no memory_ids were returned by status(); nothing could be deleted"
    deleted = failed = 0
    for memory_id in memory_ids:
        try:
            await sdk.memories.delete(uuid.UUID(str(memory_id)))
            deleted += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ! delete({memory_id}) failed: {exc}")
    return f"{deleted} deleted, {failed} failed"


def render(user_id: str, customer_id: Optional[str], lifecycle: Sequence[Dict[str, Any]],
           checks: Sequence[Dict[str, Any]], latency: Sequence[Dict[str, Any]], delete_note: str) -> str:
    lines = ["## synap-harnessx: live verification", "",
             f"Scope `user_id={user_id}` `customer_id={customer_id}`, {len(SEED_DOCUMENTS)} seed documents.", ""]
    if lifecycle:
        lines += ["### Ingestion", "", "| ingestion | status | wait ms | memories |", "|---|---|---:|---:|"]
        lines += [f"| `{r['ingestion_id'][:8]}` | {r['status']} | {r['ms']:.0f} | {r['memories_created']} |"
                  for r in lifecycle]
        lines.append("")
    lines += ["### Latency", "", "| call | p50 ms | p95 ms |", "|---|---:|---:|"]
    lines += [f"| {r['what']} | {r['p50']:.0f} | {r['p95']:.0f} |" for r in latency]
    lines += ["", "### Checks", "", "| check | result | detail | ms |", "|---|---|---|---:|"]
    lines += [f"| {r['check']} | {'pass' if r['ok'] else '**FAIL**'} | {r['detail']} | {r['ms']:.0f} |" for r in checks]
    lines += ["", "### Cleanup", "", delete_note, ""]
    return "\n".join(lines)


# -- dry run ---------------------------------------------------------------


def build_fake_sdk() -> Any:
    """An in-memory stand-in, so the script can be checked without credentials.

    It keeps what is created and recalls by shared words, scoped by user id. It
    proves the script's own logic, not Synap's behaviour.
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    store: List[Tuple[str, str, str]] = []  # (user_id, id, text)

    def words(text: str) -> set:
        return {w.strip(".,'").lower() for w in text.split() if len(w) > 3}

    async def create(document, user_id=None, **_):
        mid = str(uuid.uuid4())
        store.append((user_id, mid, document))
        return SimpleNamespace(ingestion_id=uuid.uuid4(), memory_id=mid)

    async def fetch(user_id=None, search_query=None, **_):
        query = words(" ".join(search_query or []))
        # Stands in for semantic matching: a lone topic word ties a question to a memory.
        topics = {"release": "ship", "formatter": "formatter", "database": "databases", "theme": "theme"}
        query |= {topics[w] for w in query if w in topics}
        hits = [t for u, _, t in store if u == user_id and (words(t) & query)]
        return SimpleNamespace(formatted_context="\n".join(f"- {h}" for h in hits))

    async def status(ingestion_id):
        ids = [m for _, m, _ in store]
        return SimpleNamespace(status="completed", memories_created=1, memory_ids=ids[-1:])

    sdk = MagicMock()
    sdk.memories.create = AsyncMock(side_effect=create)
    sdk.memories.wait_for_completion = AsyncMock()
    sdk.memories.status = AsyncMock(side_effect=status)
    sdk.memories.delete = AsyncMock()
    sdk.fetch = AsyncMock(side_effect=fetch)
    sdk.conversation.record_message = AsyncMock()
    sdk.instance.is_listening = False
    sdk.instance.listen = AsyncMock()
    sdk.instance.stop_listening = AsyncMock()
    for name in ("send_message", "record_tool_call", "record_tool_result", "end_session"):
        setattr(sdk.instance, name, AsyncMock())

    async def listen() -> None:
        sdk.instance.is_listening = True

    sdk.instance.listen = AsyncMock(side_effect=listen)
    return sdk


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", help="reuse an existing user_id instead of seeding")
    parser.add_argument("--customer", help="customer_id for a B2B instance")
    parser.add_argument("--keep", action="store_true", help="do not delete seeded memories")
    parser.add_argument("--out", help="write the results block to this file")
    parser.add_argument("--dry-run", action="store_true", help="run against an in-memory fake")
    args = parser.parse_args()

    if args.dry_run:
        sdk = build_fake_sdk()
        customer_id: Optional[str] = None
    else:
        api_key = os.environ.get("SYNAP_API_KEY", "").strip()
        if not api_key:
            print("SYNAP_API_KEY is not set. This script needs a live instance (or --dry-run).", file=sys.stderr)
            return 2
        from maximem_synap import MaximemSynapSDK

        sdk = MaximemSynapSDK(api_key=api_key, instance_id=os.environ.get("SYNAP_INSTANCE_ID") or None)
        await sdk.initialize()
        # B2C rejects a customer_id with HTTP 400, so it is sent only on B2B.
        isolation = getattr(sdk, "_user_context_isolation", None)
        print(f"instance user_context_isolation={isolation}")
        if args.customer:
            customer_id = args.customer
        elif isolation == "equals_customer":
            customer_id = None
        else:
            customer_id = f"hx-verify-co-{uuid.uuid4().hex[:8]}"

    user_id = args.scope or f"hx-verify-{uuid.uuid4().hex[:10]}"

    capture = LogCapture()
    for name in ("synap_harnessx", "synap_integrations_common"):
        target = logging.getLogger(name)
        target.addHandler(capture)
        target.setLevel(logging.DEBUG)
        target.propagate = False  # captured above; keep stderr to the report

    lifecycle: List[Dict[str, Any]] = []
    if not args.scope:
        lifecycle = await seed(sdk, user_id, customer_id)

    print("running checks ...")
    checks, latency, stored = await run_checks(sdk, user_id, customer_id, capture)

    delete_note = "skipped (--keep)"
    if not args.keep:
        print("cleaning up ...")
        delete_note = await cleanup(sdk, lifecycle, stored)

    block = render(user_id, customer_id, lifecycle, checks, latency, delete_note)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(block)
        print(f"\nwrote {args.out}")
    else:
        print("\n" + block)
    return 0 if all(row["ok"] for row in checks) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
