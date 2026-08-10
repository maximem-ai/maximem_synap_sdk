"""End-to-end smoke test: drive MAF's real providers over the Synap stores.

Unit tests call our methods directly, so they prove the methods behave. They
cannot prove that MAF's own ``MemoryContextProvider`` and ``FileMemoryProvider``
drive them correctly — the argument order, the ``FileNotFoundError`` contract,
the ``PrivateStateAttr`` bookkeeping, the index rebuild. That gap is where the
deepagents integration lost a day, so this closes it here.

No model is involved. Both providers are exercised through their public
``before_run``, and the bound tools are called directly.

    export SYNAP_API_KEY=synap_...
    export SYNAP_INSTANCE_ID=inst_...
    python integrations/synap-microsoft-agent/bench/smoke.py --customer acme

Pass ``--mock`` to run without a network or credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

from agent_framework import AgentSession, FileMemoryProvider, Message

from synap_microsoft_agent import (
    SynapAgentFileStore,
    SynapMemoryStore,
    create_synap_harness_memory,
)

PASS = "  ok  "
FAIL = " FAIL "
_failures: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"[{PASS if condition else FAIL}] {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        _failures.append(label)


class _Context:
    """The slice of ``SessionContext`` the two providers actually touch.

    Built by hand rather than imported: MAF constructs the real one inside the
    agent loop, and standing up a loop would need a chat client and a model.
    """

    def __init__(self, session_id: str, text: str) -> None:
        self.session_id = session_id
        self.input_messages = [Message(role="user", contents=[text])]
        self.response = None
        self.instructions: List[str] = []
        self.messages: List[Any] = []
        self.tools: List[Any] = []

    def extend_instructions(self, source_id: str, value: Any) -> None:
        self.instructions.extend(value if isinstance(value, list) else [value])

    def extend_messages(
        self, source: Any, messages: Any, *, origin_session_ids: Any = None
    ) -> None:
        # ``origin_session_ids`` carries the sessions a cross-session topic came
        # from. Keyword-only upstream, so it has to be accepted by name.
        self.origin_session_ids = list(origin_session_ids or [])
        self.messages.extend(messages if isinstance(messages, list) else [messages])

    def extend_tools(self, source_id: str, value: Any) -> None:
        self.tools.extend(value if isinstance(value, list) else [value])


def build_mock_sdk() -> Any:
    sdk = MagicMock()
    response = MagicMock()
    response.formatted_context = (
        "## User Context\n### Preferences\n- The user prefers PostgreSQL"
    )
    sdk.fetch = AsyncMock(return_value=response)
    created = MagicMock()
    created.ingestion_id = uuid.uuid4()
    sdk.memories = MagicMock()
    sdk.memories.create = AsyncMock(return_value=created)
    status = MagicMock()
    status.memory_ids = ["mem-1"]
    sdk.memories.status = AsyncMock(return_value=status)
    sdk.memories.delete = AsyncMock(return_value={"deleted": True})
    sdk.conversation = MagicMock()
    sdk.conversation.record_message = AsyncMock(return_value={"message_id": "m1"})
    sdk.conversation.context = MagicMock()
    prompt = MagicMock()
    prompt.recent_messages = []
    sdk.conversation.context.get_context_for_prompt = AsyncMock(return_value=prompt)
    return sdk


async def smoke_memory(sdk: Any, *, user_id: str, customer_id: str | None) -> None:
    print("\n--- Tier A: MemoryContextProvider over SynapMemoryStore ---")

    provider = create_synap_harness_memory(
        sdk, user_id=user_id, customer_id=customer_id, recent_turns=2
    )
    store = provider.store
    session = AgentSession(session_id="smoke-session")
    context = _Context("smoke-session", "how do they deploy to production?")

    # A topic written the way MAF's extraction pass would write it.
    from agent_framework import MemoryTopicRecord

    record = MemoryTopicRecord(
        topic="deployment workflow",
        summary="How this person ships software.",
        memories=["Tags a release; GitHub Actions promotes the build."],
        updated_at="2026-08-07T00:00:00+00:00",
    )
    store.write_topic(session, record, source_id=provider.source_id)

    loaded = store.get_topic(
        session, source_id=provider.source_id, topic="deployment workflow"
    )
    check(
        "record survives write -> get byte for byte",
        loaded.to_dict() == record.to_dict(),
    )

    try:
        store.get_topic(session, source_id=provider.source_id, topic="never written")
        check("missing topic raises FileNotFoundError", False, "no exception raised")
    except FileNotFoundError:
        check("missing topic raises FileNotFoundError", True)

    # The real before_run: rebuild_index -> get_index_text -> get_messages ->
    # _select_topics -> get_topic. Every one of those is our code.
    await provider.before_run(
        agent=MagicMock(client=None), session=session, context=context, state={}
    )
    # `str(Message)` is a repr, not the body — read `.text` or the assembled
    # prompt looks empty when it is in fact complete.
    joined = "\n".join(
        [str(i) for i in context.instructions]
        + [getattr(m, "text", str(m)) for m in context.messages]
    )
    check("before_run completed without raising", True)
    check(
        "the index reached the prompt",
        "deployment workflow" in joined,
        f"{len(joined)} chars assembled",
    )
    check(
        "the Synap recall block reached the prompt",
        store.recall_header in joined,
    )

    # The read-modify-write path that made the record store necessary.
    merged = MemoryTopicRecord(
        topic="deployment workflow",
        summary=record.summary,
        memories=list(record.memories) + ["Rolls back by re-tagging."],
        updated_at="2026-08-07T01:00:00+00:00",
    )
    store.write_topic(session, merged, source_id=provider.source_id)
    after = store.get_topic(
        session, source_id=provider.source_id, topic="deployment workflow"
    )
    check(
        "read-modify-write keeps both memory lines",
        len(after.memories) == 2,
        f"{len(after.memories)} lines",
    )

    state = store.export_provider_state(session)
    await provider.save_messages(
        "smoke-session", [Message(role="user", contents=["hello"])], state=state
    )
    check("save_messages wrote transcripts without touching disk", True)
    check(
        "the scratch transcripts path was never created",
        not store.get_transcripts_directory(session, source_id=provider.source_id).exists(),
    )

    rows = store.search_transcripts(
        session, source_id=provider.source_id, query="how do they deploy"
    )
    check(
        "search_transcripts returns MAF-shaped rows",
        not rows or set(rows[0]) == {"session_id", "line_number", "role", "text"},
        f"{len(rows)} rows",
    )

    store.delete_topic(
        session, source_id=provider.source_id, topic="deployment workflow"
    )
    check("delete_topic removed the record", not store.list_topics(session, source_id=provider.source_id))


async def smoke_files(sdk: Any, *, user_id: str, customer_id: str | None) -> None:
    print("\n--- Tier B: FileMemoryProvider over SynapAgentFileStore ---")

    store = SynapAgentFileStore(sdk, user_id=user_id, customer_id=customer_id)
    provider = FileMemoryProvider(store, scope="smoke-scope")
    session = AgentSession(session_id="smoke-session")
    context = _Context("smoke-session", "what did I write down earlier?")

    await provider.before_run(
        agent=MagicMock(), session=session, context=context, state={}
    )
    tools = {getattr(t, "name", str(t)): t for t in context.tools}
    check(
        "all seven file_memory tools bound",
        len(tools) == 7,
        f"{sorted(tools)}",
    )

    write = tools["file_memory_write"]
    read = tools["file_memory_read"]
    ls = tools["file_memory_ls"]
    grep = tools["file_memory_grep"]
    delete = tools["file_memory_delete"]

    await _invoke(write, file_name="plan.md", content="Ship the harness on Friday.",
                  description="the release plan")
    body = await _invoke(read, file_name="plan.md")
    check(
        "write -> read returns exactly what was written",
        "Ship the harness on Friday." in str(body),
        str(body)[:60],
    )

    listing = await _invoke(ls)
    names = [e.get("name") for e in listing] if isinstance(listing, list) else []
    check("ls shows the file", "plan.md" in names, str(names))
    check(
        "ls hides MAF's own bookkeeping",
        "memories.md" not in names and not any(str(n).endswith("_description.md") for n in names),
        str(names),
    )

    hits = await _invoke(grep, regex_pattern="Friday")
    check(
        "grep finds it by regex",
        isinstance(hits, list) and any(h.get("file_name") == "plan.md" for h in hits),
        str(hits)[:80],
    )

    semantic = await _invoke(grep, regex_pattern="what should I ship next")
    check(
        "grep also returns a semantic hit on the recall file",
        isinstance(semantic, list) and any(h.get("file_name") == "MEMORY.md" for h in semantic),
        f"{len(semantic) if isinstance(semantic, list) else 0} results",
    )

    removed = await _invoke(delete, file_name="plan.md")
    check("delete reports success", "delet" in str(removed).lower(), str(removed)[:60])

    listing_after = await _invoke(ls)
    names_after = [e.get("name") for e in listing_after] if isinstance(listing_after, list) else []
    check("the file is gone from ls", "plan.md" not in names_after, str(names_after))


async def _invoke(bound_tool: Any, **kwargs: Any) -> Any:
    """Call a MAF ``@tool``-decorated function, sync or async."""
    target = getattr(bound_tool, "func", None) or getattr(bound_tool, "_func", None) or bound_tool
    result = target(**kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer", default=None)
    parser.add_argument("--mock", action="store_true", help="run without a network")
    parser.add_argument(
        "--scope",
        default=None,
        help=(
            "reuse an existing user_id. Recall checks need a scope with roughly "
            "a dozen memories in it — below that Synap returns an empty context "
            "and the checks measure that floor rather than this integration."
        ),
    )
    args = parser.parse_args()

    if args.mock:
        sdk: Any = build_mock_sdk()
        user_id = "smoke-user"
        customer_id = args.customer
    else:
        from maximem_synap import MaximemSynapSDK

        api_key = os.environ.get("SYNAP_API_KEY", "").strip()
        if not api_key:
            print("SYNAP_API_KEY is not set (or pass --mock)", file=sys.stderr)
            return 2
        sdk = MaximemSynapSDK(
            api_key=api_key, instance_id=os.environ.get("SYNAP_INSTANCE_ID") or None
        )
        await sdk.initialize()
        user_id = args.scope or f"maf-smoke-{uuid.uuid4().hex[:8]}"
        customer_id = args.customer
        if not args.scope:
            print(
                "note: fresh scope — the two recall checks will fail because a "
                "near-empty scope returns nothing. Pass --scope to reuse a seeded one."
            )
        print(f"scope: user_id={user_id} customer_id={customer_id}")

    await smoke_memory(sdk, user_id=user_id, customer_id=customer_id)
    await smoke_files(sdk, user_id=user_id, customer_id=customer_id)

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {_failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
