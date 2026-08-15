"""Phase 5 — MemoryFileStore vs SynapMemoryStore, on identical corpora.

MAF ships no memory benchmark, so this is the honest minimum: the same harness
memory provider, the same topics, the same model, two stores underneath.

What is being compared
----------------------

Both arms run MAF's real ``MemoryContextProvider.before_run``, which is where
the harness assembles its memory into the prompt: rebuild the index, render
``MEMORY.md``, score topics against the question, load the winners. The
assembled text is then handed to one model call and the answer is graded on
substring presence.

That isolates the memory subsystem, which is the only thing the two arms
differ in. Running a full agent loop would add tool-calling noise and measure
the model more than the store.

The difference between the arms is narrow and worth stating plainly. Topic
*records* are identical — both stores hold them exactly, in process. What
differs is:

- ``MemoryFileStore`` selects topics by **word overlap** between the question
  and each topic's title and summary, and renders nothing else.
- ``SynapMemoryStore`` renders the same pointer lines **plus a retrieval block**
  fetched from Synap.

So the fixtures are built to separate those two things, and one of them is
built for the file store to win.

Grading is deliberately dumb — case-insensitive substring presence — because a
model-graded benchmark measures the grader.

    export SYNAP_API_KEY=synap_...
    export SYNAP_INSTANCE_ID=inst_...
    export OPENAI_API_KEY=sk-...

    python integrations/synap-microsoft-agent/bench/conformance.py --dry-run
    python integrations/synap-microsoft-agent/bench/conformance.py --customer acme

Run ``--dry-run`` first. It seeds and assembles both arms without spending a
model token, and it is what catches a broken arm before the results look like
a finding.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

from agent_framework import (
    AgentSession,
    MemoryContextProvider,
    MemoryFileStore,
    MemoryTopicRecord,
    Message,
)

from synap_microsoft_agent import SynapMemoryStore, create_synap_harness_memory

SOURCE_ID = "memory"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Topic:
    topic: str
    summary: str
    memories: Sequence[str]


@dataclass(frozen=True)
class Fixture:
    """One comparison.

    Attributes:
        name: Identifier used in the results table.
        topics: Seeded into both arms, identically.
        question: Asked of both arms.
        expect_contains: Every string must appear in the answer.
        expect_absent: No string may appear in the answer.
        rationale: Which arm should win, and why. Written before the run.
    """

    name: str
    topics: Sequence[Topic]
    question: str
    expect_contains: Sequence[str]
    rationale: str
    expect_absent: Sequence[str] = field(default_factory=tuple)


# Background topics mixed into every fixture. Two jobs: they push the Synap
# scope past the floor below which it returns nothing at all, and they give the
# lexical selector enough candidates that picking the right one is a real
# choice rather than the only one. Both arms get the identical set.
BACKGROUND: Sequence[Topic] = (
    Topic("code review", "What this person looks for when reviewing code.",
          ["Cares most about error handling at the boundaries.",
           "Internal helpers can be terse; anything touching the network cannot."]),
    Topic("testing", "How this person runs tests.",
          ["Runs integration tests against a disposable container.",
           "Treats a flaky test as an outage and quarantines it the same day."]),
    Topic("secrets", "How this person handles credentials.",
          ["Keeps secrets in a managed vault.", "Rotates them quarterly."]),
    Topic("planning", "How this person plans work.",
          ["Writes a design doc before large changes.",
           "Circulates it for a week before starting."]),
    Topic("monitoring", "What this person watches in production.",
          ["Monitors p99 latency rather than averages.",
           "Averages hide the failures users actually notice."]),
    Topic("pull requests", "How this person prefers changes to arrive.",
          ["Prefers small pull requests reviewed same-day.",
           "Dislikes large batched reviews."]),
    Topic("documentation", "How this person treats docs.",
          ["Keeps a runbook per service.",
           "Treats a missing runbook as a launch blocker."]),
    Topic("rollout", "How changes reach users.",
          ["Uses feature flags for anything user-visible.",
           "Rollout is decoupled from deploy."]),
)


VOCABULARY_MISMATCH = Fixture(
    name="vocabulary-mismatch",
    topics=(
        *BACKGROUND,
        Topic(
            "billing",
            "How the finance side of the product works.",
            [
                "Customers are charged monthly in arrears, on the first of the month.",
                "A failed charge retries three times over five days before suspension.",
            ],
        ),
    ),
    question="When do we send invoices to customers, and what happens if one fails?",
    expect_contains=("month",),
    rationale=(
        "The topic is filed under 'billing'; the question says 'invoices'. MAF's "
        "selector scores topics by word overlap, so 'billing' scores zero against "
        "this question and the topic is never loaded. Synap should reach it by "
        "meaning. This is the single clearest expression of the gap."
    ),
)


SUPERSESSION = Fixture(
    name="supersession",
    topics=(
        *BACKGROUND,
        Topic(
            "database choice",
            "Which database this person uses.",
            [
                "Used MongoDB for the first prototype in 2023.",
                "Migrated to PostgreSQL in 2024 and now reaches for it first.",
                "Would only consider a document store for genuinely document-shaped data.",
            ],
        ),
    ),
    question="Which database should I use for a new service here?",
    expect_contains=("postgres",),
    # `expect_absent=("mongodb",)` was the original grader and it was WRONG.
    # The 2026-08-07 run failed the file arm on this answer:
    #
    #   "PostgreSQL. That's our default since 2024 — only consider a document
    #    store like MongoDB if the data is genuinely document-shaped."
    #
    # That answer is correct, and arguably better than the Synap arm's, which
    # said the same thing without naming the alternative. The record legitimately
    # mentions MongoDB as the superseded choice, so a model that explains the
    # supersession has to name it. The grader was measuring word avoidance, not
    # supersession. Scored as a tie in the results, and the rule is gone.
    rationale=(
        "Both arms hold the same record, so both can see the supersession. This "
        "checks that the Synap arm's extra recall block does not resurface the "
        "superseded fact as if it were current. Expect a tie; a Synap loss here "
        "would be a real problem with the recall block."
    ),
)


INDEX_OVERFLOW = Fixture(
    name="index-overflow",
    topics=(
        *BACKGROUND,
        *[
            Topic(
                f"service {index}",
                f"Notes on internal service number {index}.",
                [f"Service {index} is owned by the platform team."],
            )
            for index in range(24)
        ],
        Topic(
            "incident response",
            "What to do when production breaks.",
            [
                "Page the on-call engineer through Opsgenie, never Slack.",
                "Open an incident channel before starting any mitigation.",
            ],
        ),
    ),
    question="How do I page someone when production breaks?",
    expect_contains=("opsgenie",),
    rationale=(
        "Thirty-three topics against a selection limit of three. The right topic "
        "does share words with the question, so the file store can find it — but "
        "the index itself is now long. This measures how much of the prompt each "
        "arm spends to deliver the same answer."
    ),
)


SINGLE_SHORT_SESSION = Fixture(
    name="single-short-session",
    topics=(
        Topic(
            "deployment workflow",
            "How this person ships software.",
            ["Ships by tagging a release; GitHub Actions promotes the build."],
        ),
    ),
    question="How does this person deploy?",
    expect_contains=("tag",),
    rationale=(
        "One topic, worded the same way as the question. The file store should "
        "win: no network call, and lexical overlap is all that is needed. Written "
        "expecting Synap to lose, and reported as a loss if it does."
    ),
)


FIXTURES: Sequence[Fixture] = (
    VOCABULARY_MISMATCH,
    SUPERSESSION,
    INDEX_OVERFLOW,
    SINGLE_SHORT_SESSION,
)


# ---------------------------------------------------------------------------
# Prompt assembly — the real MemoryContextProvider, per arm
# ---------------------------------------------------------------------------


class _Context:
    """The slice of ``SessionContext`` ``before_run`` touches."""

    def __init__(self, session_id: str, question: str) -> None:
        self.session_id = session_id
        self.input_messages = [Message(role="user", contents=[question])]
        self.response = None
        self.instructions: List[str] = []
        self.messages: List[Any] = []
        self.tools: List[Any] = []

    def extend_instructions(self, source_id: str, value: Any) -> None:
        self.instructions.extend(value if isinstance(value, list) else [value])

    def extend_messages(self, source: Any, messages: Any, *, origin_session_ids: Any = None) -> None:
        del origin_session_ids
        self.messages.extend(messages if isinstance(messages, list) else [messages])

    def extend_tools(self, source_id: str, value: Any) -> None:
        self.tools.extend(value if isinstance(value, list) else [value])

    def assembled(self) -> str:
        # `str(Message)` is a repr, not the body. Reading it would report an
        # empty prompt for a complete one.
        return "\n".join(
            [str(i) for i in self.instructions]
            + [getattr(m, "text", str(m)) for m in self.messages]
        )


def _record(topic: Topic, updated_at: str) -> MemoryTopicRecord:
    return MemoryTopicRecord(
        topic=topic.topic,
        summary=topic.summary,
        memories=list(topic.memories),
        updated_at=updated_at,
    )


async def build_file_arm(fixture: Fixture, root: Path) -> MemoryContextProvider:
    store = MemoryFileStore(root, owner_state_key="owner")
    provider = MemoryContextProvider(store=store, source_id=SOURCE_ID)
    session = AgentSession(session_id="bench")
    session.state["owner"] = "bench-owner"
    for index, topic in enumerate(fixture.topics):
        store.write_topic(
            session, _record(topic, f"2026-08-0{1 + index % 7}T00:00:00+00:00"),
            source_id=SOURCE_ID,
        )
    return provider


async def build_synap_arm(
    fixture: Fixture, sdk: Any, *, user_id: str, customer_id: Optional[str]
) -> MemoryContextProvider:
    provider = create_synap_harness_memory(
        sdk, user_id=user_id, customer_id=customer_id
    )
    session = AgentSession(session_id="bench")
    ingestions = []
    for index, topic in enumerate(fixture.topics):
        provider.store.write_topic(
            session, _record(topic, f"2026-08-0{1 + index % 7}T00:00:00+00:00"),
            source_id=SOURCE_ID,
        )
    # Every write is queued. Without waiting, the recall block measures
    # ingestion lag rather than retrieval, and the arm looks worse than it is.
    for ids in provider.store._ingestions.values():
        ingestions.extend(ids)
    await asyncio.gather(
        *(
            sdk.memories.wait_for_completion(ingestion_id, timeout_seconds=600)
            for ingestion_id in ingestions
        )
    )
    return provider


async def assemble(provider: MemoryContextProvider, fixture: Fixture) -> Tuple[str, float]:
    session = AgentSession(session_id="bench")
    session.state["owner"] = "bench-owner"
    context = _Context("bench", fixture.question)
    start = time.perf_counter()
    await provider.before_run(
        agent=MagicMock(client=None), session=session, context=context, state={}
    )
    elapsed = (time.perf_counter() - start) * 1000.0
    return context.assembled(), elapsed


# ---------------------------------------------------------------------------
# Model call + grading
# ---------------------------------------------------------------------------

SYSTEM = (
    "Answer the user's question using only the memory below. If the memory does "
    "not contain the answer, say you do not know. Be brief.\n\n{memory}"
)


async def ask(memory: str, question: str, model: str) -> Tuple[str, float]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM.format(memory=memory)},
            {"role": "user", "content": question},
        ],
    )
    elapsed = (time.perf_counter() - start) * 1000.0
    return (response.choices[0].message.content or "").strip(), elapsed


def grade(fixture: Fixture, answer: str) -> Tuple[bool, str]:
    lowered = answer.lower()
    missing = [s for s in fixture.expect_contains if s.lower() not in lowered]
    present = [s for s in fixture.expect_absent if s.lower() in lowered]
    if missing:
        return False, f"missing {missing}"
    if present:
        return False, f"should not mention {present}"
    return True, ""


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


async def run_cell(
    arm: str,
    fixture: Fixture,
    *,
    sdk: Any,
    model: str,
    customer_id: Optional[str],
    dry_run: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"arm": arm, "fixture": fixture.name}
    try:
        # Setup is inside the boundary. Seeding is a long network operation and
        # one blip should cost one cell, not the whole run.
        if arm == "file":
            with tempfile.TemporaryDirectory() as tmp:
                provider = await build_file_arm(fixture, Path(tmp))
                memory, assemble_ms = await assemble(provider, fixture)
                result.update(_measure(memory, assemble_ms))
                if not dry_run:
                    answer, ask_ms = await ask(memory, fixture.question, model)
                    result.update(_judge(fixture, answer, ask_ms))
        else:
            user_id = f"maf-bench-{fixture.name}-{uuid.uuid4().hex[:6]}"
            provider = await build_synap_arm(
                fixture, sdk, user_id=user_id, customer_id=customer_id
            )
            result["scope"] = user_id
            memory, assemble_ms = await assemble(provider, fixture)
            result.update(_measure(memory, assemble_ms))
            if not dry_run:
                answer, ask_ms = await ask(memory, fixture.question, model)
                result.update(_judge(fixture, answer, ask_ms))
    except Exception as exc:  # noqa: BLE001 — one bad cell, not one bad run
        result.update({"error": f"{type(exc).__name__}: {exc}", "passed": False})
    return result


def _measure(memory: str, assemble_ms: float) -> Dict[str, Any]:
    return {
        "memory_chars": len(memory),
        "assemble_ms": round(assemble_ms, 1),
    }


def _judge(fixture: Fixture, answer: str, ask_ms: float) -> Dict[str, Any]:
    passed, reason = grade(fixture, answer)
    return {
        "passed": passed,
        "reason": reason,
        "answer": answer[:200],
        "ask_ms": round(ask_ms, 1),
    }


def render(rows: List[Dict[str, Any]], *, dry_run: bool) -> str:
    out = ["", "=" * 78]
    out.append("MAF harness conformance — MemoryFileStore vs SynapMemoryStore")
    if dry_run:
        out.append("(dry run: prompts assembled, no model called)")
    out.append("=" * 78)
    header = f"{'fixture':<24}{'arm':<8}{'chars':>8}{'assemble':>10}"
    if not dry_run:
        header += f"{'pass':>7}{'ask ms':>9}"
    out.append(header)
    out.append("-" * 78)
    for row in rows:
        line = (
            f"{row['fixture']:<24}{row['arm']:<8}"
            f"{row.get('memory_chars', 0):>8}{row.get('assemble_ms', 0):>9.0f}ms"
        )
        if not dry_run:
            line += f"{('yes' if row.get('passed') else 'NO'):>7}{row.get('ask_ms', 0):>8.0f}ms"
        out.append(line)
        if row.get("error"):
            out.append(f"    ERROR {row['error']}")
        elif row.get("reason"):
            out.append(f"    {row['reason']}  |  {row.get('answer', '')[:110]}")
    out.append("")
    return "\n".join(out)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer", default=None)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--dry-run", action="store_true", help="assemble only, no model")
    parser.add_argument("--fixtures", default=None, help="comma-separated names")
    parser.add_argument("--arms", default="file,synap")
    parser.add_argument("--out", default=None, help="write raw rows as JSON")
    args = parser.parse_args()

    api_key = os.environ.get("SYNAP_API_KEY", "").strip()
    if not api_key:
        print("SYNAP_API_KEY is not set", file=sys.stderr)
        return 2
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (or pass --dry-run)", file=sys.stderr)
        return 2

    from maximem_synap import MaximemSynapSDK

    sdk = MaximemSynapSDK(
        api_key=api_key, instance_id=os.environ.get("SYNAP_INSTANCE_ID") or None
    )
    await sdk.initialize()

    selected = FIXTURES
    if args.fixtures:
        wanted = {n.strip() for n in args.fixtures.split(",")}
        selected = [f for f in FIXTURES if f.name in wanted]
    arms = [a.strip() for a in args.arms.split(",")]

    rows: List[Dict[str, Any]] = []
    for fixture in selected:
        for arm in arms:
            print(f"running {fixture.name} / {arm} ...")
            rows.append(
                await run_cell(
                    arm,
                    fixture,
                    sdk=sdk,
                    model=args.model,
                    customer_id=args.customer,
                    dry_run=args.dry_run,
                )
            )

    print(render(rows, dry_run=args.dry_run))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"raw rows -> {args.out}")

    print("Rationales (written before the run):")
    for fixture in selected:
        print(f"  {fixture.name}: {fixture.rationale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
