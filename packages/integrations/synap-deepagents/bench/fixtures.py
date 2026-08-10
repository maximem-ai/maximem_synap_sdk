"""Fixtures for the deepagents memory conformance run.

Each fixture is a memory corpus plus a question with a checkable answer. The
grading is deliberately dumb — substring presence, case-insensitive — because
an LLM judge would add a second source of noise to a comparison whose whole
point is to isolate one variable: the memory backend.

Two of the fixtures are designed so Synap should win, one so it should lose,
and one so the arms should tie. A benchmark that only contains fixtures you win
is a brochure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass(frozen=True)
class Fixture:
    """One graded scenario.

    Attributes:
        name: Stable identifier used in the results table.
        rationale: Why this fixture exists and which arm should win it.
        memories: The corpus, one durable statement per entry.
        question: What the agent is asked, in one turn.
        expect_contains: Every string must appear in the final answer.
        expect_absent: No string may appear in the final answer.
    """

    name: str
    rationale: str
    memories: Sequence[str]
    question: str
    expect_contains: Sequence[str]
    expect_absent: Sequence[str] = field(default_factory=tuple)


# Background corpus mixed into the small fixtures.
#
# This is not padding for realism's sake. Measured against a live instance on
# 2026-08-06: a scope holding three memories returns an empty context from
# `sdk.fetch` on every mode and precision combination, while the same scope at
# twelve memories returns content reliably. A three-memory fixture therefore
# measures that floor rather than the backend, and the Synap arm fails for a
# reason that has nothing to do with the comparison being made.
#
# Every arm gets the identical corpus, so this costs the file-shaped arms some
# prompt budget and advantages nobody.
BACKGROUND: Sequence[str] = (
    "I prefer asynchronous written updates over status meetings, and I ask my "
    "team to post a short written summary rather than schedule a call.",
    "I read code review comments in the morning and write them in the "
    "afternoon, because reviewing well takes more attention than writing.",
    "I keep a running decision log per project and refer back to it during "
    "retrospectives rather than relying on anyone's recollection.",
    "I dislike tools that hide what they are doing. If a command has a dry-run "
    "mode I will use it before the real one, every time.",
    "I onboard new engineers by having them ship something small to production "
    "in the first week, with a pair, rather than reading documentation.",
    "I schedule deep work in the morning and leave the afternoon for meetings "
    "and reviews, and I protect that split fairly aggressively.",
    "When an incident happens I write the timeline before the analysis, "
    "because the analysis is always wrong if the timeline is.",
    "I would rather ship a smaller change twice than a larger change once, and "
    "I push back on batching work into big releases.",
    "I take notes in plain Markdown files kept in a git repository, not in a "
    "notes application, because I want them to outlive the tool.",
)


def _filler(count: int) -> List[str]:
    """Plausible, irrelevant memories, to make the corpus exceed a paste budget.

    The point of the large-scope fixture is that a whole-file backend has to put
    all of this in the prompt to find the one line that matters. Filler that is
    obviously junk would let a model skim it; these read like real notes.
    """
    topics = [
        "sprint planning", "on-call rotation", "expense reports", "laptop setup",
        "conference travel", "lunch preferences", "meeting cadence", "podcasts",
        "keyboard layout", "note-taking", "reading list", "commute", "desk setup",
        "calendar blocking", "email triage", "slack etiquette", "documentation",
        "onboarding buddies", "retro format", "demo days",
    ]
    out: List[str] = []
    for index in range(count):
        topic = topics[index % len(topics)]
        out.append(
            f"Note {index + 1} about {topic}: the usual approach is unchanged "
            f"from last quarter and nobody has raised a concern about it."
        )
    return out


CROSS_SESSION_IDENTITY = Fixture(
    name="cross-session-identity",
    rationale=(
        "The corpus was written in an earlier session. Every arm can pass this "
        "one — it is the control that proves the harness works."
    ),
    memories=(
        "My name is Priya and I lead the payments platform team.",
        "I have been at the company for four years.",
        "I work from the Bangalore office two days a week.",
    )
    + tuple(BACKGROUND),
    question="Who am I and what do I work on?",
    expect_contains=("priya", "payments"),
)

SUPERSESSION = Fixture(
    name="supersession",
    rationale=(
        "An old fact is contradicted by a newer one. A whole-file paste shows "
        "the model both and lets it pick; a memory service that consolidates "
        "should surface only the current answer."
    ),
    memories=(
        "In 2023 I used MongoDB as the primary datastore for the billing service.",
        "In 2024 I migrated the billing service off MongoDB.",
        "As of 2025 the billing service runs entirely on PostgreSQL, and I "
        "consider the MongoDB era a mistake I do not want repeated.",
    )
    + tuple(BACKGROUND),
    question="What database does my billing service use? Answer with just the name.",
    expect_contains=("postgres",),
    expect_absent=("mongo",),
)

LARGE_SCOPE = Fixture(
    name="large-scope",
    rationale=(
        "The argument. One relevant memory buried in ~200 irrelevant ones. A "
        "whole-file backend must paste the lot; retrieval pastes the one. If "
        "Synap does not win here the integration has no case."
    ),
    memories=tuple(
        _filler(100)
        + [
            "The production database credentials are rotated by the platform "
            "team on the first Monday of every month, and the rotation runbook "
            "lives in the infra repo under runbooks/rotate-db-credentials.md."
        ]
        + _filler(100)
    ),
    question=(
        "Where is the runbook for rotating production database credentials, "
        "and how often does the rotation happen?"
    ),
    expect_contains=("rotate-db-credentials", "month"),
)

SMALL_REPO_TASK = Fixture(
    name="small-repo-task",
    rationale=(
        "Three lines of memory and a direct question. A local file read is one "
        "syscall; a retrieval call is a network round trip. FilesystemBackend "
        "should win on latency and we report that rather than hiding it."
    ),
    memories=(
        "This repository uses uv for dependency management.",
        "Tests run with pytest from the repository root.",
        "The default branch is main.",
    )
    + tuple(BACKGROUND),
    question="What command do I use to run the tests in this repository?",
    expect_contains=("pytest",),
)

FIXTURES: Sequence[Fixture] = (
    CROSS_SESSION_IDENTITY,
    SUPERSESSION,
    LARGE_SCOPE,
    SMALL_REPO_TASK,
)
