"""Tests for ``SynapAgentFileStore`` — the Tier B harness file surface.

Two behaviours carry the most weight:

- ``delete`` returns ``True`` only when something was really removed. It backs
  a model-facing tool, so a ``True`` the agent cannot verify is a lie it acts on.
- MAF's bookkeeping files (``memories.md``, ``*_description.md``) are cached but
  never ingested. Sending them to Synap would fill the user's scope with prose
  about file names.
"""

from __future__ import annotations

import re

import pytest
from unittest.mock import AsyncMock

from tests._harness import requires_harness, wire_ingestion

pytestmark = requires_harness


def build(sdk, **kwargs):
    from synap_microsoft_agent import SynapAgentFileStore

    kwargs.setdefault("user_id", "alice")
    kwargs.setdefault("customer_id", "acme")
    return SynapAgentFileStore(sdk, **kwargs)


class TestConstruction:
    def test_rejects_none_sdk(self):
        from synap_microsoft_agent import SynapAgentFileStore

        with pytest.raises(ValueError, match="non-None sdk"):
            SynapAgentFileStore(None, user_id="alice")

    def test_rejects_missing_scope(self, mock_sdk):
        from synap_microsoft_agent import SynapAgentFileStore

        with pytest.raises(ValueError, match="user_id or customer_id"):
            SynapAgentFileStore(mock_sdk)

    def test_rejects_blank_recall_filename(self, mock_sdk):
        from synap_microsoft_agent import SynapAgentFileStore

        with pytest.raises(ValueError, match="recall_filename"):
            SynapAgentFileStore(mock_sdk, user_id="alice", recall_filename="  ")

    def test_satisfies_the_abc(self):
        from synap_microsoft_agent import SynapAgentFileStore

        assert not SynapAgentFileStore.__abstractmethods__


class TestPathRules:
    """Matches MAF's own ``_normalize_relative_path``, which is private."""

    @pytest.mark.parametrize("bad", ["/rooted.md", "..", "a/../b.md", "C:/x.md", "  "])
    async def test_rejected_paths(self, mock_sdk, bad):
        with pytest.raises(ValueError):
            await build(mock_sdk).read(bad)

    async def test_trailing_separator_is_rejected_for_files(self, mock_sdk):
        with pytest.raises(ValueError, match="path separator"):
            await build(mock_sdk).read("notes/")

    async def test_empty_directory_means_root(self, mock_sdk):
        assert await build(mock_sdk).list_children("") == []


class TestWriteRead:
    async def test_write_then_read_returns_exactly_what_was_written(self, mock_sdk):
        store = build(mock_sdk)
        await store.write("plan.md", "step one\nstep two")
        assert await store.read("plan.md") == "step one\nstep two"

    async def test_write_submits_to_synap(self, mock_sdk):
        store = build(mock_sdk)
        await store.write("plan.md", "ship on Tuesday")

        mock_sdk.memories.create.assert_awaited_once()
        kwargs = mock_sdk.memories.create.await_args.kwargs
        assert kwargs["document"] == "ship on Tuesday"
        assert kwargs["user_id"] == "alice"

    async def test_read_after_write_does_not_depend_on_synap(self, mock_sdk):
        """Ingestion is queued, so a fresh write is not yet retrievable."""
        store = build(mock_sdk)
        await store.write("plan.md", "ship on Tuesday")
        mock_sdk.fetch.reset_mock()

        assert await store.read("plan.md") == "ship on Tuesday"
        mock_sdk.fetch.assert_not_awaited()

    async def test_overwrite_false_raises_when_present(self, mock_sdk):
        store = build(mock_sdk)
        await store.write("plan.md", "first")
        with pytest.raises(FileExistsError):
            await store.write("plan.md", "second", overwrite=False)

    async def test_overwrite_false_is_fine_when_absent(self, mock_sdk):
        store = build(mock_sdk)
        await store.write("plan.md", "first", overwrite=False)
        assert await store.read("plan.md") == "first"

    async def test_write_failure_raises(self, failing_sdk):
        from synap_integrations_common import SynapIntegrationError

        with pytest.raises(SynapIntegrationError):
            await build(failing_sdk).write("plan.md", "content")

    async def test_recall_file_is_synthesized_from_synap(self, mock_sdk):
        content = await build(mock_sdk).read("MEMORY.md")
        assert content and "User is an engineer" in content
        assert mock_sdk.fetch.await_args.kwargs["search_query"] is None

    async def test_unknown_file_becomes_a_question(self, mock_sdk):
        await build(mock_sdk).read("deployment-notes.md")
        assert mock_sdk.fetch.await_args.kwargs["search_query"] == ["deployment notes"]

    async def test_read_degrades_to_none_on_failure(self, failing_sdk):
        assert await build(failing_sdk).read("anything.md") is None

    async def test_file_exists_follows_read(self, mock_sdk):
        store = build(mock_sdk)
        await store.write("plan.md", "content")
        assert await store.file_exists("plan.md") is True

    async def test_file_exists_is_false_when_nothing_comes_back(self, mock_sdk):
        mock_sdk.fetch = AsyncMock(return_value=type("R", (), {"formatted_context": ""})())
        assert await build(mock_sdk).file_exists("unknown.md") is False


class TestInternalFiles:
    """MAF's bookkeeping is cached, never ingested."""

    @pytest.mark.parametrize("name", ["memories.md", "plan_description.md"])
    async def test_bookkeeping_is_not_sent_to_synap(self, mock_sdk, name):
        store = build(mock_sdk)
        await store.write(name, "- plan.md: the plan")
        mock_sdk.memories.create.assert_not_awaited()

    async def test_bookkeeping_is_still_readable_back(self, mock_sdk):
        store = build(mock_sdk)
        await store.write("memories.md", "- plan.md: the plan")
        assert await store.read("memories.md") == "- plan.md: the plan"

    async def test_missing_bookkeeping_is_not_answered_with_recall(self, mock_sdk):
        """Answering a missing index with memory prose would have MAF parse
        sentences as file names."""
        assert await build(mock_sdk).read("memories.md") is None
        mock_sdk.fetch.assert_not_awaited()


class TestDelete:
    async def test_delete_removes_the_file_and_its_memories(self, mock_sdk):
        wire_ingestion(mock_sdk, ["mem-1", "mem-2"])
        store = build(mock_sdk)
        await store.write("plan.md", "content")

        assert await store.delete("plan.md") is True
        assert mock_sdk.memories.delete.await_count == 2
        mock_sdk.fetch.reset_mock()
        mock_sdk.fetch = AsyncMock(return_value=type("R", (), {"formatted_context": ""})())
        assert await store.read("plan.md") is None

    async def test_delete_of_an_unknown_file_is_false(self, mock_sdk):
        wire_ingestion(mock_sdk)
        assert await build(mock_sdk).delete("never-written.md") is False
        mock_sdk.memories.delete.assert_not_awaited()

    async def test_bookkeeping_delete_is_true_without_touching_synap(self, mock_sdk):
        wire_ingestion(mock_sdk)
        store = build(mock_sdk)
        await store.write("plan_description.md", "the plan")

        assert await store.delete("plan_description.md") is True
        mock_sdk.memories.delete.assert_not_awaited()

    async def test_delete_reports_true_when_only_the_cache_had_it(self, mock_sdk):
        """A cached-but-unsubmitted file was still really removed."""
        store = build(mock_sdk, cache_ttl_seconds=3600)
        mock_sdk.memories.create = AsyncMock(return_value=type("R", (), {"ingestion_id": None})())
        wire_ingestion(mock_sdk)
        await store.write("plan.md", "content")

        assert await store.delete("plan.md") is True

    async def test_delete_survives_a_failing_status_call(self, mock_sdk):
        wire_ingestion(mock_sdk)
        mock_sdk.memories.status = AsyncMock(side_effect=RuntimeError("boom"))
        store = build(mock_sdk)
        await store.write("plan.md", "content")

        # The cache entry was still removed, so the answer is honest.
        assert await store.delete("plan.md") is True


class TestListChildren:
    async def test_lists_what_this_process_wrote(self, mock_sdk):
        from agent_framework import FileStoreEntry

        store = build(mock_sdk)
        await store.write("a.md", "one")
        await store.write("b.md", "two")

        names = {e.name for e in await store.list_children("")}
        assert names == {"a.md", "b.md"}
        assert all(e.type == FileStoreEntry.FILE for e in await store.list_children(""))

    async def test_directories_come_first_and_are_deduplicated(self, mock_sdk):
        from agent_framework import FileStoreEntry

        store = build(mock_sdk)
        await store.write("notes/one.md", "one")
        await store.write("notes/two.md", "two")
        await store.write("top.md", "top")

        entries = await store.list_children("")
        assert entries[0].type == FileStoreEntry.DIRECTORY
        assert entries[0].name == "notes"
        assert [e.name for e in entries if e.type == FileStoreEntry.FILE] == ["top.md"]

    async def test_names_are_relative_to_the_directory(self, mock_sdk):
        store = build(mock_sdk)
        await store.write("scope/plan.md", "content")
        assert [e.name for e in await store.list_children("scope")] == ["plan.md"]

    async def test_listing_is_empty_for_a_cold_store(self, mock_sdk):
        """The documented gap: there is no list-memories-by-scope API."""
        assert await build(mock_sdk).list_children("") == []


class TestSearch:
    async def test_regex_matches_cached_files(self, mock_sdk):
        store = build(mock_sdk)
        await store.write("plan.md", "deploy on Tuesday\nrollback on Friday")

        results = await store.search("", "rollback")
        names = {r.file_name for r in results}
        assert "plan.md" in names

        plan = next(r for r in results if r.file_name == "plan.md")
        assert [m.line_number for m in plan.matching_lines] == [2]

    async def test_regex_is_case_insensitive(self, mock_sdk):
        store = build(mock_sdk)
        await store.write("plan.md", "Deploy on Tuesday")
        assert any(r.file_name == "plan.md" for r in await store.search("", "DEPLOY"))

    async def test_semantic_hit_is_attributed_to_the_recall_file(self, mock_sdk):
        results = await build(mock_sdk).search("", "what does this person do")
        assert any(r.file_name == "MEMORY.md" for r in results)
        assert mock_sdk.fetch.await_args.kwargs["search_query"] == [
            "what does this person do"
        ]

    async def test_both_halves_can_return_together(self, mock_sdk):
        store = build(mock_sdk)
        await store.write("plan.md", "engineer notes")

        names = {r.file_name for r in await store.search("", "engineer")}
        assert names == {"plan.md", "MEMORY.md"}

    async def test_glob_filters_the_recall_file_out(self, mock_sdk):
        store = build(mock_sdk)
        await store.write("plan.txt", "engineer notes")

        names = {r.file_name for r in await store.search("", "engineer", "*.txt")}
        assert names == {"plan.txt"}

    async def test_non_recursive_skips_subdirectories(self, mock_sdk):
        store = build(mock_sdk)
        await store.write("notes/deep.md", "engineer")

        names = {r.file_name for r in await store.search("", "engineer", recursive=False)}
        assert "notes/deep.md" not in names

    async def test_recursive_includes_subdirectories(self, mock_sdk):
        store = build(mock_sdk)
        await store.write("notes/deep.md", "engineer")

        names = {r.file_name for r in await store.search("", "engineer", recursive=True)}
        assert "notes/deep.md" in names

    async def test_overlong_pattern_raises_like_maf(self, mock_sdk):
        with pytest.raises(ValueError, match="too long"):
            await build(mock_sdk).search("", "a" * 257)

    async def test_invalid_regex_reaches_the_model_unwrapped(self, mock_sdk):
        """MAF's tool catches ``re.error`` and shows it so the model retries."""
        with pytest.raises(re.error):
            await build(mock_sdk).search("", "[unclosed")

    async def test_search_degrades_when_synap_fails(self, failing_sdk):
        store = build(failing_sdk)
        assert await store.search("", "anything") == []


class TestCreateDirectory:
    async def test_is_a_no_op(self, mock_sdk):
        store = build(mock_sdk)
        await store.create_directory("notes")
        mock_sdk.memories.create.assert_not_awaited()

    async def test_still_validates_the_path(self, mock_sdk):
        with pytest.raises(ValueError):
            await build(mock_sdk).create_directory("../escape")
