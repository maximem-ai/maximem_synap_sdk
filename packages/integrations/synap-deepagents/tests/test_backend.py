"""Tests for :mod:`synap_deepagents.backend`."""

from unittest.mock import AsyncMock

import pytest
from deepagents.backends.protocol import FILE_NOT_FOUND
from synap_integrations_common import SynapIntegrationError

from synap_deepagents.backend import SynapBackend, _normalize, _WriteCache

RECALL = "/AGENTS.md"


def make_backend(sdk, **kwargs):
    kwargs.setdefault("user_id", "alice")
    return SynapBackend(sdk, **kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_requires_sdk():
    with pytest.raises(ValueError, match="non-None sdk"):
        SynapBackend(None, user_id="alice")


def test_requires_a_scope(mock_sdk):
    with pytest.raises(ValueError, match="user_id or customer_id"):
        SynapBackend(mock_sdk)


def test_customer_id_alone_is_a_valid_scope(mock_sdk):
    backend = SynapBackend(mock_sdk, customer_id="acme")
    assert backend.customer_id == "acme"
    assert backend.user_id is None


def test_requires_non_empty_recall_filename(mock_sdk):
    with pytest.raises(ValueError, match="recall_filename"):
        SynapBackend(mock_sdk, user_id="alice", recall_filename="   ")


def test_defaults(mock_sdk):
    backend = make_backend(mock_sdk)
    assert backend.recall_filename == "AGENTS.md"
    assert backend.mode == "fast"
    assert backend.grep_mode == "accurate"
    assert backend.document_type == "document"
    assert backend.include_conversation_context is False


def test_repr_mentions_scope(mock_sdk):
    assert "alice" in repr(make_backend(mock_sdk))


# ---------------------------------------------------------------------------
# Path normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AGENTS.md", "/AGENTS.md"),
        ("/AGENTS.md", "/AGENTS.md"),
        ("//AGENTS.md", "/AGENTS.md"),
        ("  /notes.md  ", "/notes.md"),
        ("/dir/", "/dir"),
        ("/", "/"),
        ("", "/"),
    ],
)
def test_normalize(raw, expected):
    assert _normalize(raw) == expected


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


async def test_aread_recall_returns_formatted_context(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.aread(RECALL)
    assert result.error is None
    assert "User is an engineer" in result.file_data["content"]
    assert result.file_data["encoding"] == "utf-8"


async def test_aread_recall_uses_no_search_query(mock_sdk):
    backend = make_backend(mock_sdk)
    await backend.aread(RECALL)
    assert mock_sdk.fetch.await_args.kwargs["search_query"] is None
    assert mock_sdk.fetch.await_args.kwargs["mode"] == "fast"


async def test_aread_sets_valid_pagination_window(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.aread(RECALL)
    assert result.start_line == 1
    assert result.end_line == result.total_lines
    assert result.next_offset == result.end_line


async def test_aread_respects_offset_and_limit(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.aread(RECALL, offset=1, limit=1)
    assert result.start_line == 2
    assert result.end_line == 2
    assert result.next_offset == 2
    assert result.file_data["content"] == "### Facts"


async def test_aread_zero_limit_flags_uninspected(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.aread(RECALL, limit=0)
    assert result.no_lines_requested is True
    assert result.file_data is None
    assert result.error is None


async def test_aread_negative_offset_is_clamped(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.aread(RECALL, offset=-5)
    assert result.start_line == 1


async def test_aread_offset_past_end_is_file_not_found(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.aread(RECALL, offset=999)
    assert result.error == FILE_NOT_FOUND


async def test_aread_degrades_to_file_not_found_on_sdk_failure(failing_sdk):
    backend = make_backend(failing_sdk)
    result = await backend.aread(RECALL)
    assert result.error == FILE_NOT_FOUND
    assert result.file_data is None


async def test_aread_empty_context_is_file_not_found(mock_sdk):
    mock_sdk.fetch.return_value.formatted_context = ""
    backend = make_backend(mock_sdk)
    result = await backend.aread(RECALL)
    assert result.error == FILE_NOT_FOUND


async def test_aread_unknown_path_searches_by_basename(mock_sdk):
    backend = make_backend(mock_sdk)
    await backend.aread("/billing-notes.md")
    assert mock_sdk.fetch.await_args.kwargs["search_query"] == ["billing notes"]


def test_read_sync_wrapper(mock_sdk):
    backend = make_backend(mock_sdk)
    result = backend.read(RECALL)
    assert result.error is None
    assert "engineer" in result.file_data["content"]


# ---------------------------------------------------------------------------
# grep — the query seam
# ---------------------------------------------------------------------------


async def test_agrep_passes_pattern_as_search_query(mock_sdk):
    backend = make_backend(mock_sdk)
    await backend.agrep("what does the user do for work")
    assert mock_sdk.fetch.await_args.kwargs["search_query"] == [
        "what does the user do for work"
    ]


async def test_agrep_uses_grep_mode_not_read_mode(mock_sdk):
    backend = make_backend(mock_sdk)
    await backend.agrep("anything")
    assert mock_sdk.fetch.await_args.kwargs["mode"] == "accurate"


async def test_agrep_returns_one_match_per_line(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.agrep("work")
    assert result.error is None
    assert len(result.matches) == 3
    assert result.matches[0]["line"] == 1
    assert result.matches[0]["path"] == RECALL


async def test_agrep_skips_blank_lines(mock_sdk):
    mock_sdk.fetch.return_value.formatted_context = "one\n\n\ntwo"
    backend = make_backend(mock_sdk)
    result = await backend.agrep("q")
    assert [m["text"] for m in result.matches] == ["one", "two"]


async def test_agrep_honours_max_count(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.agrep("work", max_count=2)
    assert len(result.matches) == 2
    assert result.truncated is True


async def test_agrep_uses_supplied_path_for_matches(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.agrep("work", path="/notes.md")
    assert all(m["path"] == "/notes.md" for m in result.matches)


async def test_agrep_of_the_root_attributes_matches_to_the_recall_file(mock_sdk):
    """A grep of the mount root must return a readable file path.

    ``CompositeBackend`` strips its route prefix on the way in and re-adds it
    to every returned path, so a grep of ``/memories/`` reaches us as ``/``.
    Echoing ``/`` back would surface as ``/memories/`` — a directory the agent
    cannot read.
    """
    backend = make_backend(mock_sdk)
    for path in ("/", None):
        result = await backend.agrep("work", path=path)
        assert all(m["path"] == RECALL for m in result.matches)


@pytest.mark.parametrize("pattern", ["", "   ", None])
async def test_agrep_empty_pattern_short_circuits(mock_sdk, pattern):
    backend = make_backend(mock_sdk)
    result = await backend.agrep(pattern)
    assert result.matches == []
    mock_sdk.fetch.assert_not_awaited()


async def test_agrep_degrades_on_sdk_failure(failing_sdk):
    backend = make_backend(failing_sdk)
    result = await backend.agrep("anything")
    assert result.error is None
    assert result.matches == []


def test_grep_sync_wrapper(mock_sdk):
    backend = make_backend(mock_sdk)
    assert len(backend.grep("work").matches) == 3


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


async def test_awrite_calls_memories_create(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.awrite("/notes.md", "User ships on Fridays")
    assert result.error is None
    assert result.path == "/notes.md"
    kwargs = mock_sdk.memories.create.await_args.kwargs
    assert kwargs["document"] == "User ships on Fridays"
    assert kwargs["user_id"] == "alice"
    assert kwargs["document_type"] == "document"
    assert kwargs["mode"] == "fast"


async def test_awrite_raises_on_sdk_failure(failing_sdk):
    backend = make_backend(failing_sdk)
    with pytest.raises(SynapIntegrationError):
        await backend.awrite("/notes.md", "content")


async def test_write_is_readable_back_immediately(mock_sdk):
    backend = make_backend(mock_sdk)
    await backend.awrite("/notes.md", "User ships on Fridays")
    result = await backend.aread("/notes.md")
    assert result.file_data["content"] == "User ships on Fridays"


async def test_read_after_write_does_not_hit_the_sdk(mock_sdk):
    backend = make_backend(mock_sdk)
    await backend.awrite("/notes.md", "cached body")
    mock_sdk.fetch.reset_mock()
    await backend.aread("/notes.md")
    mock_sdk.fetch.assert_not_awaited()


async def test_cache_can_be_disabled(mock_sdk):
    backend = make_backend(mock_sdk, cache_ttl_seconds=0)
    await backend.awrite("/notes.md", "not cached")
    result = await backend.aread("/notes.md")
    # Falls through to a Synap search rather than the local copy.
    assert "not cached" not in (result.file_data or {}).get("content", "")


def test_write_sync_wrapper(mock_sdk):
    backend = make_backend(mock_sdk)
    assert backend.write("/notes.md", "body").path == "/notes.md"


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


async def test_aedit_stores_only_the_addition(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.aedit(RECALL, "existing", "existing\nnew fact")
    assert result.occurrences == 1
    assert mock_sdk.memories.create.await_args.kwargs["document"] == "new fact"


async def test_aedit_stores_whole_string_when_old_absent(mock_sdk):
    backend = make_backend(mock_sdk)
    await backend.aedit(RECALL, "not-present", "a brand new fact")
    assert (
        mock_sdk.memories.create.await_args.kwargs["document"] == "a brand new fact"
    )


async def test_aedit_pure_deletion_is_a_noop(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.aedit(RECALL, "gone", "gone")
    assert result.occurrences == 0
    mock_sdk.memories.create.assert_not_awaited()


async def test_aedit_raises_on_sdk_failure(failing_sdk):
    backend = make_backend(failing_sdk)
    with pytest.raises(SynapIntegrationError):
        await backend.aedit(RECALL, "old", "old\nnew")


async def test_aedit_appends_to_cached_body(mock_sdk):
    backend = make_backend(mock_sdk)
    await backend.awrite("/notes.md", "first")
    await backend.aedit("/notes.md", "first", "first\nsecond")
    result = await backend.aread("/notes.md")
    assert result.file_data["content"] == "first\nsecond"


def test_edit_sync_wrapper(mock_sdk):
    backend = make_backend(mock_sdk)
    assert backend.edit(RECALL, "a", "a\nb").occurrences == 1


# ---------------------------------------------------------------------------
# ls / glob
# ---------------------------------------------------------------------------


async def test_als_always_shows_the_recall_file(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.als("/")
    assert [e["path"] for e in result.entries] == [RECALL]


async def test_als_includes_written_paths(mock_sdk):
    backend = make_backend(mock_sdk)
    await backend.awrite("/notes.md", "body")
    result = await backend.als("/")
    assert set(e["path"] for e in result.entries) == {RECALL, "/notes.md"}


async def test_als_reports_size_for_written_entries(mock_sdk):
    backend = make_backend(mock_sdk)
    await backend.awrite("/notes.md", "12345")
    entry = next(e for e in (await backend.als("/")).entries if e["path"] == "/notes.md")
    assert entry["size"] == 5
    assert entry["is_dir"] is False


async def test_aglob_matches_by_extension(mock_sdk):
    backend = make_backend(mock_sdk)
    await backend.awrite("/notes.txt", "body")
    result = await backend.aglob("*.md")
    assert [m["path"] for m in result.matches] == [RECALL]


async def test_aglob_no_match_returns_empty(mock_sdk):
    backend = make_backend(mock_sdk)
    result = await backend.aglob("*.py")
    assert result.matches == []
    assert result.error is None


def test_ls_and_glob_sync_wrappers(mock_sdk):
    backend = make_backend(mock_sdk)
    assert backend.ls("/").entries
    assert backend.glob("*.md").matches


# ---------------------------------------------------------------------------
# download_files — the MemoryMiddleware path
# ---------------------------------------------------------------------------


async def test_adownload_files_returns_content(mock_sdk):
    backend = make_backend(mock_sdk)
    [response] = await backend.adownload_files([RECALL])
    assert response.error is None
    assert b"engineer" in response.content


async def test_adownload_files_preserves_order(mock_sdk):
    backend = make_backend(mock_sdk)
    paths = [RECALL, "/missing.md", RECALL]
    responses = await backend.adownload_files(paths)
    assert [r.path for r in responses] == paths


async def test_adownload_files_only_ever_returns_file_not_found(failing_sdk):
    """The load-bearing guarantee.

    ``MemoryMiddleware.abefore_agent`` raises ``ValueError`` on any error code
    except ``file_not_found``. A Synap outage that surfaced anything else would
    end the agent run instead of degrading to an empty memory block.
    """
    backend = make_backend(failing_sdk)
    responses = await backend.adownload_files(["/a.md", "/b.md", RECALL])
    assert all(r.error == FILE_NOT_FOUND for r in responses)
    assert all(r.content is None for r in responses)


async def test_adownload_files_empty_list(mock_sdk):
    backend = make_backend(mock_sdk)
    assert await backend.adownload_files([]) == []


def test_download_files_sync_wrapper(mock_sdk):
    backend = make_backend(mock_sdk)
    assert backend.download_files([RECALL])[0].error is None


# ---------------------------------------------------------------------------
# upload_files
# ---------------------------------------------------------------------------


async def test_aupload_files_writes_text(mock_sdk):
    backend = make_backend(mock_sdk)
    [response] = await backend.aupload_files([("/notes.md", b"hello")])
    assert response.error is None
    assert mock_sdk.memories.create.await_args.kwargs["document"] == "hello"


async def test_aupload_files_rejects_binary(mock_sdk):
    backend = make_backend(mock_sdk)
    [response] = await backend.aupload_files([("/img.png", b"\xff\xfe\x00binary")])
    assert response.error is not None
    assert "binary" in response.error
    mock_sdk.memories.create.assert_not_awaited()


async def test_aupload_files_raises_on_sdk_failure(failing_sdk):
    backend = make_backend(failing_sdk)
    with pytest.raises(SynapIntegrationError):
        await backend.aupload_files([("/notes.md", b"hello")])


def test_upload_files_sync_wrapper(mock_sdk):
    backend = make_backend(mock_sdk)
    assert backend.upload_files([("/a.md", b"x")])[0].error is None


# ---------------------------------------------------------------------------
# delete — deliberately unimplemented
# ---------------------------------------------------------------------------


def test_delete_is_not_supported(mock_sdk):
    """``memories.create`` returns an ingestion id, not a memory id.

    A path written through this backend cannot be resolved back to a durable
    memory, so honouring a path-addressed delete is impossible. ``delete`` is
    optional in the protocol; we inherit the default rather than pretend.
    """
    backend = make_backend(mock_sdk)
    with pytest.raises(NotImplementedError):
        backend.delete(RECALL)


async def test_adelete_is_not_supported(mock_sdk):
    backend = make_backend(mock_sdk)
    with pytest.raises(NotImplementedError):
        await backend.adelete(RECALL)


# ---------------------------------------------------------------------------
# scope plumbing
# ---------------------------------------------------------------------------


async def test_scope_ids_are_passed_to_fetch(mock_sdk):
    backend = SynapBackend(
        mock_sdk, user_id="alice", customer_id="acme", conversation_id="conv-1"
    )
    await backend.aread(RECALL)
    kwargs = mock_sdk.fetch.await_args.kwargs
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] == "acme"
    assert kwargs["conversation_id"] == "conv-1"


async def test_scope_ids_are_passed_to_create(mock_sdk):
    backend = SynapBackend(mock_sdk, user_id="alice", customer_id="acme")
    await backend.awrite("/n.md", "body")
    kwargs = mock_sdk.memories.create.await_args.kwargs
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] == "acme"


async def test_retrieval_knobs_are_forwarded(mock_sdk):
    backend = make_backend(
        mock_sdk, max_results=7, precision_level="medium", mode="accurate"
    )
    await backend.aread(RECALL)
    kwargs = mock_sdk.fetch.await_args.kwargs
    assert kwargs["max_results"] == 7
    assert kwargs["precision_level"] == "medium"
    assert kwargs["mode"] == "accurate"


async def test_custom_recall_filename(mock_sdk):
    backend = make_backend(mock_sdk, recall_filename="MEMORY.md")
    result = await backend.aread("/MEMORY.md")
    assert result.error is None
    assert mock_sdk.fetch.await_args.kwargs["search_query"] is None


async def test_custom_ingest_settings(mock_sdk):
    backend = make_backend(
        mock_sdk, document_type="email", ingest_mode="long-range"
    )
    await backend.awrite("/n.md", "body")
    kwargs = mock_sdk.memories.create.await_args.kwargs
    assert kwargs["document_type"] == "email"
    assert kwargs["mode"] == "long-range"


# ---------------------------------------------------------------------------
# _WriteCache
# ---------------------------------------------------------------------------


def test_cache_round_trip():
    cache = _WriteCache(ttl_seconds=60)
    cache.put("/a.md", "body")
    assert cache.get("/a.md") == "body"
    assert cache.get("a.md") == "body"  # normalized


def test_cache_expires():
    cache = _WriteCache(ttl_seconds=0)
    cache.put("/a.md", "body")
    assert cache.get("/a.md") is None


def test_cache_paths_are_sorted():
    cache = _WriteCache()
    cache.put("/b.md", "x")
    cache.put("/a.md", "x")
    assert cache.paths() == ["/a.md", "/b.md"]


def test_cache_drop():
    cache = _WriteCache()
    cache.put("/a.md", "x")
    cache.drop("/a.md")
    assert cache.get("/a.md") is None


def test_cache_miss_returns_none():
    assert _WriteCache().get("/nope.md") is None


# ---------------------------------------------------------------------------
# partial-failure isolation
# ---------------------------------------------------------------------------


async def test_one_failing_path_does_not_poison_the_batch(mock_sdk):
    backend = make_backend(mock_sdk)
    await backend.awrite("/good.md", "cached body")

    calls = {"n": 0}

    async def flaky(**kwargs):
        calls["n"] += 1
        raise RuntimeError("boom")

    mock_sdk.fetch = AsyncMock(side_effect=flaky)
    responses = await backend.adownload_files(["/good.md", "/bad.md"])
    assert responses[0].content == b"cached body"
    assert responses[1].error == FILE_NOT_FOUND
