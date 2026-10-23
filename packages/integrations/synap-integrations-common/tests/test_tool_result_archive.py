"""report_tool_result keeps the result in the SDK's local archive (20 Oct, decision B)."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from synap_integrations_common.stream_events import report_tool_result


def sdk(listening):
    s = MagicMock()
    s.instance.is_listening = listening
    s.instance.record_tool_result = AsyncMock()
    return s


@pytest.mark.asyncio
async def test_without_a_stream_the_result_is_archived_locally():
    s = sdk(False)
    assert await report_tool_result(s, result={"ok": True}, tool_name="ping", conversation_id="c1") is False
    s.archive_tool_result.assert_called_once_with("c1", {"ok": True}, tool_name="ping")
    s.instance.record_tool_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_with_a_stream_the_sdk_archives_it_itself():
    s = sdk(True)
    assert await report_tool_result(s, result={"ok": True}, tool_name="ping", conversation_id="c1") is True
    s.archive_tool_result.assert_not_called()  # record_tool_result does it
    s.instance.record_tool_result.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_archive_failure_never_breaks_the_run():
    s = sdk(False)
    s.archive_tool_result.side_effect = RuntimeError("disk full")
    assert await report_tool_result(s, result="x", conversation_id="c1") is False
