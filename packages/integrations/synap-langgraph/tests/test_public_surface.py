"""Tests for synap_langgraph public surface and create_synap_node re-export.

Documented in __init__.py:
- SynapStore re-exported from synap_langgraph.store
- SynapCheckpointSaver re-exported from synap_langgraph.checkpointer
- create_synap_node re-exported from synap_langchain.graph (backward compat)
- synap_st_prompt re-exported from synap_langgraph.short_term
- create_synap_st_node re-exported from synap_langgraph.short_term
- __all__ lists every export

Coverage shape:
- All five public names importable from synap_langgraph directly
- __all__ completeness
- create_synap_node happy-path invocation via the re-export (validates it
  wires through to langchain.graph correctly)
- create_synap_node failure path via the re-export
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Module-level import check
# ---------------------------------------------------------------------------


def test_all_public_exports_importable():
    from synap_langgraph import (
        SynapCheckpointSaver,
        SynapStore,
        create_synap_node,
        create_synap_st_node,
        synap_st_prompt,
    )
    assert SynapStore is not None
    assert SynapCheckpointSaver is not None
    assert create_synap_node is not None
    assert create_synap_st_node is not None
    assert synap_st_prompt is not None


def test_all_list_is_complete():
    import synap_langgraph as pkg

    expected = {"SynapStore", "SynapCheckpointSaver", "create_synap_node", "create_synap_st_node", "synap_st_prompt"}
    assert set(pkg.__all__) == expected


def test_each_all_name_is_present_as_attribute():
    import synap_langgraph as pkg

    for name in pkg.__all__:
        assert hasattr(pkg, name), f"synap_langgraph is missing __all__ entry: {name!r}"


# ---------------------------------------------------------------------------
# create_synap_node — re-exported from synap_langchain.graph
# ---------------------------------------------------------------------------


def test_create_synap_node_is_callable():
    from synap_langgraph import create_synap_node

    assert callable(create_synap_node)


def test_create_synap_node_returns_callable_with_correct_name():
    from synap_langgraph import create_synap_node

    sdk = MagicMock()
    node = create_synap_node(sdk, user_id="u1")
    assert callable(node)
    assert node.__name__ == "synap_memory"


def test_create_synap_node_raises_on_none_sdk():
    from synap_langgraph import create_synap_node

    with pytest.raises(ValueError, match="non-None sdk"):
        create_synap_node(None, user_id="u1")  # type: ignore[arg-type]


def test_create_synap_node_raises_on_empty_user_id():
    from synap_langgraph import create_synap_node

    sdk = MagicMock()
    with pytest.raises(ValueError, match="non-empty user_id"):
        create_synap_node(sdk, user_id="")


@pytest.mark.asyncio
async def test_create_synap_node_happy_path_via_re_export():
    """The re-exported create_synap_node produces a functional node."""
    from synap_langgraph import create_synap_node
    from synap_integrations_common import SynapIntegrationError

    sdk = MagicMock()
    sdk.fetch = AsyncMock(return_value=MagicMock(formatted_context="memory context"))
    node = create_synap_node(sdk, user_id="u1", conversation_id="conv-1")

    human_msg = MagicMock(type="human", content="what is my budget?")
    result = await node({"messages": [human_msg]})

    assert result == {"synap_context": "memory context"}
    kw = sdk.fetch.call_args.kwargs
    assert kw["search_query"] == ["what is my budget?"]
    assert kw["conversation_id"] == "conv-1"


@pytest.mark.asyncio
async def test_create_synap_node_failure_via_re_export():
    """SDK failure inside the re-exported node raises SynapIntegrationError."""
    from synap_langgraph import create_synap_node
    from synap_integrations_common import SynapIntegrationError

    sdk = MagicMock()
    sdk.fetch = AsyncMock(side_effect=RuntimeError("sdk boom"))
    node = create_synap_node(sdk, user_id="u1")

    with pytest.raises(SynapIntegrationError):
        await node({"messages": []})


# ---------------------------------------------------------------------------
# SynapStore — brief re-export validation
# ---------------------------------------------------------------------------


def test_synap_store_re_export_identity():
    from synap_langgraph import SynapStore
    from synap_langgraph.store import SynapStore as _SynapStore

    assert SynapStore is _SynapStore


# ---------------------------------------------------------------------------
# SynapCheckpointSaver — brief re-export validation
# ---------------------------------------------------------------------------


def test_synap_checkpointsaver_re_export_identity():
    from synap_langgraph import SynapCheckpointSaver
    from synap_langgraph.checkpointer import SynapCheckpointSaver as _SCS

    assert SynapCheckpointSaver is _SCS


# ---------------------------------------------------------------------------
# Short-term exports — brief re-export validation
# ---------------------------------------------------------------------------


def test_synap_st_prompt_re_export_identity():
    from synap_langgraph import synap_st_prompt
    from synap_langgraph.short_term import synap_st_prompt as _ssp

    assert synap_st_prompt is _ssp


def test_create_synap_st_node_re_export_identity():
    from synap_langgraph import create_synap_st_node
    from synap_langgraph.short_term import create_synap_st_node as _cssn

    assert create_synap_st_node is _cssn
