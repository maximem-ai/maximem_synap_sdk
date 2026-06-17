"""Tests for synap_claude_agent public surface (__init__.py exports).

Every symbol listed in ``__all__`` must be importable at the package level and
must be the canonical object (not a re-export shadow).
"""

from __future__ import annotations

import inspect


def test_package_importable():
    import synap_claude_agent  # noqa: F401


def test_all_exports_defined():
    import synap_claude_agent
    assert hasattr(synap_claude_agent, "__all__")
    assert len(synap_claude_agent.__all__) >= 3


def test_create_synap_hooks_exported():
    import synap_claude_agent
    assert "create_synap_hooks" in synap_claude_agent.__all__
    assert hasattr(synap_claude_agent, "create_synap_hooks")


def test_create_synap_mcp_server_exported():
    import synap_claude_agent
    assert "create_synap_mcp_server" in synap_claude_agent.__all__
    assert hasattr(synap_claude_agent, "create_synap_mcp_server")


def test_create_synap_st_hook_exported():
    import synap_claude_agent
    assert "create_synap_st_hook" in synap_claude_agent.__all__
    assert hasattr(synap_claude_agent, "create_synap_st_hook")


def test_create_synap_hooks_is_callable():
    from synap_claude_agent import create_synap_hooks
    assert callable(create_synap_hooks)


def test_create_synap_mcp_server_is_callable():
    from synap_claude_agent import create_synap_mcp_server
    assert callable(create_synap_mcp_server)


def test_create_synap_st_hook_is_callable():
    from synap_claude_agent import create_synap_st_hook
    assert callable(create_synap_st_hook)


def test_create_synap_hooks_is_same_object_as_module():
    """Ensure the re-export is the same function object, not a copy."""
    from synap_claude_agent import create_synap_hooks
    from synap_claude_agent.hooks import create_synap_hooks as _hooks_create_synap_hooks
    assert create_synap_hooks is _hooks_create_synap_hooks


def test_create_synap_mcp_server_is_same_object_as_module():
    from synap_claude_agent import create_synap_mcp_server
    from synap_claude_agent.mcp_server import create_synap_mcp_server as _mcp_create
    assert create_synap_mcp_server is _mcp_create


def test_create_synap_st_hook_is_same_object_as_module():
    from synap_claude_agent import create_synap_st_hook
    from synap_claude_agent.short_term import create_synap_st_hook as _st_create
    assert create_synap_st_hook is _st_create


def test_create_synap_hooks_signature():
    """create_synap_hooks(sdk, user_id, ...) — positional args are sdk and user_id."""
    from synap_claude_agent import create_synap_hooks
    sig = inspect.signature(create_synap_hooks)
    params = list(sig.parameters)
    assert params[0] == "sdk"
    assert params[1] == "user_id"


def test_create_synap_mcp_server_signature():
    from synap_claude_agent import create_synap_mcp_server
    sig = inspect.signature(create_synap_mcp_server)
    params = list(sig.parameters)
    assert params[0] == "sdk"
    assert params[1] == "user_id"


def test_create_synap_st_hook_signature():
    from synap_claude_agent import create_synap_st_hook
    sig = inspect.signature(create_synap_st_hook)
    params = list(sig.parameters)
    assert params[0] == "sdk"
    assert params[1] == "conversation_id"


def test_no_unexpected_public_names_in_all():
    """__all__ contains exactly the three documented public functions."""
    import synap_claude_agent
    expected = {"create_synap_hooks", "create_synap_mcp_server", "create_synap_st_hook"}
    assert set(synap_claude_agent.__all__) == expected


def test_all_exported_objects_are_functions():
    import synap_claude_agent
    for name in synap_claude_agent.__all__:
        obj = getattr(synap_claude_agent, name)
        assert callable(obj), f"{name} must be callable"
