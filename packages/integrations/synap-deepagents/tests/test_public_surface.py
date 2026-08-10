"""Guards on the package's public surface.

These assert the contract users and docs depend on. A failure here means a
published import path changed, which is a breaking change even when every
other test still passes.
"""

import inspect

import pytest

import synap_deepagents


EXPECTED_EXPORTS = {
    "SYNAP_MEMORY_PROMPT",
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_RECALL_FILENAME",
    "SynapBackend",
    "SynapMemoryMiddleware",
    "SynapSearchTool",
    "SynapShortTermMiddleware",
    "SynapStoreTool",
    "compose_system_prompt",
    "fetch_st_block",
    "synap_st_instructions",
}


def test_all_matches_expected_exports():
    assert set(synap_deepagents.__all__) == EXPECTED_EXPORTS


def test_all_is_sorted():
    assert synap_deepagents.__all__ == sorted(synap_deepagents.__all__)


@pytest.mark.parametrize("name", sorted(EXPECTED_EXPORTS))
def test_every_export_is_importable(name):
    assert getattr(synap_deepagents, name, None) is not None


def test_no_version_attribute():
    """Version lives only in pyproject.toml — see the integration playbook."""
    assert not hasattr(synap_deepagents, "__version__")


def test_module_docstring_shows_the_composite_mount():
    """The README and docstring must never show a bare ``backend=SynapBackend``.

    Passing it as the default backend routes the agent's source-code reads
    through a memory API and loses the working tree.
    """
    doc = synap_deepagents.__doc__
    assert "CompositeBackend" in doc
    assert "backend=SynapBackend(" not in doc


@pytest.mark.parametrize(
    "name",
    ["SynapBackend", "SynapMemoryMiddleware", "SynapShortTermMiddleware",
     "SynapSearchTool", "SynapStoreTool"],
)
def test_public_classes_are_documented(name):
    obj = getattr(synap_deepagents, name)
    assert inspect.getdoc(obj), f"{name} has no docstring"


@pytest.mark.parametrize(
    "name", ["synap_st_instructions", "fetch_st_block", "compose_system_prompt"]
)
def test_public_functions_are_documented(name):
    obj = getattr(synap_deepagents, name)
    assert inspect.getdoc(obj), f"{name} has no docstring"


def test_backend_implements_the_protocol():
    from deepagents.backends.protocol import BackendProtocol

    assert issubclass(synap_deepagents.SynapBackend, BackendProtocol)


def test_backend_is_not_a_sandbox():
    """Synap is not a shell. ``execute`` must not be advertised."""
    from deepagents.backends.protocol import SandboxBackendProtocol

    assert not issubclass(synap_deepagents.SynapBackend, SandboxBackendProtocol)


@pytest.mark.parametrize(
    "method",
    [
        "ls", "als", "read", "aread", "grep", "agrep", "glob", "aglob",
        "write", "awrite", "edit", "aedit",
        "upload_files", "aupload_files", "download_files", "adownload_files",
    ],
)
def test_backend_implements_both_sync_and_async_halves(method):
    """``MemoryMiddleware`` has sync and async entry points and picks by caller.

    A backend implementing only one half breaks the other silently — the
    failure surfaces as empty memory, not as an error.
    """
    from deepagents.backends.protocol import BackendProtocol

    own = getattr(synap_deepagents.SynapBackend, method)
    base = getattr(BackendProtocol, method)
    assert own is not base, f"{method} falls through to the protocol default"
