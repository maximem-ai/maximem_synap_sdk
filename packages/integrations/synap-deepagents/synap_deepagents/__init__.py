"""Synap memory integration for deepagents.

deepagents is LangChain's agent harness. It reaches storage through
``BackendProtocol`` — a filesystem interface — and reaches memory through
``MemoryMiddleware``, which pastes whole ``AGENTS.md`` files into the system
prompt. This package plugs Synap into both, and adds a retrieval path the stock
middleware cannot express.

Three surfaces, useful separately or together:

**1. A backend.** :class:`SynapBackend` implements ``BackendProtocol`` so the
agent's own ``read_file`` / ``grep`` / ``write_file`` tools reach Synap. Mount it
on a route — never as the default backend, or the agent loses its working tree::

    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, FilesystemBackend
    from maximem_synap import MaximemSynapSDK
    from synap_deepagents import SynapBackend

    sdk = MaximemSynapSDK(api_key="sk-...")

    backend = CompositeBackend(
        default=FilesystemBackend(root_dir="/path/to/repo"),
        routes={"/memories/": SynapBackend(sdk, user_id="alice")},
    )

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-5",
        backend=backend,
        memory=["/memories/AGENTS.md"],
    )

Note what ``grep`` does here: on a Synap route it is a **semantic search**, not a
regex match. The agent asks a question in plain language and gets an answer.

**2. Middleware.** :class:`SynapMemoryMiddleware` retrieves memory using the
user's actual question as the search query, which ``MemoryMiddleware`` cannot do
— it only knows file paths. Use it instead of ``memory=[...]``, not alongside::

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-5",
        middleware=[SynapMemoryMiddleware(sdk=sdk, user_id="alice")],
    )

:class:`SynapShortTermMiddleware` does the same for compacted conversation
history, refreshed each turn.

**3. Tools.** :class:`SynapSearchTool` and :class:`SynapStoreTool` are ordinary
LangChain tools, for when you want the model to reach for memory deliberately
rather than have it arrive automatically.

Error policy: reads degrade — a Synap outage yields an empty memory block and a
logged error, never a failed agent run. Writes raise ``SynapIntegrationError``.
There is no ``delete``: ``memories.create`` returns an ingestion id rather than a
memory id, so a path cannot be resolved back to a durable memory. Remove
memories through the Synap API with a memory id.
"""

from synap_deepagents.backend import (
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_RECALL_FILENAME,
    SynapBackend,
)
from synap_deepagents.middleware import (
    SYNAP_MEMORY_PROMPT,
    SynapMemoryMiddleware,
    SynapShortTermMiddleware,
)
from synap_deepagents.short_term import (
    compose_system_prompt,
    fetch_st_block,
    synap_st_instructions,
)
from synap_deepagents.tools import SynapSearchTool, SynapStoreTool

__all__ = [
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_RECALL_FILENAME",
    "SYNAP_MEMORY_PROMPT",
    "SynapBackend",
    "SynapMemoryMiddleware",
    "SynapSearchTool",
    "SynapShortTermMiddleware",
    "SynapStoreTool",
    "compose_system_prompt",
    "fetch_st_block",
    "synap_st_instructions",
]
