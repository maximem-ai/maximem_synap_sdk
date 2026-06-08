"""NAT plugin entry point.

Importing this module has the side effect of registering both:

- :class:`SynapMemoryClientConfig` (YAML ``_type: synap_memory``) — backs
  Synap as a long-term memory provider via :class:`SynapMemoryEditor`.
- :class:`SynapShortTermConfig` (YAML ``_type: synap_short_term``) — a
  workflow Function returning Synap short-term context for the
  current conversation.

Wired via the ``nat.components`` entry-point in ``pyproject.toml`` so
NAT picks it up automatically on workflow load.
"""

# flake8: noqa — side-effectful imports; don't remove
# isort:skip_file

from . import plugin  # noqa: F401 — registers @register_memory
from . import short_term  # noqa: F401 — registers @register_function (synap_short_term)
