"""NAT plugin entry point.

Importing this module has the side effect of registering
:class:`SynapMemoryClientConfig` + the ``synap_memory`` YAML ``_type``
with NAT's :class:`GlobalTypeRegistry`. Wired via the ``nat.components``
entry-point in ``pyproject.toml`` so NAT picks it up automatically on
workflow load.
"""

# flake8: noqa — side-effectful import; don't remove
# isort:skip_file

from . import plugin  # noqa: F401 — registers @register_memory
