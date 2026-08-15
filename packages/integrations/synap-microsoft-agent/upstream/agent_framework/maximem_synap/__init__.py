# Copyright (c) Microsoft. All rights reserved.

"""Maximem Synap integration namespace for optional Agent Framework connectors.

This module lazily re-exports objects from:
- ``maximem-synap-microsoft-agent``

Supported classes:
- SynapAgentFileStore
- SynapContextProvider
- SynapHistoryProvider
- SynapMemoryContextProvider
- SynapMemoryStore
- SynapShortTermContextProvider
- create_synap_harness_memory
"""

import importlib
from typing import Any

IMPORT_PATH = "synap_microsoft_agent"
PACKAGE_NAME = "maximem-synap-microsoft-agent"
_IMPORTS = [
    "SynapAgentFileStore",
    "SynapContextProvider",
    "SynapHistoryProvider",
    "SynapMemoryContextProvider",
    "SynapMemoryStore",
    "SynapShortTermContextProvider",
    "create_synap_harness_memory",
]


def __getattr__(name: str) -> Any:
    if name in _IMPORTS:
        try:
            return getattr(importlib.import_module(IMPORT_PATH), name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"The '{PACKAGE_NAME}' package is not installed, please do `pip install {PACKAGE_NAME}`"
            ) from exc
    raise AttributeError(f"Module {IMPORT_PATH} has no attribute {name}.")


def __dir__() -> list[str]:
    return _IMPORTS
