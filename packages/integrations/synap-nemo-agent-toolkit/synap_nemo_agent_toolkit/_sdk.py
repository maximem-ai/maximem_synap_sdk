"""One Synap SDK per credentials, shared by every component in the process.

``synap_memory``, ``synap_short_term``, ``synap_stream`` and ``synap_history``
are separate NAT components, and each used to build, initialise and tear down an
SDK of its own. Anticipation and compaction live on the SDK that has the gRPC
stream open: its prefetch cache is what ``fetch()`` reads, and what it has been
told about the conversation is what ``get_context_for_prompt`` answers from. A
read SDK that never saw the stream the exporter opened got none of it.

So the components share. The first to ask for an SDK builds and initialises it;
the last to release it stops the stream (if one is open) and shuts it down.
Sharing is keyed on the API key and instance id, so two configs pointing at two
instances still get two SDKs.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional, Tuple

from nat.data_models.common import get_secret_value

from maximem_synap import MaximemSynapSDK

logger = logging.getLogger(__name__)


@dataclass
class _Shared:
    loop: asyncio.AbstractEventLoop
    ready: "asyncio.Future[Any]"
    sdk: Any = None
    refs: int = 0
    key: Tuple[str, str] = field(default=("", ""))


_registry: Dict[Tuple[str, str], _Shared] = {}


def _credentials(config: Any) -> Tuple[str, str]:
    api_key: Optional[str] = get_secret_value(getattr(config, "api_key", None)) or os.environ.get(
        "SYNAP_API_KEY"
    )
    if api_key is None:
        raise RuntimeError(
            f"Synap API key is not set. Provide it via {type(config).__name__}.api_key "
            "or the SYNAP_API_KEY environment variable."
        )
    instance_id = getattr(config, "instance_id", "") or os.environ.get("SYNAP_INSTANCE_ID", "")
    return api_key, instance_id


async def _acquire(config: Any) -> _Shared:
    key = _credentials(config)
    loop = asyncio.get_running_loop()
    shared = _registry.get(key)
    if shared is not None and shared.loop.is_closed():
        _registry.pop(key, None)  # a previous run's loop is gone; so is its stream
        shared = None
    if shared is None:
        shared = _Shared(loop=loop, ready=loop.create_future(), key=key)
        _registry[key] = shared
        try:
            sdk = MaximemSynapSDK(instance_id=key[1], api_key=key[0], _force_new=True)
            await sdk.initialize()
        except BaseException as exc:
            _registry.pop(key, None)
            shared.ready.set_exception(exc)
            shared.ready.exception()  # nobody else may be waiting: mark it retrieved
            raise
        shared.sdk = sdk
        shared.ready.set_result(None)
    else:
        await shared.ready
    shared.refs += 1
    return shared


async def _release(shared: _Shared) -> None:
    shared.refs -= 1
    if shared.refs > 0:
        return
    if _registry.get(shared.key) is shared:
        del _registry[shared.key]
    sdk = shared.sdk
    # stop_listening closes the session the stream opened. The telemetry row for
    # a turn is written when the NEXT user message arrives, so without it the
    # last turn of the workflow is never committed.
    with contextlib.suppress(Exception):
        if getattr(getattr(sdk, "instance", None), "is_listening", False):
            await sdk.instance.stop_listening()
    # The SDK's teardown is shutdown(); there is no close(). close() is still
    # accepted second so a duck-typed test double keeps working.
    teardown = getattr(sdk, "shutdown", None) or getattr(sdk, "close", None)
    if teardown is not None:
        try:
            await teardown()
        except Exception:  # noqa: BLE001 — teardown must not mask workflow errors
            logger.exception("synap: SDK teardown raised (suppressing)")


@contextlib.asynccontextmanager
async def shared_sdk(config: Any) -> AsyncIterator[MaximemSynapSDK]:
    """The process's SDK for this config's credentials, initialised."""
    shared = await _acquire(config)
    try:
        yield shared.sdk
    finally:
        await _release(shared)


__all__ = ["shared_sdk"]
