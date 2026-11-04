"""Put Synap's compaction where Pydantic AI's message history goes.

Pydantic AI sends the model whatever ``message_history`` the caller passes plus
the new prompt, and its one seam over that list is a *history processor*: a
callable run before every model request that returns the messages the model
will see. This module's processor asks Synap for the conversation as Synap
holds it and, once a compaction exists, replaces every turn before the current
one with the compaction block and the turns after it
(:func:`synap_integrations_common.splice_history`).

The processor is installed on the agent by :func:`register_synap_tools`,
:func:`register_synap_st_system_prompt` and :func:`synap_run`, so it is on by
default. Pydantic AI's own processors are left alone: they now see a short
history and do not fire, and they are still the answer when Synap has nothing.

Two things about how Pydantic AI treats the result are worth knowing. The
processed list replaces the run's history, so ``result.all_messages()`` carries
the compaction block; a caller who feeds it back as ``message_history`` gets it
replaced again by the next splice, never stacked. And the processor runs on
the first model request of a run only: a run that calls tools makes several,
and the answer cannot change between them.
"""

from __future__ import annotations

import inspect
import logging
import threading
import weakref
from dataclasses import replace
from typing import Any, Callable, List, Optional

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from synap_integrations_common import fetch_compacted_history, splice_history

from synap_pydantic_ai.deps import SynapDeps

logger = logging.getLogger(__name__)

_MARK = "_synap_history_processor"

# Once per process: an agent without the seam is a property of the install.
_warned_no_seam = False


def _text(value: Any) -> str:
    """The text of a prompt, which is a string or a sequence of user content."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(item for item in value if isinstance(item, str))
    return ""


def _role(message: ModelMessage) -> str:
    if isinstance(message, ModelResponse):
        return "assistant"
    if isinstance(message, ModelRequest):
        if any(isinstance(p, UserPromptPart) for p in message.parts):
            return "user"
        if message.parts and all(isinstance(p, SystemPromptPart) for p in message.parts):
            return "system"
    return "other"


def _message_text(message: ModelMessage) -> str:
    if isinstance(message, ModelResponse):
        return "\n".join(p.content for p in message.parts if isinstance(p, TextPart))
    if isinstance(message, ModelRequest):
        kept = (UserPromptPart, SystemPromptPart)
        return "\n".join(_text(p.content) for p in message.parts if isinstance(p, kept))
    return ""


def _make_message(role: str, text: str) -> ModelMessage:
    if role == "assistant":
        return ModelResponse(parts=[TextPart(content=text)])
    if role == "system":
        return ModelRequest(parts=[SystemPromptPart(content=text)])
    return ModelRequest(parts=[UserPromptPart(content=text)])


def _split_system_parts(messages: List[ModelMessage]) -> List[ModelMessage]:
    """One system prompt per message, apart from the user prompt.

    The first request carries the system prompt and the first user prompt
    together, and Pydantic AI merges the compaction block into that request on
    the way back out. The splice treats a message as one role and tells its own
    block by what a message starts with, so pull them apart.
    """
    out: List[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest) and any(
            isinstance(p, SystemPromptPart) for p in message.parts
        ):
            rest = [p for p in message.parts if not isinstance(p, SystemPromptPart)]
            out.extend(ModelRequest(parts=[p]) for p in message.parts
                       if isinstance(p, SystemPromptPart))
            if rest:
                out.append(replace(message, parts=rest))
            continue
        out.append(message)
    return out


def synap_history_processor() -> Callable[..., Any]:
    """A Pydantic AI history processor that swaps in Synap's compaction.

    It reads ``sdk`` and ``conversation_id`` from the run's :class:`SynapDeps`.
    With no ``conversation_id``, with no compaction yet, and on any error, the
    messages are returned as they came.

    Example::

        # Pydantic AI 1.x
        agent = Agent("openai:gpt-4o", deps_type=SynapDeps,
                      history_processors=[synap_history_processor()])

        # Pydantic AI 2.x, where ``history_processors`` no longer exists
        from pydantic_ai.capabilities import ProcessHistory
        agent = Agent("openai:gpt-4o", deps_type=SynapDeps,
                      capabilities=[ProcessHistory(synap_history_processor())])

    The register functions and :func:`synap_run` call :func:`install_synap_history`
    for you, which does the right one of these for the installed version.
    """

    async def _synap_history(
        ctx: RunContext[SynapDeps], messages: List[ModelMessage],
    ) -> List[ModelMessage]:
        deps = ctx.deps
        sdk = getattr(deps, "sdk", None)
        conversation_id = getattr(deps, "conversation_id", None)
        if sdk is None or not conversation_id or ctx.run_step > 1:
            return messages
        history = await fetch_compacted_history(sdk, conversation_id)
        if history is None:
            return messages
        try:
            return splice_history(
                _split_system_parts(messages), history,
                role_of=_role, text_of=_message_text, make_message=_make_message,
            )
        except Exception:  # noqa: BLE001 — the framework's own history still stands
            logger.warning("synap pydantic-ai: splice failed (conversation_id=%s)",
                           conversation_id, exc_info=True)
            return messages

    setattr(_synap_history, _MARK, True)
    return _synap_history


def _is_ours(item: Any) -> bool:
    if getattr(item, _MARK, False):
        return True
    return getattr(getattr(item, "processor", None), _MARK, False)


def _in_constructor_capabilities(agent: Any) -> bool:
    """Whether the agent was BUILT with our processor: ``Agent(capabilities=[...])``."""
    root = getattr(agent, "root_capability", None)
    caps = getattr(root, "capabilities", None)
    return isinstance(caps, (list, tuple)) and any(_is_ours(c) for c in caps)


def _accepts_capabilities(method: Any) -> bool:
    """Whether ``method`` names a ``capabilities`` parameter (not just ``**kwargs``:
    passing an argument the agent does not know is a TypeError in the caller's run)."""
    try:
        return "capabilities" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


def _takes_kwargs(method: Any) -> bool:
    try:
        return any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        return False


def _has_capabilities_seam(agent: Any) -> bool:
    """Whether ``agent.iter(capabilities=...)`` will reach the library.

    The agent's own ``iter`` names the parameter, or it is a wrapper someone put
    on the instance that takes ``(*args, **kwargs)`` and the class's ``iter``
    (what such a wrapper forwards to) names it.
    """
    method = getattr(agent, "iter", None)
    if _accepts_capabilities(method):
        return True
    return _takes_kwargs(method) and _class_iter_reaches_capabilities(type(agent))


def _class_iter_reaches_capabilities(cls: type) -> bool:
    """Whether a call to ``cls.iter(..., capabilities=)`` reaches an ``iter`` that names it.

    Walks the MRO: each ``iter`` before the library's must take ``**kwargs`` (a user
    subclass overriding ``iter(self, *a, **k)`` and calling ``super()``), so the
    argument gets through to the one that declares ``capabilities``.
    """
    for klass in cls.__mro__:
        method = klass.__dict__.get("iter")
        if method is None:
            continue
        if _accepts_capabilities(method):
            return True
        if not _takes_kwargs(method):
            return False
    return False


def _library_names_capabilities(cls: type) -> bool:
    return any(_accepts_capabilities(k.__dict__.get("iter")) for k in cls.__mro__
               if k.__dict__.get("iter") is not None)


def _with_synap_capability(kwargs: dict, capability: Any) -> dict:
    given = kwargs.get("capabilities")
    if given is None or not any(_is_ours(c) for c in given):
        kwargs["capabilities"] = [*(given or ()), capability]
    return kwargs


# Base classes are held weakly, and so are the subclasses made from them: a class
# built at run time (a test, a plugin) is not kept alive by this cache.
_subclasses: "weakref.WeakKeyDictionary[type, weakref.ref]" = weakref.WeakKeyDictionary()
_subclasses_lock = threading.Lock()


def _subclass_with_history(base: type) -> type:
    """``base`` with an ``iter`` that adds the processor to every run.

    The run is the agent in ``self``, so a ``copy.copy(agent)`` made after the
    install runs through its OWN model and tools, not the original's.
    """
    with _subclasses_lock:
        ref = _subclasses.get(base)
        sub = ref() if ref is not None else None
        if sub is None:
            def iter(self, *args: Any, **kwargs: Any) -> Any:  # noqa: A001 - overrides Agent.iter
                from pydantic_ai.capabilities import ProcessHistory
                _with_synap_capability(kwargs, ProcessHistory(synap_history_processor()))
                return super(sub, self).iter(*args, **kwargs)

            setattr(iter, _MARK, True)
            # A name of its own, so pickling an installed agent fails with "attribute
            # lookup failed" on ``...WithSynapHistory`` instead of claiming it is not the
            # same object as ``Agent``. (Agents do not pickle anyway: they hold ContextVars.)
            name = f"{base.__name__}WithSynapHistory"
            sub = type(name, (base,), {"iter": iter, "__module__": base.__module__,
                                       "__qualname__": f"{base.__qualname__}WithSynapHistory"})
            _subclasses[base] = weakref.ref(sub)
        return sub


def _install_per_run(agent: Any) -> bool:
    """Hand the processor to every run of this agent as a per-run capability.

    Pydantic AI 2.x fixes the agent's capability list when the ``Agent`` is
    built (``CombinedCapability._dispatch`` is computed in ``__post_init__``), so
    a ``ProcessHistory`` appended to ``agent.root_capability.capabilities``
    afterwards is never called. Every run entry point (``run``, ``run_sync``,
    ``run_stream``, ``run_stream_events``) goes through ``Agent.iter``, which takes
    ``capabilities=`` for the one run, so ``iter`` is overridden to add the
    processor there.

    The override is a subclass of the agent's class, assigned to ``agent.__class__``
    (the instance keeps its state, ``isinstance`` still holds, and a
    ``copy.copy(agent)`` made later is an instance of the same subclass and so
    runs on its own model). When that is not possible (the instance already has
    its own ``iter``, or the class does not allow it), ``iter`` is wrapped on the
    instance instead: that works, but a copy of such an agent still runs through
    the original's ``iter``.
    """
    from pydantic_ai.capabilities import ProcessHistory

    if "iter" not in getattr(agent, "__dict__", {"iter": None}):
        try:
            agent.__class__ = _subclass_with_history(type(agent))
            return True
        except TypeError:
            pass  # a class that cannot be swapped: wrap the instance below

    original = agent.iter
    capability = ProcessHistory(synap_history_processor())

    def iter_with_synap_history(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **_with_synap_capability(kwargs, capability))

    setattr(iter_with_synap_history, _MARK, True)
    agent.iter = iter_with_synap_history
    return True


def install_synap_history(agent: Any) -> bool:
    """Put :func:`synap_history_processor` on the agent. Idempotent. Returns
    ``True`` only if the processor WILL run on this agent's model requests;
    otherwise ``False`` and one warning is logged.

    How it gets there depends on the Pydantic AI version (verified against the
    installed library source and by running the tests on 1.0.0, 1.107.9, 2.0.0
    and 2.64.0; see the package tests):

    - Pydantic AI 1.0 keeps a ``history_processors`` list on the agent, appended
      to here.
    - From 1.107 (and all of 2.x) the agent's capabilities are fixed when it is
      built; ``history_processors`` is a read-back copy (1.107) or gone (2.x), and
      appending to ``agent.root_capability.capabilities`` afterwards does
      nothing. Here the processor is passed to each run instead, through
      ``Agent.iter(capabilities=[...])`` (which every ``run*`` method uses), by
      wrapping ``iter`` on this agent instance. If you build the agent yourself
      you can skip this: ``Agent(..., capabilities=[ProcessHistory(
      synap_history_processor())])`` is detected and left as it is, and so is a
      per-run ``agent.run(..., capabilities=[...])``.
    """
    try:
        if _in_constructor_capabilities(agent):
            return True
        if getattr(getattr(agent, "iter", None), _MARK, False):
            return True  # already wrapped
        # Capability-based agents first: from 1.107 the ``history_processors``
        # list is only a read-back copy of what the agent was built with (agent/
        # __init__.py: "stays for 1.x users who read it back"), so appending to it
        # does nothing there. Older 1.x has no ``iter(capabilities=)`` and its
        # list is the real seam.
        if _has_capabilities_seam(agent):
            return _install_per_run(agent)
        # The list is the real seam only on a library whose ``iter`` has no
        # ``capabilities``. Where it has (1.107+), the list is a read-back copy: an
        # append there succeeds and does nothing, so a ``True`` would be false.
        processors: Optional[list] = getattr(agent, "history_processors", None)
        if isinstance(processors, list) and not _library_names_capabilities(type(agent)):
            if not any(_is_ours(p) for p in processors):
                processors.append(synap_history_processor())
            return True
    except Exception:  # noqa: BLE001 — never break the caller's agent set-up
        logger.warning("synap pydantic-ai: could not install the history processor",
                       exc_info=True)
        return False
    global _warned_no_seam
    if not _warned_no_seam:
        _warned_no_seam = True
        logger.warning(
            "synap pydantic-ai: this agent takes neither history_processors nor per-run "
            "capabilities, so Synap's compaction is NOT applied. Pass "
            "synap_history_processor() to your Pydantic AI version's own hook.")
    return False


__all__ = ["synap_history_processor", "install_synap_history"]
