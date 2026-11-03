"""``install_synap_history`` must say True only when the processor will run."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Agent

from synap_pydantic_ai import SynapDeps, install_synap_history, synap_history_processor
from synap_pydantic_ai import history as history_mod

from tests.test_history import Recorder, _deps, _history, make_sdk

capabilities = pytest.importorskip("pydantic_ai.capabilities", reason="pre-capabilities pydantic-ai")
ProcessHistory = getattr(capabilities, "ProcessHistory", None)
pytestmark = pytest.mark.skipif(ProcessHistory is None, reason="no ProcessHistory capability")


def _fetches(sdk):
    return sdk.conversation.context.get_context_for_prompt.await_count


async def test_a_late_install_really_runs_on_a_plain_run():
    sdk = make_sdk()
    agent = Agent(Recorder().model(), deps_type=SynapDeps)
    assert install_synap_history(agent) is True
    await agent.run("hi", deps=_deps(sdk), message_history=_history())
    assert _fetches(sdk) == 1


async def test_it_also_runs_on_run_sync_and_run_stream():
    sdk = make_sdk()
    agent = Agent(Recorder().model(), deps_type=SynapDeps)
    install_synap_history(agent)
    async with agent.run_stream("hi", deps=_deps(sdk), message_history=_history()) as r:
        await r.get_output()
    assert _fetches(sdk) == 1


async def test_an_agent_built_with_the_capability_is_left_alone_and_runs_once():
    sdk = make_sdk()
    agent = Agent(Recorder().model(), deps_type=SynapDeps,
                  capabilities=[ProcessHistory(synap_history_processor())])
    assert install_synap_history(agent) is True
    assert not getattr(agent.iter, "_synap_history_processor", False)  # not wrapped
    await agent.run("hi", deps=_deps(sdk), message_history=_history())
    assert _fetches(sdk) == 1


async def test_a_per_call_capability_is_not_doubled():
    sdk = make_sdk()
    agent = Agent(Recorder().model(), deps_type=SynapDeps)
    install_synap_history(agent)
    await agent.run("hi", deps=_deps(sdk), message_history=_history(),
                    capabilities=[ProcessHistory(synap_history_processor())])
    assert _fetches(sdk) == 1


async def test_other_per_call_capabilities_are_kept():
    seen = []

    class Mark(capabilities.AbstractCapability):
        async def before_model_request(self, ctx, request_context):
            seen.append(True)
            return request_context

    sdk = make_sdk()
    agent = Agent(Recorder().model(), deps_type=SynapDeps)
    install_synap_history(agent)
    await agent.run("hi", deps=_deps(sdk), message_history=_history(), capabilities=[Mark()])
    assert seen and _fetches(sdk) == 1


def test_it_says_false_and_warns_when_there_is_no_seam(caplog, monkeypatch):
    monkeypatch.setattr(history_mod, "_warned_no_seam", False)

    class Bare:  # no iter(capabilities=), no history_processors list
        def iter(self, *a, **k): ...

    with caplog.at_level(logging.WARNING):
        assert install_synap_history(Bare()) is False
    assert "NOT applied" in caplog.text


# ── an agent copied after the install runs on its own model ─────────────────


async def test_a_copy_made_after_install_runs_on_its_own_model():
    import copy
    sdk = make_sdk()
    r1, r2 = Recorder(), Recorder()
    agent = Agent(r1.model(), deps_type=SynapDeps)
    install_synap_history(agent)
    clone = copy.copy(agent)
    clone.model = r2.model()
    await clone.run("hi", deps=_deps(sdk), message_history=_history())
    assert len(r2.calls) == 1 and len(r1.calls) == 0
    assert _fetches(sdk) == 1
    assert BLOCK_IN(r2)
    await agent.run("hi", deps=_deps(sdk), message_history=_history())
    assert len(r1.calls) == 1 and len(r2.calls) == 1


def BLOCK_IN(recorder):
    from tests.test_history import BLOCK
    return BLOCK in recorder.seen


async def test_a_second_install_returns_true_and_does_not_wrap_again():
    sdk = make_sdk()
    rec = Recorder()
    agent = Agent(rec.model(), deps_type=SynapDeps)
    assert install_synap_history(agent) is True
    first = type(agent)
    assert install_synap_history(agent) is True   # not False + a warning
    assert type(agent) is first and "iter" not in agent.__dict__
    await agent.run("hi", deps=_deps(sdk), message_history=_history())
    assert _fetches(sdk) == 1                      # one processor, one fetch


async def test_a_second_install_on_an_instance_wrapped_agent_returns_true_and_does_not_wrap_again():
    sdk = make_sdk()
    agent = Agent(Recorder().model(), deps_type=SynapDeps)
    agent.iter = _user_wrapper(agent.iter)         # forces the instance-wrapper path
    assert install_synap_history(agent) is True
    wrapped = agent.iter
    assert install_synap_history(agent) is True
    assert agent.iter is wrapped
    await agent.run("hi", deps=_deps(sdk), message_history=_history())
    assert _fetches(sdk) == 1


def _user_wrapper(inner):
    """A user's own (*args, **kwargs) wrapper around iter, e.g. for tracing."""
    def traced(*args, **kwargs):
        return inner(*args, **kwargs)
    return traced


async def test_a_user_wrapper_with_args_kwargs_is_a_seam():
    sdk = make_sdk()
    rec = Recorder()
    agent = Agent(rec.model(), deps_type=SynapDeps)
    seen = []
    inner = agent.iter

    def traced(*args, **kwargs):
        seen.append(True)
        return inner(*args, **kwargs)

    agent.iter = traced
    assert install_synap_history(agent) is True
    await agent.run("hi", deps=_deps(sdk), message_history=_history())
    assert seen and _fetches(sdk) == 1 and BLOCK_IN(rec)


def test_a_kwargs_wrapper_over_an_agent_with_no_seam_is_still_false(caplog, monkeypatch):
    monkeypatch.setattr(history_mod, "_warned_no_seam", False)

    class Old:
        def iter(self, user_prompt=None, *, deps=None): ...

    o = Old()
    inner = o.iter
    o.iter = lambda *a, **k: inner(*a, **k)       # forwards capabilities into a TypeError
    with caplog.at_level(logging.WARNING):
        assert install_synap_history(o) is False


# ── a user subclass that overrides iter(self, *a, **k) ──────────────────────


async def test_a_user_subclass_overriding_iter_with_kwargs_is_a_seam():
    seen = []

    class Traced(Agent):
        def iter(self, *a, **k):
            seen.append(True)
            return super().iter(*a, **k)

    sdk = make_sdk()
    rec = Recorder()
    agent = Traced(rec.model(), deps_type=SynapDeps)
    assert install_synap_history(agent) is True
    await agent.run("hi", deps=_deps(sdk), message_history=_history())
    assert seen and _fetches(sdk) == 1 and BLOCK_IN(rec)   # a true True: the fetch happens
    assert isinstance(agent, Traced)


def test_a_subclass_override_that_cannot_pass_capabilities_is_false(caplog, monkeypatch):
    monkeypatch.setattr(history_mod, "_warned_no_seam", False)

    class Strict(Agent):
        def iter(self, user_prompt=None, *, deps=None):  # drops everything else
            return super().iter(user_prompt, deps=deps)

    agent = Strict(Recorder().model(), deps_type=SynapDeps)
    with caplog.at_level(logging.WARNING):
        assert install_synap_history(agent) is False
    assert "NOT applied" in caplog.text


# ── the cached subclass ─────────────────────────────────────────────────────


def test_two_agents_of_one_class_share_one_cached_subclass():
    a = Agent(Recorder().model(), deps_type=SynapDeps)
    b = Agent(Recorder().model(), deps_type=SynapDeps)
    install_synap_history(a)
    install_synap_history(b)
    assert type(a) is type(b) and type(a) is not Agent
    assert Agent(Recorder().model()).__class__ is Agent   # a plain agent is untouched


def test_the_cache_holds_classes_weakly():
    import gc
    import weakref

    class Mine(Agent):
        pass

    agent = Mine(Recorder().model(), deps_type=SynapDeps)
    install_synap_history(agent)
    ref = weakref.ref(Mine)
    del agent, Mine
    gc.collect()
    assert ref() is None


def test_an_installed_agent_has_a_class_name_of_its_own_so_pickling_says_why():
    import pickle
    agent = Agent(Recorder().model(), deps_type=SynapDeps)
    install_synap_history(agent)
    assert type(agent).__qualname__.endswith("WithSynapHistory")
    with pytest.raises(Exception) as err:
        pickle.dumps(agent)
    assert "WithSynapHistory" in str(err.value) or "pickle" in str(err.value).lower()
