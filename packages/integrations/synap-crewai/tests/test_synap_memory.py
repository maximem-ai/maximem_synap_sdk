"""SynapMemory against REAL CrewAI ``Memory`` objects (only the Synap SDK is mocked).

The original package tests handed the backend a query text by hand, which CrewAI never
does, so nobody noticed recall could not work. These tests go through CrewAI's own
``Memory.remember`` / ``Memory.recall`` / ``arecall``.
"""

import asyncio
import logging
import threading

import pytest
from crewai.memory import Memory

from synap_crewai import HashingEmbedder, SynapMemory, SynapStorageBackend
from synap_crewai.memory import QueryRegistry
from synap_integrations_common import SynapIntegrationError
from synap_integrations_common.testing import make_fact, make_preference, make_unified_response


@pytest.fixture(autouse=True)
def _dummy_llm_key(monkeypatch):
    """Some CrewAI versions build the LLM object eagerly; it is never called here."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")


def _backend(sdk):
    return SynapStorageBackend(sdk=sdk, user_id="alice")


def _memory(sdk, **kw):
    # Explicit llm string: nothing here may reach an LLM.
    return SynapMemory(storage=_backend(sdk), llm="gpt-5.4-mini", **kw)


def _save(memory, text="Isha is vegetarian and eats paneer"):
    return memory.remember(
        text, scope="/diet", categories=["preference"], importance=0.8, source="t"
    )


# --- the premise: stock CrewAI never passes the text --------------------------------


def test_stock_crewai_memory_hands_storage_only_an_embedding(mock_sdk):
    """If CrewAI ever starts passing the text, SynapMemory can be retired."""
    seen = {}

    class Spy(SynapStorageBackend):
        def search(self, query_embedding, scope_prefix=None, categories=None,
                   metadata_filter=None, limit=10, min_score=0.0):
            seen["metadata_filter"] = metadata_filter
            seen["got_embedding"] = bool(query_embedding)
            return []

    memory = Memory(storage=Spy(sdk=mock_sdk, user_id="alice"), embedder=HashingEmbedder(), llm="gpt-5.4-mini")
    memory.recall("What does Isha eat?", depth="shallow")
    assert seen["got_embedding"] is True
    assert not seen["metadata_filter"]  # no text anywhere


def test_stock_memory_with_the_synap_backend_cannot_recall(mock_sdk):
    """This is the reported bug: recall through CrewAI's own Memory always fails."""
    memory = Memory(storage=_backend(mock_sdk), embedder=HashingEmbedder(), llm="gpt-5.4-mini")
    with pytest.raises(SynapIntegrationError, match="query_text missing"):
        memory.recall("What does Isha eat?", depth="shallow")
    mock_sdk.fetch.assert_not_awaited()


# --- recall through SynapMemory ------------------------------------------------------


def test_shallow_recall_sends_the_question_to_synap(mock_sdk):
    mock_sdk.fetch.return_value = make_unified_response(
        facts=[make_fact("f1", "Isha is vegetarian", 0.9)], preferences=[]
    )
    matches = _memory(mock_sdk).recall("What does Isha eat?", depth="shallow")
    assert [m.record.content for m in matches] == ["Isha is vegetarian"]
    kwargs = mock_sdk.fetch.await_args.kwargs
    assert kwargs["search_query"] == ["What does Isha eat?"]
    assert kwargs["user_id"] == "alice"


def test_deep_recall_sends_the_question_to_synap(mock_sdk):
    mock_sdk.fetch.return_value = make_unified_response(
        facts=[make_fact("f1", "Isha is vegetarian", 0.9)], preferences=[]
    )
    matches = _memory(mock_sdk).recall("What does Isha eat?", depth="deep")
    assert [m.record.content for m in matches] == ["Isha is vegetarian"]
    assert all(
        call.kwargs["search_query"] == ["What does Isha eat?"]
        for call in mock_sdk.fetch.await_args_list
    )


def test_preferences_come_back_too(mock_sdk):
    mock_sdk.fetch.return_value = make_unified_response(
        facts=[], preferences=[make_preference("p1", "Likes oat milk", 0.7)]
    )
    matches = _memory(mock_sdk).recall("drink", depth="shallow")
    assert [m.record.content for m in matches] == ["Likes oat milk"]


def test_recall_from_a_background_thread_works(mock_sdk):
    mock_sdk.fetch.return_value = make_unified_response(facts=[make_fact()], preferences=[])
    memory = _memory(mock_sdk)
    out = {}

    def work():
        out["matches"] = memory.recall("anything", depth="shallow")

    t = threading.Thread(target=work)
    t.start()
    t.join(30)
    assert len(out["matches"]) == 1


@pytest.mark.asyncio
async def test_arecall_inside_a_running_loop_works(mock_sdk):
    mock_sdk.fetch.return_value = make_unified_response(facts=[make_fact()], preferences=[])
    matches = await _memory(mock_sdk).arecall("anything", depth="shallow")
    assert len(matches) == 1


def test_recording_stops_after_recall(mock_sdk):
    memory = _memory(mock_sdk)
    memory.recall("secret question", depth="shallow")
    assert memory._synap_recorder._depth == 0
    before = len(memory._synap_recorder.registry)
    _save(memory, "a brand new fact about nothing in particular")
    memory.drain_writes()
    assert len(memory._synap_recorder.registry) == before  # saving records no questions


def test_concurrent_recalls_each_send_their_own_question(mock_sdk):
    memory = _memory(mock_sdk)
    questions = [f"question number {i} about topic {i * 7}" for i in range(8)]
    threads = [threading.Thread(target=memory.recall, args=(q,), kwargs={"depth": "shallow"}) for q in questions]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    asked = sorted(c.kwargs["search_query"][0] for c in mock_sdk.fetch.await_args_list)
    assert asked == sorted(questions)


def test_word_order_changes_the_vector_so_questions_stay_distinct():
    e = HashingEmbedder()
    assert e(["dog bites man"])[0] != e(["man bites dog"])[0]


def test_a_custom_callable_embedder_is_wrapped_and_still_used(mock_sdk):
    calls = []

    def mine(texts):
        calls.append(list(texts))
        return [[float(len(t)), 1.0, 2.0] for t in texts]

    memory = SynapMemory(storage=_backend(mock_sdk), embedder=mine, llm="gpt-5.4-mini")
    memory.recall("hello there", depth="shallow")
    assert calls and mock_sdk.fetch.await_args.kwargs["search_query"] == ["hello there"]


def test_a_provider_config_dict_embedder_is_rejected_with_a_clear_message(mock_sdk):
    with pytest.raises(ValueError, match="callable embedder"):
        SynapMemory(storage=_backend(mock_sdk), embedder={"provider": "openai", "config": {}}, llm="gpt-5.4-mini")


def test_the_registry_is_bounded_and_forgets_the_oldest():
    reg = QueryRegistry(max_entries=3)
    for i in range(5):
        reg.register([float(i), 1.0], f"q{i}")
    assert len(reg) == 3 and reg.lookup([0.0, 1.0]) is None and reg.lookup([4.0, 1.0]) == "q4"


def test_two_questions_in_a_row_each_reach_synap(mock_sdk):
    memory = _memory(mock_sdk)
    memory.recall("first question", depth="shallow")
    memory.recall("second question", depth="shallow")
    asked = [c.kwargs["search_query"] for c in mock_sdk.fetch.await_args_list]
    assert asked == [["first question"], ["second question"]]


# --- saving through SynapMemory ------------------------------------------------------


def test_remember_writes_to_synap_and_does_not_search_or_warn(mock_sdk, caplog):
    memory = _memory(mock_sdk)
    with caplog.at_level(logging.WARNING):
        record = _save(memory)
        memory.drain_writes()
    assert record is not None
    mock_sdk.memories.create.assert_awaited_once()
    assert mock_sdk.memories.create.await_args.kwargs["document"] == "Isha is vegetarian and eats paneer"
    mock_sdk.fetch.assert_not_awaited()  # CrewAI's duplicate search is answered locally
    assert "Storage search failed" not in caplog.text


def test_remember_many_writes_each_distinct_item(mock_sdk):
    memory = _memory(mock_sdk)
    memory.remember_many(
        ["Isha eats paneer", "Ravi flies morning flights", "Meera runs marathons"],
        scope="/facts", categories=["fact"], importance=0.5, source="t",
    )
    memory.drain_writes()
    assert mock_sdk.memories.create.await_count == 3
    mock_sdk.fetch.assert_not_awaited()


def test_remember_then_recall_round_trip_with_the_same_memory(mock_sdk):
    mock_sdk.fetch.return_value = make_unified_response(
        facts=[make_fact("f1", "Isha eats paneer", 0.9)], preferences=[]
    )
    memory = _memory(mock_sdk)
    _save(memory, "Isha eats paneer")
    memory.drain_writes()
    matches = memory.recall("What does Isha eat?", depth="shallow")
    assert matches and matches[0].record.content == "Isha eats paneer"


def test_a_synap_write_failure_still_surfaces(failing_sdk):
    memory = _memory(failing_sdk)
    with pytest.raises(Exception):
        _save(memory)
        memory.drain_writes()


# --- backend contract that must keep working -----------------------------------------


@pytest.mark.asyncio
async def test_direct_backend_use_with_explicit_text_still_works(mock_sdk):
    mock_sdk.fetch.return_value = make_unified_response()
    results = await _backend(mock_sdk).asearch([], metadata_filter={"_query_text": "coffee"})
    assert results and mock_sdk.fetch.await_args.kwargs["search_query"] == ["coffee"]


@pytest.mark.asyncio
async def test_direct_backend_use_without_text_still_fails_loudly(mock_sdk):
    with pytest.raises(SynapIntegrationError, match="SynapMemory"):
        await _backend(mock_sdk).asearch([])


# --- HashingEmbedder -----------------------------------------------------------------


def test_embedder_is_deterministic_normalised_and_non_empty():
    e = HashingEmbedder()
    a, b = e(["Isha eats paneer"])[0], e(["Isha eats paneer"])[0]
    assert a == b and len(a) == 256
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
    assert any(x != 0 for x in e(["!!! ???"])[0])  # no word characters still gives a vector


def test_embedder_puts_similar_texts_closer_than_unrelated_ones():
    e = HashingEmbedder()
    cos = lambda u, v: sum(x * y for x, y in zip(u, v))
    base, near, far = e(["Isha eats paneer for lunch", "Isha eats paneer at lunch", "rocket engine thrust vectoring"])
    assert cos(base, near) > cos(base, far)


def test_identical_items_in_one_batch_are_still_deduplicated_by_crewai(mock_sdk):
    memory = _memory(mock_sdk)
    memory.remember_many(
        ["Isha eats paneer for lunch", "Isha eats paneer for lunch"],
        scope="/facts", categories=["fact"], importance=0.5, source="t",
    )
    memory.drain_writes()
    assert mock_sdk.memories.create.await_count == 1
