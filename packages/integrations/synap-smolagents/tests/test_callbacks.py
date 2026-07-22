"""Tests for create_synap_recorder — Smolagents step-callback turn recorder."""

from types import SimpleNamespace

import pytest

from synap_smolagents import create_synap_recorder
from synap_smolagents.callbacks import _as_text


def step(model_output, step_number=1, error=None):
    return SimpleNamespace(
        model_output=model_output, step_number=step_number, error=error
    )


# ── validation ───────────────────────────────────────────────────────────────


def test_requires_sdk():
    with pytest.raises(ValueError):
        create_synap_recorder(None, "alice", "conv")


def test_requires_user_id(mock_sdk):
    with pytest.raises(ValueError):
        create_synap_recorder(mock_sdk, "", "conv")


def test_requires_conversation_id(mock_sdk):
    with pytest.raises(ValueError):
        create_synap_recorder(mock_sdk, "alice", "")


# ── recording ────────────────────────────────────────────────────────────────


def test_records_action_step_with_per_step_doc_id(mock_sdk):
    recorder = create_synap_recorder(mock_sdk, "alice", "conv1", customer_id="acme")
    recorder(step("agent did a thing", step_number=2), agent=None)
    mock_sdk.memories.create.assert_called_once()
    kwargs = mock_sdk.memories.create.call_args.kwargs
    assert kwargs["document"] == "agent did a thing"
    assert kwargs["document_id"] == "smolagents-conv1-2"
    assert kwargs["document_type"] == "ai-chat-conversation"
    assert kwargs["user_id"] == "alice"
    assert kwargs["customer_id"] == "acme"
    assert kwargs["metadata"]["step"] == 2


def test_per_step_ids_differ_across_steps(mock_sdk):
    recorder = create_synap_recorder(mock_sdk, "alice", "conv1")
    recorder(step("s1", step_number=1))
    recorder(step("s2", step_number=2))
    ids = [c.kwargs["document_id"] for c in mock_sdk.memories.create.call_args_list]
    assert ids == ["smolagents-conv1-1", "smolagents-conv1-2"]


def test_skips_error_steps(mock_sdk):
    recorder = create_synap_recorder(mock_sdk, "alice", "conv1")
    recorder(step("output", error=ValueError("boom")))
    mock_sdk.memories.create.assert_not_called()


def test_skips_empty_output(mock_sdk):
    recorder = create_synap_recorder(mock_sdk, "alice", "conv1")
    recorder(step(""))
    recorder(step(None))
    recorder(step("   "))
    mock_sdk.memories.create.assert_not_called()


def test_coerces_list_model_output(mock_sdk):
    recorder = create_synap_recorder(mock_sdk, "alice", "conv1")
    recorder(step([{"type": "text", "text": "hello"}, {"text": "world"}]))
    assert mock_sdk.memories.create.call_args.kwargs["document"] == "hello\nworld"


def test_failure_is_logged_not_raised(failing_sdk):
    recorder = create_synap_recorder(failing_sdk, "alice", "conv1")
    # must NOT raise — a raising callback would abort the agent run
    recorder(step("something"))


# ── field-name guard against upstream renames ────────────────────────────────


def test_actionstep_has_expected_fields():
    from smolagents.memory import ActionStep

    fields = set(ActionStep.__dataclass_fields__)
    assert {"model_output", "error", "step_number"} <= fields


def test_as_text_helper():
    assert _as_text(None) == ""
    assert _as_text("hi") == "hi"
    assert _as_text([{"text": "a"}, "b"]) == "a\nb"
