

def test_the_agent_model_is_named_for_compaction_sizing():
    from types import SimpleNamespace
    from synap_openai_agents.hooks import _model_metadata

    assert _model_metadata("gpt-4.1") == {"agent_model": "gpt-4.1"}
    assert _model_metadata(SimpleNamespace(model="o3")) == {"agent_model": "o3"}
    assert _model_metadata(None) is None
