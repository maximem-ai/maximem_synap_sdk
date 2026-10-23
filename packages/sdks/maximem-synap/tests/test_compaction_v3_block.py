"""Compaction v3 in the SDK: the server's framed block is printed once, and
turns carry the agent's model. JS twin: js/src/__tests__/compaction-v3-block.test.ts.
"""
from maximem_synap.agent_model import with_agent_model
from maximem_synap.cache.short_term_store import ShortTermContextStore
from maximem_synap.formatter.context_for_prompt import LocalContextFormatter
from maximem_synap.models.config import SDKConfig

BLOCK = (
    "## Conversation so far (compacted: turns 1–2, 2026-10-17 09:00–09:01 UTC)\n"
    "[CONTEXT COMPACTION, REFERENCE ONLY] A record of what was said earlier.\n\n"
    "### Customer and order\nPriya reported order ORD-1 arrived broken."
)


def _bundle(**cc):
    return {"_bundle_type": "compaction_update",
            "conversation_context": {"conversation_id": "c1", "compaction_id": cc.pop("compaction_id", "k1"),
                                     "compacted_at": cc.pop("compacted_at", "2026-10-17T09:05:00+00:00"), **cc}}


def _store_with_v2_then_v3():
    store = ShortTermContextStore()
    store.apply_compaction(_bundle(summary="old", factual_paragraph="OLD FACTS", conversational_paragraph="OLD TALK",
                                   compacted_at="2026-10-17T09:00:00+00:00"))
    store.apply_compaction(_bundle(summary=BLOCK, context_block=BLOCK, factual_paragraph="",
                                   conversational_paragraph="", compaction_id="k2"))
    return store


def test_a_v3_compaction_clears_cached_v2_paragraphs():
    entry = _store_with_v2_then_v3().get("c1")
    assert entry.context_block == BLOCK
    assert entry.factual_paragraph is None and entry.conversational_paragraph is None


def test_every_style_prints_the_block_once():
    entry = _store_with_v2_then_v3().get("c1")
    for style in ("structured", "narrative", "bullet_points"):
        text = LocalContextFormatter.render_for_prompt(entry, style=style).formatted_context
        assert text.count("## Conversation so far") == 1, style
        assert "OLD FACTS" not in text and "## Summary" not in text and "- Summary:" not in text, style


def test_a_summary_that_is_the_block_is_recognised():
    """REST-fed entries, and pushes decoded by stubs that predate context_block."""
    store = ShortTermContextStore()
    store.apply_compaction(_bundle(summary=BLOCK))
    assert store.get("c1").context_block == BLOCK


def test_a_v2_compaction_renders_as_before():
    store = ShortTermContextStore()
    store.apply_compaction(_bundle(summary="S", factual_paragraph="F"))
    text = LocalContextFormatter.render_for_prompt(store.get("c1")).formatted_context
    assert "## Summary\n\nS" in text and "## Facts\n\nF" in text


def test_agent_model_precedence():
    config = SDKConfig(agent_model="gpt-4o", agent_context_window=128000)
    assert with_agent_model(None, config=SDKConfig()) is None
    assert with_agent_model({"a": 1}, config=config) == {"a": 1, "agent_model": "gpt-4o", "agent_context_window": 128000}
    assert with_agent_model(None, model="gpt-5", config=config)["agent_model"] == "gpt-5"
    assert with_agent_model(None, context_window=400000, config=config, as_strings=True) == {
        "agent_model": "gpt-4o", "agent_context_window": "400000"}


def test_the_response_carries_the_block_alone_for_history_stores():
    """Integrations that keep their own message list put this in a system message."""
    entry = _store_with_v2_then_v3().get("c1")
    response = LocalContextFormatter.render_for_prompt(entry)
    assert response.context_block == BLOCK and response.compaction_id == "k2"
