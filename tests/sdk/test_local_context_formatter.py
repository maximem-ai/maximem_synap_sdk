"""Unit tests for LocalContextFormatter."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from maximem_synap.cache.short_term_store import (
    CachedShortTermContext,
    ShortTermContextStore,
)
from maximem_synap.formatter.context_for_prompt import LocalContextFormatter


def _entry_with_compaction() -> CachedShortTermContext:
    return CachedShortTermContext(
        conversation_id="conv-1",
        summary="Aria and Bo discussed the migration plan.",
        factual_paragraph="Migration is scheduled for Tuesday.",
        conversational_paragraph="Aria walked Bo through the rollback steps.",
        current_state={"status": "active", "next_steps": "approve plan"},
        key_extractions={
            "facts": [{"content": "Migration on Tuesday"}],
            "decisions": [{"content": "Roll back if errors > 1%"}],
        },
        compaction_id="comp-1",
        compacted_at=datetime(2026, 5, 22, 10, 0, 0, tzinfo=timezone.utc),
        end_timestamp=datetime(2026, 5, 22, 10, 0, 0, tzinfo=timezone.utc),
        recent_turns=[
            {
                "role": "user",
                "content": "any update?",
                "timestamp": "2026-05-22T11:00:00+00:00",
            },
        ],
    )


class TestRenderForPrompt:
    def test_structured_includes_all_sections(self):
        entry = _entry_with_compaction()
        resp = LocalContextFormatter.render_for_prompt(entry, style="structured")
        assert resp.available is True
        assert resp.formatted_context is not None
        body = resp.formatted_context
        assert "## Summary" in body
        assert "## Facts" in body
        assert "## Conversation" in body
        assert "## Current State" in body
        assert "## Key Extractions" in body
        assert "## Recent Turns" in body
        assert "any update?" in body

    def test_narrative_omits_section_headers(self):
        entry = _entry_with_compaction()
        resp = LocalContextFormatter.render_for_prompt(entry, style="narrative")
        body = resp.formatted_context
        assert body is not None
        assert "##" not in body  # no markdown headers
        assert "migration plan" in body.lower()
        assert "any update?" in body

    def test_bullets_uses_dashes(self):
        entry = _entry_with_compaction()
        resp = LocalContextFormatter.render_for_prompt(entry, style="bullet_points")
        body = resp.formatted_context
        assert body is not None
        # Multiple bullet lines
        bullet_lines = [l for l in body.splitlines() if l.startswith("- ")]
        assert len(bullet_lines) >= 3

    def test_unknown_style_falls_back_to_structured(self):
        entry = _entry_with_compaction()
        resp = LocalContextFormatter.render_for_prompt(entry, style="nope")
        body = resp.formatted_context
        assert body is not None
        assert "## Summary" in body

    def test_recent_messages_round_trip(self):
        entry = _entry_with_compaction()
        resp = LocalContextFormatter.render_for_prompt(entry)
        assert resp.recent_message_count == 1
        assert resp.recent_messages[0].content == "any update?"
        assert resp.recent_messages[0].role == "user"
        assert resp.recent_messages[0].timestamp.tzinfo is not None

    def test_no_data_yields_unavailable(self):
        entry = CachedShortTermContext(conversation_id="conv-empty")
        resp = LocalContextFormatter.render_for_prompt(entry)
        assert resp.available is False
        assert resp.formatted_context is None
        assert resp.recent_message_count == 0

    def test_only_recent_turns_is_available(self):
        store = ShortTermContextStore()
        store.append_turn("c1", "user", "hi")
        entry = store.get("c1")
        resp = LocalContextFormatter.render_for_prompt(entry)
        assert resp.available is True
        assert "hi" in resp.formatted_context


class TestRenderCompacted:
    def test_returns_none_when_no_compaction(self):
        entry = CachedShortTermContext(conversation_id="conv-empty")
        assert LocalContextFormatter.render_compacted(entry) is None

    def test_returns_compaction_response(self):
        entry = _entry_with_compaction()
        resp = LocalContextFormatter.render_compacted(entry, correlation_id="abc")
        assert resp is not None
        assert resp.compaction_id == "comp-1"
        assert "Migration on Tuesday" in (resp.compacted_context or "")
        assert resp.metadata.correlation_id == "abc"
        assert resp.metadata.source == "anticipation_cache"
        # facts/decisions/preferences passthrough
        assert any("Migration" in f.get("content", "") for f in resp.facts)
        assert any("rollback" in d.get("content", "").lower() or "Roll back" in d.get("content", "") for d in resp.decisions)

    def test_handles_string_only_key_extractions(self):
        entry = CachedShortTermContext(
            conversation_id="c",
            summary="s",
            compaction_id="comp",
            key_extractions={"facts": ["plain string fact"]},
        )
        resp = LocalContextFormatter.render_compacted(entry)
        assert resp is not None
        assert resp.facts == [{"content": "plain string fact"}]
