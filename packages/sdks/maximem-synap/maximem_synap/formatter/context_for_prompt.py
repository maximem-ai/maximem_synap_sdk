"""Local renderer for compacted + recent-turns context.

Mirrors the server's ``/v1/conversations/{id}/context-for-prompt`` rendering
in structure (sections, headings, turn counts, compaction metadata) but
not necessarily byte-for-byte. Per Q5 of the plan, we accept structural
equivalence rather than chasing the server's exact whitespace/formatting,
to avoid a maintenance trap.

Three styles are supported:
  * ``structured``  – section headers (## Summary, ## Recent Turns, ...)
  * ``narrative``   – prose paragraphs with inline headers
  * ``bullet_points`` – terse bullet lists per section

All three return the same ``ContextForPromptResponse`` shape as the
server endpoint, so callers can swap one for the other without changing
downstream code.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..models.context import (
    ContextForPromptResponse,
    RecentMessage,
    ResponseMetadata,
    CompactionResponse,
)
from ..models.enums import CompactionLevel
from ..cache.short_term_store import CachedShortTermContext


logger = logging.getLogger("synap.sdk.formatter.context_for_prompt")

SUPPORTED_STYLES = ("structured", "narrative", "bullet_points")
DEFAULT_STYLE = "structured"


class LocalContextFormatter:
    """Render a cached ST entry into the same shape as the server endpoints."""

    @staticmethod
    def render_for_prompt(
        entry: CachedShortTermContext,
        style: str = DEFAULT_STYLE,
    ) -> ContextForPromptResponse:
        """Build a ContextForPromptResponse from a cached entry."""
        if style not in SUPPORTED_STYLES:
            style = DEFAULT_STYLE

        recent = _build_recent_messages(entry.recent_turns)
        formatted = _format_block(entry, recent, style)

        available = (
            entry.compaction_id is not None
            or len(entry.recent_turns) > 0
        )

        compaction_age = None
        if entry.compacted_at is not None:
            try:
                compaction_age = int(
                    (datetime.now(timezone.utc) - entry.compacted_at).total_seconds()
                )
            except Exception:
                compaction_age = None

        return ContextForPromptResponse(
            formatted_context=formatted if available else None,
            available=available,
            is_stale=False,
            compression_ratio=None,
            validation_score=None,
            compaction_age_seconds=compaction_age,
            quality_warning=False,
            recent_messages=recent,
            recent_message_count=len(recent),
            compacted_message_count=_count_compacted_messages(entry),
            total_message_count=len(recent) + _count_compacted_messages(entry),
        )

    @staticmethod
    def render_compacted(
        entry: CachedShortTermContext,
        format: str = "structured",
        correlation_id: str = "",
    ) -> Optional[CompactionResponse]:
        """Build a CompactionResponse from a cached entry, or None if the
        entry hasn't received a compaction_update yet.
        """
        if not entry.has_compaction():
            return None

        formatted = _format_block(
            entry,
            _build_recent_messages(entry.recent_turns),
            "structured" if format != "narrative" else "narrative",
        )

        metadata = ResponseMetadata(
            correlation_id=correlation_id or "",
            ttl_seconds=300,
            source="anticipation_cache",
            retrieved_at=datetime.now(timezone.utc),
        )

        # facts / decisions / preferences are dicts in key_extractions on
        # the cached entry. Server wraps them as list-of-dict; passthrough.
        key_ex = entry.key_extractions or {}
        facts = _ensure_list_of_dict(key_ex.get("facts") or key_ex.get("explicit_facts"))
        decisions = _ensure_list_of_dict(key_ex.get("decisions"))
        preferences = _ensure_list_of_dict(
            key_ex.get("preferences") or key_ex.get("user_preferences")
        )

        return CompactionResponse(
            compacted_context=formatted or "",
            original_token_count=0,
            compacted_token_count=0,
            compression_ratio=0.0,
            level_applied=CompactionLevel("adaptive"),
            metadata=metadata,
            compaction_id=entry.compaction_id,
            strategy_used=None,
            validation_score=None,
            validation_passed=None,
            facts=facts,
            decisions=decisions,
            preferences=preferences,
            current_state=entry.current_state or None,
            quality_warning=None,
        )


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------


def _build_recent_messages(raw_turns: List[Dict[str, Any]]) -> List[RecentMessage]:
    out: List[RecentMessage] = []
    for idx, t in enumerate(raw_turns or []):
        ts = t.get("timestamp")
        if isinstance(ts, str):
            try:
                if ts.endswith("Z"):
                    ts = ts[:-1] + "+00:00"
                parsed = datetime.fromisoformat(ts)
            except ValueError:
                parsed = datetime.now(timezone.utc)
        elif isinstance(ts, datetime):
            parsed = ts
        else:
            parsed = datetime.now(timezone.utc)
        out.append(
            RecentMessage(
                role=t.get("role", "user"),
                content=t.get("content", ""),
                timestamp=parsed,
                message_id=t.get("message_id") or f"local-{idx}",
            )
        )
    return out


def _count_compacted_messages(entry: CachedShortTermContext) -> int:
    """Best-effort count of messages covered by the compaction.

    The SDK doesn't actually know how many messages were compacted (the
    server doesn't ship that number in the bundle today), so we report 0
    when unknown. Phase 2 can pipe the count through in
    ``ConversationContextProto`` if the Requests page needs it.
    """
    return 0


def _ensure_list_of_dict(value: Any) -> List[Dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, list):
        out: List[Dict[str, Any]] = []
        for v in value:
            if isinstance(v, dict):
                out.append(v)
            elif isinstance(v, str):
                out.append({"content": v})
        return out
    if isinstance(value, dict):
        return [value]
    return []


def _format_block(
    entry: CachedShortTermContext,
    recent: List[RecentMessage],
    style: str,
) -> str:
    if style == "narrative":
        return _format_narrative(entry, recent)
    if style == "bullet_points":
        return _format_bullets(entry, recent)
    return _format_structured(entry, recent)


def _format_structured(
    entry: CachedShortTermContext,
    recent: List[RecentMessage],
) -> str:
    parts: List[str] = []
    if entry.summary:
        parts.append("## Summary")
        parts.append(entry.summary.strip())

    factual = entry.factual_paragraph
    conversational = entry.conversational_paragraph
    if factual:
        parts.append("## Facts")
        parts.append(factual.strip())
    if conversational:
        parts.append("## Conversation")
        parts.append(conversational.strip())

    cs = entry.current_state or {}
    if cs:
        parts.append("## Current State")
        for k, v in cs.items():
            parts.append(f"- {k}: {_stringify(v)}")

    ke = entry.key_extractions or {}
    if ke:
        parts.append("## Key Extractions")
        for cat, items in ke.items():
            if not items:
                continue
            parts.append(f"### {cat}")
            for it in _ensure_list_of_dict(items):
                parts.append(f"- {it.get('content') or it.get('text') or _stringify(it)}")

    if recent:
        parts.append(f"## Recent Turns ({len(recent)})")
        for m in recent:
            parts.append(f"**{m.role}**: {m.content}")

    return "\n\n".join(p for p in parts if p).strip()


def _format_narrative(
    entry: CachedShortTermContext,
    recent: List[RecentMessage],
) -> str:
    parts: List[str] = []
    if entry.summary:
        parts.append(entry.summary.strip())
    if entry.conversational_paragraph:
        parts.append(entry.conversational_paragraph.strip())
    elif entry.factual_paragraph:
        parts.append(entry.factual_paragraph.strip())

    if recent:
        tail = "\n".join(f"{m.role}: {m.content}" for m in recent)
        parts.append(f"Recent exchanges:\n{tail}")
    return "\n\n".join(p for p in parts if p).strip()


def _format_bullets(
    entry: CachedShortTermContext,
    recent: List[RecentMessage],
) -> str:
    parts: List[str] = []
    if entry.summary:
        parts.append("- Summary: " + entry.summary.strip())

    cs = entry.current_state or {}
    for k, v in cs.items():
        parts.append(f"- {k}: {_stringify(v)}")

    ke = entry.key_extractions or {}
    for cat, items in ke.items():
        for it in _ensure_list_of_dict(items):
            content = it.get("content") or it.get("text") or _stringify(it)
            parts.append(f"- ({cat}) {content}")

    for m in recent:
        parts.append(f"- {m.role}: {m.content}")

    return "\n".join(parts).strip()


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        import json as _json
        return _json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)
