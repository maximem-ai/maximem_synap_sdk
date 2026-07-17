"""Response envelope parser and error mapper."""

import logging
from datetime import datetime, timezone
from typing import Dict, Any

from ..models.context import ContextResponse, ResponseMetadata
from ..models.errors import (
    SynapError,
    InvalidInputError,
    AuthenticationError,
    ContextNotFoundError,
    RateLimitError,
    ServiceUnavailableError,
)

logger = logging.getLogger("synap.sdk.orchestration")


class ResponseHandler:
    """Parse responses and map errors to SDK exceptions."""

    def parse_response(
        self,
        raw_response: Dict[str, Any],
        correlation_id: str
    ) -> ContextResponse:
        """Parse raw response into context response.

        Args:
            raw_response: Raw response data
            correlation_id: Expected correlation ID

        Returns:
            Parsed context response

        Raises:
            SynapError: If response indicates error
        """
        if "error" in raw_response or raw_response.get("success") is False:
            raise self.handle_error_response(raw_response)

        metadata = ResponseMetadata(
            correlation_id=correlation_id,
            ttl_seconds=raw_response.get("ttl_seconds", 300),
            source=raw_response.get("source", "cloud"),
            retrieved_at=datetime.now(timezone.utc),
        )

        context_data = raw_response.get("context", raw_response)
        return ContextResponse.from_cloud_response(context_data, metadata)

    def handle_error_response(
        self,
        raw_response: Dict[str, Any]
    ) -> SynapError:
        """Map error response to appropriate SDK exception.

        Args:
            raw_response: Raw error response

        Returns:
            Appropriate SDK exception
        """
        error_code = raw_response.get(
            "error_code", raw_response.get("status_code", 500)
        )
        message = raw_response.get(
            "message", raw_response.get("detail", "Unknown error")
        )
        correlation_id = raw_response.get("correlation_id")

        try:
            status = int(error_code)
        except (ValueError, TypeError):
            status = 500

        if status == 400:
            return InvalidInputError(message, correlation_id=correlation_id)
        elif status == 401:
            return AuthenticationError(message, correlation_id=correlation_id)
        elif status == 404:
            return ContextNotFoundError(message, correlation_id=correlation_id)
        elif status == 429:
            retry_after = raw_response.get("retry_after_seconds")
            return RateLimitError(
                message,
                retry_after_seconds=retry_after,
                correlation_id=correlation_id,
            )
        elif status >= 500:
            return ServiceUnavailableError(
                message, correlation_id=correlation_id
            )
        else:
            return SynapError(message, correlation_id=correlation_id)

    # ------------------------------------------------------------------
    # Endpoint-specific parsers (used by facade controllers)
    # ------------------------------------------------------------------

    def parse_context_response(
        self,
        raw_response: Dict[str, Any],
    ) -> ContextResponse:
        """Parse context fetch response from cloud.

        Args:
            raw_response: Raw JSON response dict

        Returns:
            Parsed ContextResponse
        """
        if "error" in raw_response or raw_response.get("success") is False:
            raise self.handle_error_response(raw_response)

        correlation_id = raw_response.get("correlation_id", "unknown")
        metadata = ResponseMetadata(
            correlation_id=correlation_id,
            ttl_seconds=raw_response.get("ttl_seconds", 300),
            source=raw_response.get("source", "cloud"),
            retrieved_at=datetime.now(timezone.utc),
        )

        context_data = raw_response.get("context", raw_response)
        # Carry the C4 conversation-summary siblings (profile / conversations)
        # through this second parse path too — they live beside "context" in
        # the response envelope, not inside it. Facade parity with sdk.py's
        # UserContextInterface.fetch.
        if isinstance(context_data, dict) and context_data is not raw_response:
            for _summary_key in ("profile", "conversations", "conversation_context"):
                if _summary_key in raw_response and _summary_key not in context_data:
                    context_data[_summary_key] = raw_response[_summary_key]
        return ContextResponse.from_cloud_response(context_data, metadata)

    def parse_compaction_response(
        self,
        raw_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Parse async compaction trigger response from POST /v1/conversations/compact.

        The compact endpoint returns immediately with an in-progress status.
        Use parse_compacted_context_response() for the completed result
        from GET /v1/conversations/{id}/compacted.

        Args:
            raw_response: Raw JSON response dict

        Returns:
            Dict with compaction_id, conversation_id, status, trigger_type, etc.
        """
        if "error" in raw_response:
            raise self.handle_error_response(raw_response)

        return {
            "compaction_id": raw_response.get("compaction_id"),
            "conversation_id": raw_response.get("conversation_id"),
            "status": raw_response.get("status"),
            "trigger_type": raw_response.get("trigger_type"),
            "initiated_at": raw_response.get("initiated_at"),
            "estimated_completion_seconds": raw_response.get("estimated_completion_seconds"),
        }

    def parse_compacted_context_response(
        self,
        raw_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Parse GET /v1/conversations/{id}/compacted response.

        Args:
            raw_response: Raw JSON response dict

        Returns:
            Dict with formatted_context, facts, preferences, decisions, etc.
        """
        if "error" in raw_response:
            raise self.handle_error_response(raw_response)

        return {
            "compaction_id": raw_response.get("compaction_id"),
            "conversation_id": raw_response.get("conversation_id"),
            "version": raw_response.get("version"),
            "created_at": raw_response.get("created_at"),
            "compression_ratio": raw_response.get("compression_ratio"),
            "original_token_count": raw_response.get("original_token_count"),
            "compacted_token_count": raw_response.get("compacted_token_count"),
            "formatted_context": raw_response.get("formatted_context"),
            "facts": raw_response.get("facts", []),
            "preferences": raw_response.get("preferences", []),
            "decisions": raw_response.get("decisions", []),
            "current_state": raw_response.get("current_state", {}),
            "validation_score": raw_response.get("validation_score"),
            "quality_warning": raw_response.get("quality_warning", False),
            "warnings": raw_response.get("warnings", []),
        }

    def parse_compaction_status_response(
        self,
        raw_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Parse GET /v1/conversations/{id}/compaction/status response.

        Args:
            raw_response: Raw JSON response dict

        Returns:
            Dict with conversation_id, status, and completion details
        """
        if "error" in raw_response:
            raise self.handle_error_response(raw_response)

        return {
            "conversation_id": raw_response.get("conversation_id"),
            "status": raw_response.get("status"),
            "compaction_id": raw_response.get("compaction_id"),
            "completed_at": raw_response.get("completed_at"),
            "compression_ratio": raw_response.get("compression_ratio"),
            "validation_score": raw_response.get("validation_score"),
            "estimated_completion_seconds": raw_response.get("estimated_completion_seconds"),
            "error_message": raw_response.get("error_message"),
            "latest_version": raw_response.get("latest_version"),
            "latest_created_at": raw_response.get("latest_created_at"),
        }

    def parse_record_message_response(
        self,
        raw_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Parse POST /v1/conversations/messages response.

        Args:
            raw_response: Raw JSON response dict

        Returns:
            Dict with message_id, conversation_id, session_id, recorded_at
        """
        if "error" in raw_response:
            raise self.handle_error_response(raw_response)

        return {
            "message_id": raw_response.get("message_id"),
            "conversation_id": raw_response.get("conversation_id"),
            "session_id": raw_response.get("session_id"),
            "recorded_at": raw_response.get("recorded_at"),
        }

    def parse_record_batch_response(
        self,
        raw_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Parse POST /v1/conversations/messages/batch response.

        Args:
            raw_response: Raw JSON response dict

        Returns:
            Dict with total, succeeded, failed, results
        """
        if "error" in raw_response:
            raise self.handle_error_response(raw_response)

        return {
            "total": raw_response.get("total"),
            "succeeded": raw_response.get("succeeded"),
            "failed": raw_response.get("failed"),
            "results": raw_response.get("results", []),
        }
