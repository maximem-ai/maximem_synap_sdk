"""Synap SDK Exception Hierarchy.

Exception design:
- SynapError: Base for all SDK errors
- SynapTransientError: Retryable errors (network, rate limits)
- SynapPermanentError: Non-retryable errors (auth, invalid input)
"""

from typing import Optional


class SynapError(Exception):
    """Base exception for all Synap SDK errors."""

    def __init__(self, message: str, correlation_id: Optional[str] = None):
        super().__init__(message)
        self.correlation_id = correlation_id


class SynapTransientError(SynapError):
    """Transient error - retry may succeed."""

    pass


class SynapPermanentError(SynapError):
    """Permanent error - retry will not help."""

    pass


# Transient errors
class NetworkTimeoutError(SynapTransientError):
    """Network operation timed out."""

    pass


class RateLimitError(SynapTransientError):
    """Rate limit exceeded. Includes retry_after if available."""

    def __init__(
        self,
        message: str,
        retry_after_seconds: Optional[int] = None,
        correlation_id: Optional[str] = None,
    ):
        super().__init__(message, correlation_id=correlation_id)
        self.retry_after_seconds = retry_after_seconds


class InsufficientCreditsError(SynapPermanentError):
    """The caller's credit wallet cannot cover this request.

    Raised for HTTP 402 (from :mod:`synap.cloud.application.credits.enforcement`)
    and gRPC ``RESOURCE_EXHAUSTED`` with credit-related trailing metadata.
    Callers should redeem a code, request more credits, or contact support.

    Attributes:
        balance_credits: Snapshot balance at the time of rejection.
        minimum_required_credits: What the endpoint's minimum charge was.
        recovery_url: Where the balance can be viewed (defaults to /v1/credits/balance).
        redeem_url: Where the customer can enter a redeem code.
    """

    def __init__(
        self,
        message: str,
        balance_credits: Optional[float] = None,
        minimum_required_credits: Optional[float] = None,
        recovery_url: Optional[str] = None,
        redeem_url: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        super().__init__(message, correlation_id=correlation_id)
        self.balance_credits = balance_credits
        self.minimum_required_credits = minimum_required_credits
        self.recovery_url = recovery_url
        self.redeem_url = redeem_url


class ServiceUnavailableError(SynapTransientError):
    """Synap service temporarily unavailable."""

    pass


# Permanent errors
class InvalidInputError(SynapPermanentError):
    """Invalid input provided to SDK method."""

    pass


class InvalidInstanceIdError(InvalidInputError):
    """Invalid instance ID format."""

    def __init__(self, instance_id: str, correlation_id: Optional[str] = None):
        super().__init__(
            f"Invalid instance ID: {instance_id}", correlation_id=correlation_id
        )


class InvalidConversationIdError(InvalidInputError):
    """Invalid conversation ID format."""

    def __init__(self, conversation_id: str, correlation_id: Optional[str] = None):
        super().__init__(
            f"Invalid conversation ID: {conversation_id}", correlation_id=correlation_id
        )


class AuthenticationError(SynapPermanentError):
    """Authentication failed - credentials invalid or expired."""

    pass


class ContextNotFoundError(SynapPermanentError):
    """Requested context does not exist."""

    pass


class SessionExpiredError(SynapPermanentError):
    """Session has expired and cannot be resumed."""

    pass


class ListeningAlreadyActiveError(SynapPermanentError):
    """Listening stream is already active."""

    def __init__(
        self,
        message: str = "Listening already active",
        correlation_id: Optional[str] = None,
    ):
        super().__init__(message, correlation_id=correlation_id)


class ListeningNotActiveError(SynapPermanentError):
    """No active listening stream — must call listen() first."""

    def __init__(
        self,
        message: str = "No active listening stream. Call listen() first.",
        correlation_id: Optional[str] = None,
    ):
        super().__init__(message, correlation_id=correlation_id)


class AgentUnavailableError(SynapTransientError):
    """Agent is unavailable."""

    def __init__(
        self, message: str = "Agent unavailable", correlation_id: Optional[str] = None
    ):
        super().__init__(message, correlation_id=correlation_id)


# Backward compatibility aliases
SDKError = SynapError
TransientError = SynapTransientError
PermanentError = SynapPermanentError
ConnectionError = NetworkTimeoutError
