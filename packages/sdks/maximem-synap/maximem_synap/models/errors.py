"""Synap SDK Exception Hierarchy.

Exception design:
- SynapError: Base for all SDK errors
- SynapTransientError: Retryable errors (network, rate limits)
- SynapPermanentError: Non-retryable errors (auth, invalid input)
"""

from typing import Optional


class SynapError(Exception):
    """Base exception for all Synap SDK errors."""

    #: The HTTP status the server answered with, when the error came from an
    #: HTTP response. None for errors raised before or without a response.
    status_code: Optional[int] = None
    #: The server's machine-readable error code (``file_too_large``,
    #: ``invalid_utility``, ...), when the response carried one.
    code: Optional[str] = None

    def __init__(self, message: str, correlation_id: Optional[str] = None):
        super().__init__(message)
        self.correlation_id = correlation_id


class SynapTransientError(SynapError):
    """Transient error - retry may succeed."""

    #: True when it is genuinely unknown whether the server processed the request (a gateway 504,
    #: a read timeout). A caller must not retry a non-idempotent call after such an error: it could
    #: be stored and billed twice. The transport sets it; JS decides the same in
    #: ``isOutcomeUnknown`` (transport/http.ts) rather than on the error object.
    outcome_unknown: bool = False


class SynapPermanentError(SynapError):
    """Permanent error - retry will not help."""

    pass


# Transient errors
class NetworkTimeoutError(SynapTransientError):
    """Network operation timed out."""

    pass


class RateLimitError(SynapTransientError):
    """Rate limit exceeded. Includes retry_after if available.

    Also carries the two credit stops the server reports as HTTP 429 (and
    gRPC ``RESOURCE_EXHAUSTED``), because both recover with time or a plan
    change rather than with a retry of this request:

    - ``reason="trial_limit_reached"``: the Trial cap is spent. ``upgrade_url``
      points at the upgrade page; the next cycle also clears it.
    - ``reason="subscription_inactive"``: a paid plan whose subscription is
      past due, unpaid or cancelled. ``manage_url`` points at Plan & billing.

    Attributes:
        retry_after_seconds: Server's Retry-After, when it sent one.
        reason: Machine-readable cause, when the stop was a credit stop.
        upgrade_url: Where a Trial customer upgrades.
        manage_url: Where billing is managed.
    """

    def __init__(
        self,
        message: str,
        retry_after_seconds: Optional[int] = None,
        correlation_id: Optional[str] = None,
        *,
        reason: Optional[str] = None,
        upgrade_url: Optional[str] = None,
        manage_url: Optional[str] = None,
    ):
        super().__init__(message, correlation_id=correlation_id)
        self.retry_after_seconds = retry_after_seconds
        self.reason = reason
        self.upgrade_url = upgrade_url
        self.manage_url = manage_url


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
        reason: Machine-readable cause. ``"overages_disabled"`` means the plan
            is paid and the balance is spent, but the account has not allowed
            usage past zero, so the work stops until someone allows overages
            or adds credits. ``None`` when an older server did not send one.
        manage_url: Where that setting and billing are managed, when the
            server sent one.
    """

    def __init__(
        self,
        message: str,
        balance_credits: Optional[float] = None,
        minimum_required_credits: Optional[float] = None,
        recovery_url: Optional[str] = None,
        redeem_url: Optional[str] = None,
        correlation_id: Optional[str] = None,
        *,
        reason: Optional[str] = None,
        manage_url: Optional[str] = None,
    ):
        super().__init__(message, correlation_id=correlation_id)
        self.balance_credits = balance_credits
        self.minimum_required_credits = minimum_required_credits
        self.recovery_url = recovery_url
        self.redeem_url = redeem_url
        self.reason = reason
        self.manage_url = manage_url


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


class ConflictError(SynapPermanentError):
    """The request conflicts with the current state of the resource (HTTP 409).

    A 409 is **permanent** — retrying the identical request will keep
    conflicting. This is why the transport does not retry it (unlike a
    transient error). See :class:`TranscriptConflictError` for the specific,
    discriminated transcript-immutability conflict.

    Note: :meth:`sdk.conversation.compact`'s "Compaction already in progress"
    409 now surfaces here as an immediate permanent error rather than being
    retried as a transient one — an intentional semantic change (SDK 0.4.0).
    """

    pass


class TranscriptConflictError(ConflictError):
    """A transcript push conflicts with an already-ingested call (HTTP 409).

    Raised when ``ingest_transcript`` is called with a ``conversation_id`` that
    was previously ingested with a *different* transcript. A call's transcript
    is immutable — mint a new ``conversation_id`` for a new call. The server
    discriminates this from a generic conflict with a structured body
    ``{"detail": {"code": "transcript_conflict", ...}}``.

    Subclasses :class:`ConflictError`, so ``except ConflictError`` catches both
    the generic and the transcript-specific conflict.
    """

    pass


class InstanceNotReadyError(ConflictError):
    """The instance's memory architecture is still being built (HTTP 409).

    A new instance usually spends 15 to 20 minutes building its memory architecture.
    Until that finishes the server refuses memory writes and reads with
    ``409 {"error": "instance_not_ready", ...}`` and a ``Retry-After`` header.
    Nothing was recorded; the same call works once the build is done.

    Subclasses :class:`ConflictError`, so existing ``except ConflictError``
    code keeps catching it. It is raised immediately and is **never retried
    inside the SDK**: a wait of this length must not hide inside a call. Catch
    it and wait ``retry_after_seconds`` yourself, or retry later.

    Attributes:
        retry_after_seconds: Server's estimate of the wait, from the body or the
            ``Retry-After`` header. ``None`` when the server sent neither.
        instance_id: The instance that is building.
        state: Build state reported by the server (``"building"``).
        from_cooldown: ``True`` when the SDK raised this itself, without a
            network call, because an earlier call to the same instance was
            refused and the server's wait has not elapsed yet. In that case
            ``retry_after_seconds`` is the REMAINING wait (rounded up, at
            least 1), not the server's original figure. ``False`` when the
            server sent this refusal just now.
    """

    def __init__(
        self,
        message: str = "This instance's memory architecture is still being built.",
        retry_after_seconds: Optional[int] = None,
        correlation_id: Optional[str] = None,
        *,
        instance_id: Optional[str] = None,
        state: Optional[str] = "building",
        from_cooldown: bool = False,
    ):
        super().__init__(message, correlation_id=correlation_id)
        self.retry_after_seconds = retry_after_seconds
        self.instance_id = instance_id
        self.state = state
        self.from_cooldown = from_cooldown


class InstanceBuildFailedError(ConflictError):
    """The instance's memory architecture build failed for good (HTTP 409).

    The server tried the build several times and gave up, so memory writes and
    reads are refused with ``409 {"error": "instance_build_failed", ...}``.
    Waiting does not help: a person has to start a rebuild (for example
    ``maximem-synap instances rebuild <instance_id>``, see ``hint``).

    Subclasses :class:`ConflictError`. It is raised immediately, is **never
    retried** and sets **no cooldown**: it is not time based, so every call is
    sent and the first one after a successful rebuild just works.

    Attributes:
        instance_id: The instance whose build failed.
        attempts: How many build attempts the server made.
        hint: The server's instruction for recovering, or ``None``.
    """

    def __init__(
        self,
        message: str = "This instance's memory architecture build failed.",
        correlation_id: Optional[str] = None,
        *,
        instance_id: Optional[str] = None,
        attempts: Optional[int] = None,
        hint: Optional[str] = None,
    ):
        super().__init__(message, correlation_id=correlation_id)
        self.instance_id = instance_id
        self.attempts = attempts
        self.hint = hint


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


# ---------------------------------------------------------------------------
# Documents (tech spec, API and wire contract, Errors)
#
# Each code the Documents routes answer with has its own class, a subclass of
# the class its HTTP status already raised, so an existing
# ``except InvalidInputError`` (or ConflictError, AuthenticationError,
# InsufficientCreditsError) still catches it. ``code`` is the server's code;
# the codes without a class of their own (invalid_utility, invalid_plan,
# invalid_source) set it on the InvalidInputError they raise.
# ---------------------------------------------------------------------------


class FileTooLargeError(InvalidInputError):
    """A file or body is over 10 MB (HTTP 413 ``file_too_large``)."""

    code = "file_too_large"


class BatchTooLargeError(InvalidInputError):
    """A batch is over 100 items or 50 MB (HTTP 413 ``batch_too_large``)."""

    code = "batch_too_large"


class UnsupportedFormatError(InvalidInputError):
    """The file's format is not accepted (HTTP 415 ``unsupported_format``)."""

    code = "unsupported_format"


class UrlNotFetchedError(InvalidInputError):
    """A file part carried only a URL (HTTP 422 ``url_not_fetched``).

    The SDK fetches a ``FilePart(url=...)`` itself; the server never fetches a
    caller's URL."""

    code = "url_not_fetched"


class FiltersNeedQueryError(InvalidInputError):
    """Document filters on a fetch with no query (HTTP 422 ``filters_need_query``).

    A fetch with ``kinds``, ``document_types``, ``document_ids`` or ``as_of``
    needs a ``search_query``. To read a document without one, use
    ``documents.get``."""

    code = "filters_need_query"


class MissingScopeError(AuthenticationError):
    """The API key lacks the scope this route needs (HTTP 403 ``missing_scope``).

    ``scope`` names the scope it needs, for example ``sdk:default`` for a
    write-only (``memories:write``) key calling a read route."""

    code = "missing_scope"

    def __init__(self, message: str, scope: Optional[str] = None,
                 correlation_id: Optional[str] = None):
        super().__init__(message, correlation_id=correlation_id)
        self.scope = scope


class FeatureNotInPlanError(AuthenticationError):
    """The plan does not include this feature (HTTP 403 ``feature_not_in_plan``).

    For example a premium parser, a connector or live fetch on a plan without
    it. ``upgrade_url`` is where to upgrade."""

    code = "feature_not_in_plan"

    def __init__(self, message: str, feature: Optional[str] = None,
                 plan_id: Optional[str] = None, upgrade_url: Optional[str] = None,
                 request_id: Optional[str] = None, correlation_id: Optional[str] = None):
        super().__init__(message, correlation_id=correlation_id)
        self.feature = feature
        self.plan_id = plan_id
        self.upgrade_url = upgrade_url
        self.request_id = request_id


class DocumentIdInUseError(ConflictError):
    """A restore found a newer document holding the same identity (HTTP 409
    ``document_id_in_use``)."""

    code = "document_id_in_use"


class AmbiguousDocumentError(ConflictError):
    """A ``document_id`` names more than one document across scopes (HTTP 409
    ``ambiguous_document``).

    ``matches`` lists each match's scope. Pass the ``user_id``,
    ``customer_id`` or ``scope`` that names one, or its ``document_key``."""

    code = "ambiguous_document"

    def __init__(self, message: str, matches: Optional[list] = None,
                 correlation_id: Optional[str] = None):
        super().__init__(message, correlation_id=correlation_id)
        self.matches = list(matches or [])


#: The name the tech spec uses.
AmbiguousDocument = AmbiguousDocumentError


class InsufficientCreditsForDocumentError(InsufficientCreditsError):
    """Overages are off and the balance is below this document's estimate
    (HTTP 402 ``insufficient_credits_for_document``).

    ``estimate_credits`` is what the document is expected to cost;
    ``balance_credits`` is what the wallet holds."""

    code = "insufficient_credits_for_document"

    def __init__(self, message: str, estimate_credits: Optional[float] = None,
                 balance_credits: Optional[float] = None,
                 correlation_id: Optional[str] = None):
        super().__init__(message, balance_credits=balance_credits,
                         correlation_id=correlation_id)
        self.estimate_credits = estimate_credits


# Backward compatibility aliases
SDKError = SynapError
TransientError = SynapTransientError
PermanentError = SynapPermanentError
ConnectionError = NetworkTimeoutError
