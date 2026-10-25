"""Interface for memory operations."""

import asyncio
import json as _json
import os
import time
from datetime import datetime
from typing import IO, Any, Dict, List, Optional, Union
from uuid import UUID

from ..utils.correlation import generate_correlation_id
from ..models.errors import InvalidInputError, SynapError
from ..telemetry.collector import emit_memory_event
from ..telemetry.models import TelemetryEventType
from .models import (
    CreateMemoryRequest,
    CreateMemoryResponse,
    BatchCreateRequest,
    BatchCreateResponse,
    MemoryStatusResponse,
    UpdateMemoryRequest,
    Memory,
    IngestStatus,
    DocumentType,
    IngestMode,
    MergeStrategy,
    ForgetMode,
    ForgetResponse,
    AttachmentResult,
)
from .parts import (
    FilePart,
    TextPart,
    compact_json,
    has_files,
    join_text,
    multipart_wire_parts,
    resolve_files,
    to_parts,
)


class MemoriesInterface:
    """
    Interface for memory operations.

    Usage:
        # Create user-scoped memory
        result = await sdk.memories.create(
            document="User prefers email communication",
            user_id="user-123",
            customer_id="customer-456",
        )

        # Create client-scope memory (no user_id / customer_id)
        result = await sdk.memories.create(
            document="Company refund policy: 30-day window.",
            document_type="document",
        )

        # A conversation with an image in it (png, jpeg or webp)
        result = await sdk.memories.create(
            document=["Here is my receipt", FilePart(path="receipt.png")],
            user_id="user-123",
            customer_id="customer-456",
        )

        # Check status
        status = await sdk.memories.status(result.ingestion_id)

        # Batch create
        batch = await sdk.memories.batch_create([
            CreateMemoryRequest(document="...", user_id="user-123", customer_id="customer-456"),
            CreateMemoryRequest(document="...", user_id="user-123", customer_id="customer-456"),
        ])

        # Get memory
        memory = await sdk.memories.get(memory_id)

        # Update memory
        await sdk.memories.update(memory_id, document="Updated content")

        # Delete memory
        await sdk.memories.delete(memory_id)

        # Forget by description (preview first, a person approves it)
        result = await sdk.memories.forget(
            "my old home address",
            user_id="user-123",
            customer_id="customer-456",
        )
    """

    def __init__(self, sdk: "MaximemSynapSDK"):
        self._sdk = sdk

    async def create(
        self,
        document: Union[
            str, Dict[str, Any], TextPart, FilePart, List[Union[str, TextPart, FilePart]]
        ],
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        document_type: Optional[str] = None,
        document_id: Optional[str] = None,
        document_created_at: Optional[datetime] = None,
        mode: str = "long-range",
        metadata: Optional[dict] = None,
        scope: Optional[Dict[str, str]] = None,
        objective: Optional[str] = None,
        **kwargs,
    ) -> CreateMemoryResponse:
        """
        Create a memory from a document.

        This is an async operation - returns immediately with an ingestion_id.
        Use `status()` to check progress.

        Args:
            document: The content to memorize. One of:
                - a string, or a dict (sent as it is);
                - a `TextPart`, sent as its text;
                - a list of strings, `TextPart` and `FilePart`. A list holding
                  only text is joined with newlines and sent exactly as that
                  string would be. A list holding a `FilePart` is uploaded
                  with its files: each image is read and described, and the
                  description takes the image's place in the text. Up to 10
                  files, 10 MB each. Only png, jpeg and webp images are
                  accepted today; use `create_from_file` for documents.
                  `FilePart(url=...)` is fetched by the SDK, never the server.
                  Needs a Synap server that supports file parts; an older one
                  raises InvalidInputError saying so.
                - a single `FilePart`, the same as a list holding only it.
                An empty list raises InvalidInputError.
            user_id: External user ID this memory is about. Optional — omit
                for customer- or client-scope ingestion.
            customer_id: External customer ID this memory belongs to. Optional —
                omit for client-scope ingestion. The server derives the
                effective scope from the IDs passed and the instance's
                user_context_isolation (B2C/B2B).
            document_type: Type of document (ai-chat-conversation, email, pdf,
                etc.). Omitted, a text document is sent as
                "ai-chat-conversation", as it always was; a document with files
                leaves it to the server, which reads it as a conversation.
            document_id: Optional client-provided ID
            document_created_at: When the document was originally created
            mode: "fast" or "long-range"
            metadata: Additional metadata
            scope: The level this memory belongs to, named in full, one entry
                per level down to it, for example
                {"customer": "acme", "team": "payments", "user": "dana"}.
                Only needed if your scope ladder has a level between customer
                and user: two ids cannot say which one a write belongs to, and
                the server refuses rather than guessing. Omit it otherwise.
                {"client": "self"} names your own client level: the same as
                omitting it with no ids, and an error with a user_id or
                customer_id.
            objective: What this write is for, stored with the ingestion. One
                of `OBJECTIVES` ("conversation", "bootstrap", "knowledge-base",
                "record", "profile") or free text, up to 500 characters. Only
                sent when given.

        Returns:
            CreateMemoryResponse with ingestion_id and status. `attachments`
            lists what happened to each file, in order, when there were files.

        Raises:
            InvalidInputError: an empty list, a bad part, a file over 10 MB,
                a path that cannot be read or a URL that cannot be fetched, a
                contradictory scope, or a server too old to take files.
        """
        self._sdk._check_customer_id(customer_id, where="memories.create")
        self._sdk._ensure_initialized()

        # A string or a dict goes to the model untouched: those callers'
        # bodies must not change by a byte. Only the new shapes are unpacked.
        parts = None
        if isinstance(document, (TextPart, FilePart, list, tuple)):
            parts = to_parts(document, what="document")
            if not has_files(parts):
                document = join_text(parts)
                parts = None

        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = time.time()

        request = CreateMemoryRequest(
            document=document if parts is None else "",
            document_type=DocumentType(
                document_type if document_type is not None else DocumentType.CONVERSATION
            ),
            document_id=document_id,
            document_created_at=document_created_at,
            user_id=user_id,
            customer_id=customer_id,
            scope=scope,
            mode=IngestMode(mode),
            metadata=metadata or {},
        )
        body = request.model_dump(mode="json")
        if objective is not None:
            body["objective"] = objective

        emit_memory_event(
            self._sdk._telemetry_collector,
            TelemetryEventType.MEMORY_CREATE,
            correlation_id=correlation_id,
            status="started",
        )

        try:
            if parts is None:
                auth_context = await self._sdk._get_auth_context(correlation_id)

                result = await self._sdk._http_transport.post(
                    path="/api/v1/memories/create",
                    auth_context=auth_context,
                    json=body,
                    correlation_id=correlation_id,
                )
            else:
                resolved = await resolve_files(parts)
                manifest_parts, files = multipart_wire_parts(resolved)
                body["document"] = manifest_parts
                # Unset stays null: the server picks the type for a document
                # with files rather than the SDK guessing one.
                body["document_type"] = (
                    DocumentType(document_type).value if document_type is not None else None
                )

                auth_context = await self._sdk._get_auth_context(correlation_id)
                try:
                    result = await self._sdk._http_transport.post_multipart(
                        path="/api/v1/memories/create-multipart",
                        auth_context=auth_context,
                        data={"manifest": compact_json(body)},
                        files=files,
                        correlation_id=correlation_id,
                    )
                except SynapError as e:
                    # A server older than the route answers 405 (another
                    # method shares the path) or 404. Read by status, since
                    # 405 and 404 map to different exception classes.
                    if e.status_code not in (404, 405):
                        raise
                    raise InvalidInputError(
                        "This Synap server does not support file parts in "
                        "memories.create. Upgrade the Synap server, or send "
                        "text only.",
                        correlation_id=correlation_id,
                    ) from e

            attachments = result.get("attachments")
            response = CreateMemoryResponse(
                ingestion_id=UUID(result["ingestion_id"]),
                document_id=result["document_id"],
                status=IngestStatus(result["status"]),
                queued_at=datetime.fromisoformat(result["queued_at"]),
                attachments=(
                    [AttachmentResult(**a) for a in attachments]
                    if attachments is not None else None
                ),
            )

            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_CREATE,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="success",
            )

            return response

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_CREATE,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="error",
                error_code=type(e).__name__,
            )
            raise

    async def batch_create(
        self,
        documents: List[CreateMemoryRequest],
        fail_fast: bool = False,
    ) -> BatchCreateResponse:
        """
        Create multiple memories in batch.

        Args:
            documents: List of CreateMemoryRequest objects
            fail_fast: Stop on first error if True

        Returns:
            BatchCreateResponse with results for each document
        """
        self._sdk._ensure_initialized()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = time.time()

        request = BatchCreateRequest(
            documents=documents,
            fail_fast=fail_fast,
        )

        emit_memory_event(
            self._sdk._telemetry_collector,
            TelemetryEventType.MEMORY_BATCH_CREATE,
            correlation_id=correlation_id,
            status="started",
        )

        try:
            auth_context = await self._sdk._get_auth_context(correlation_id)

            result = await self._sdk._http_transport.post(
                path="/api/v1/memories/batch",
                auth_context=auth_context,
                json=request.model_dump(mode="json"),
                correlation_id=correlation_id,
            )

            response = BatchCreateResponse(
                batch_id=UUID(result["batch_id"]),
                total=result["total"],
                succeeded=result["succeeded"],
                failed=result["failed"],
                results=[
                    CreateMemoryResponse(
                        ingestion_id=UUID(r["ingestion_id"]),
                        document_id=r["document_id"],
                        status=IngestStatus(r["status"]),
                        queued_at=datetime.fromisoformat(r["queued_at"]),
                        error_message=r.get("error_message"),
                    )
                    for r in result["results"]
                ],
            )

            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_BATCH_CREATE,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="success",
                batch_total=response.total,
                batch_succeeded=response.succeeded,
                batch_failed=response.failed,
            )

            return response

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_BATCH_CREATE,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="error",
                error_code=type(e).__name__,
            )
            raise

    async def status(self, ingestion_id: UUID) -> MemoryStatusResponse:
        """
        Check status of an ingestion job.

        Args:
            ingestion_id: The ingestion job ID from create()

        Returns:
            MemoryStatusResponse with current status
        """
        self._sdk._ensure_initialized()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = time.time()

        try:
            auth_context = await self._sdk._get_auth_context(correlation_id)

            result = await self._sdk._http_transport.get(
                path=f"/api/v1/memories/status/{ingestion_id}",
                auth_context=auth_context,
                correlation_id=correlation_id,
            )

            response = MemoryStatusResponse(
                ingestion_id=UUID(result["ingestion_id"]),
                document_id=result["document_id"],
                status=IngestStatus(result["status"]),
                queued_at=datetime.fromisoformat(result["queued_at"]),
                started_at=datetime.fromisoformat(result["started_at"]) if result.get("started_at") else None,
                completed_at=datetime.fromisoformat(result["completed_at"]) if result.get("completed_at") else None,
                memories_created=result.get("memories_created", 0),
                memory_ids=result.get("memory_ids", []),
                error_message=result.get("error_message"),
            )

            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_STATUS,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="success",
            )

            return response

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_STATUS,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="error",
                error_code=type(e).__name__,
            )
            raise

    async def wait_for_completion(
        self,
        ingestion_id: UUID,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 2,
    ) -> MemoryStatusResponse:
        """
        Wait for an ingestion job to complete.

        Args:
            ingestion_id: The ingestion job ID
            timeout_seconds: Maximum time to wait
            poll_interval_seconds: How often to poll

        Returns:
            Final MemoryStatusResponse

        Raises:
            TimeoutError: If job doesn't complete in time
        """
        start = time.time()

        while time.time() - start < timeout_seconds:
            status = await self.status(ingestion_id)

            if status.status in [IngestStatus.COMPLETED, IngestStatus.FAILED, IngestStatus.PARTIAL_SUCCESS]:
                return status

            await asyncio.sleep(poll_interval_seconds)

        raise TimeoutError(f"Ingestion {ingestion_id} did not complete within {timeout_seconds}s")

    async def get(self, memory_id: UUID) -> Memory:
        """
        Get a specific memory by ID.

        Args:
            memory_id: The memory ID

        Returns:
            Memory object
        """
        self._sdk._ensure_initialized()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = time.time()

        try:
            auth_context = await self._sdk._get_auth_context(correlation_id)

            result = await self._sdk._http_transport.get(
                path=f"/api/v1/memories/{memory_id}",
                auth_context=auth_context,
                correlation_id=correlation_id,
            )

            response = Memory(
                memory_id=UUID(result["memory_id"]),
                memory_type=result["memory_type"],
                content=result["content"],
                confidence=result["confidence"],
                category=result.get("category"),
                subcategory=result.get("subcategory"),
                created_at=datetime.fromisoformat(result["created_at"]),
                updated_at=datetime.fromisoformat(result["updated_at"]) if result.get("updated_at") else None,
                source_document_id=result.get("source_document_id"),
            )

            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_GET,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="success",
            )

            return response

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_GET,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="error",
                error_code=type(e).__name__,
            )
            raise

    async def update(
        self,
        memory_id: UUID,
        document: str,
        merge_strategy: str = "smart-merge",
        document_type: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Memory:
        """
        Update an existing memory.

        Args:
            memory_id: The memory to update
            document: Updated content
            merge_strategy: How to merge (replace, append, smart-merge)
            document_type: Optional new document type
            metadata: Optional updated metadata

        Returns:
            Updated Memory object
        """
        self._sdk._ensure_initialized()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = time.time()

        request = UpdateMemoryRequest(
            document=document,
            merge_strategy=MergeStrategy(merge_strategy),
            document_type=DocumentType(document_type) if document_type else None,
            metadata=metadata,
        )

        try:
            auth_context = await self._sdk._get_auth_context(correlation_id)

            result = await self._sdk._http_transport.request(
                method="PATCH",
                path=f"/api/v1/memories/{memory_id}",
                auth_context=auth_context,
                json=request.model_dump(mode="json", exclude_none=True),
                correlation_id=correlation_id,
            )

            response = Memory.model_validate(result)

            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_UPDATE,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="success",
            )

            return response

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_UPDATE,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="error",
                error_code=type(e).__name__,
            )
            raise

    async def create_from_file(
        self,
        user_id: str,
        # Optional: on a B2C instance sending one is rejected by the server, and
        # `relationship_type` already defaults to "b2c", so the old mandatory
        # signature made the documented default combination impossible to call.
        customer_id: Optional[str] = None,
        relationship_type: str = "b2c",
        file_path: Optional[str] = None,
        file: Optional[IO[bytes]] = None,
        filename: Optional[str] = None,
        text: Optional[str] = None,
        document_type: Optional[str] = None,
        mode: str = "long-range",
        metadata: Optional[dict] = None,
        scope: Optional[Dict[str, str]] = None,
    ) -> CreateMemoryResponse:
        """
        Ingest a file or raw text into the memory pipeline.

        Exactly one of `file_path`, `file`, or `text` must be provided.

        Args:
            user_id: User this memory is about
            customer_id: Customer this memory belongs to
            relationship_type: "b2b" or "b2c" (default "b2c")
            file_path: Path on disk to read and upload
            file: Open binary file-like object (must also pass `filename`)
            filename: Name to use for the file (required when passing `file=`)
            text: Raw text content to ingest instead of a file
            document_type: Override detected document type
            mode: "fast" or "long-range" (default "long-range")
            metadata: Additional metadata dict (serialized to JSON)
            scope: The level this memory belongs to, named in full, one entry
                per level down to it, for example
                {"customer": "acme", "team": "payments", "user": "dana"}.
                Only needed if your scope ladder has a level between customer
                and user: two ids cannot say which one a write belongs to, and
                the server refuses rather than guessing. Omit it otherwise.
                Sent as a JSON string, because the upload route is multipart.

        Returns:
            CreateMemoryResponse with ingestion_id and status
        """
        if scope is not None and (
            not isinstance(scope, dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in scope.items())
        ):
            raise InvalidInputError(
                "scope must be a flat dict of level key to id, both strings."
            )
        self._sdk._check_customer_id(customer_id, where="memories.create_from_file")
        self._sdk._ensure_initialized()
        # Refused here with InvalidInputError, before anything is read or
        # sent, as the JS SDK refuses them. Until 0.5.22 an empty user_id was
        # posted, several sources were accepted with the first silently
        # winning, and no source was a plain ValueError.
        if not user_id:
            raise InvalidInputError("user_id is required")
        sources = [s for s in (file_path, file, text) if s is not None]
        if not sources:
            raise InvalidInputError("One of file_path, file, or text must be provided.")
        if len(sources) > 1:
            raise InvalidInputError(
                "Provide exactly one of file_path, file, or text, not several."
            )
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = time.time()

        emit_memory_event(
            self._sdk._telemetry_collector,
            TelemetryEventType.MEMORY_CREATE,
            correlation_id=correlation_id,
            status="started",
        )

        try:
            auth_context = await self._sdk._get_auth_context(correlation_id)

            # Build form data fields
            form_data: dict = {
                "user_id": user_id,
                "relationship_type": relationship_type,
                "mode": mode,
            }
            # Only when there is one; httpx would otherwise send an empty field.
            if customer_id:
                form_data["customer_id"] = customer_id
            if scope:
                form_data["scope"] = _json.dumps(scope)
            if document_type:
                form_data["document_type"] = document_type
            if metadata:
                form_data["metadata"] = _json.dumps(metadata)

            # Build files dict for multipart
            files: dict = {}
            if file_path is not None:
                fname = os.path.basename(file_path)
                with open(file_path, "rb") as fh:
                    file_bytes = fh.read()
                files["file"] = (fname, file_bytes)
            elif file is not None:
                fname = filename or "upload"
                files["file"] = (fname, file.read())
            elif text is not None:
                form_data["text"] = text

            result = await self._sdk._http_transport.post_multipart(
                path="/api/v1/memories/upload",
                auth_context=auth_context,
                data=form_data,
                files=files if files else None,
                correlation_id=correlation_id,
            )

            response = CreateMemoryResponse(
                ingestion_id=UUID(result["ingestion_id"]),
                document_id=result["document_id"],
                status=IngestStatus(result["status"]),
                queued_at=datetime.fromisoformat(result["queued_at"]),
            )

            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_CREATE,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="success",
            )

            return response

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_CREATE,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="error",
                error_code=type(e).__name__,
            )
            raise

    async def delete(self, memory_id: UUID) -> dict:
        """
        Delete a memory.

        Args:
            memory_id: The memory to delete

        Returns:
            Confirmation with deleted memory_id and timestamp
        """
        self._sdk._ensure_initialized()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = time.time()

        try:
            auth_context = await self._sdk._get_auth_context(correlation_id)

            result = await self._sdk._http_transport.delete(
                path=f"/api/v1/memories/{memory_id}",
                auth_context=auth_context,
                correlation_id=correlation_id,
            )

            response = {
                "memory_id": UUID(result["memory_id"]),
                "deleted_at": datetime.fromisoformat(result["deleted_at"]),
            }

            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_DELETE,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="success",
            )

            return response

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_DELETE,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="error",
                error_code=type(e).__name__,
            )
            raise

    async def forget(
        self,
        description: str,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
        mode: str = "preview",
    ) -> ForgetResponse:
        """
        Forget memories by describing them in the user's own words.

        Use this when you do not hold memory ids: the server finds the
        memories the description matches. To delete one memory you already
        have the id of, use `delete()`.

        The call must be scoped to exactly one user: pass `user_id`, plus
        `customer_id` on a B2B instance. Without it the server answers 400
        (raised as `InvalidInputError`).

        Modes:
            "preview" (default): files a pending request that a person
                approves in the dashboard under Deletion requests. Nothing is
                deleted by this call. `candidates` lists what matched.
            "direct": finds and soft-deletes in one call. Not idempotent as a
                request: a repeat is harmless and finds nothing new, but it
                is a new request.

        Args:
            description: What to forget, in the user's words (3 to 500
                characters).
            user_id: External user ID whose memories to search.
            customer_id: External customer ID. Required on B2B, rejected on
                B2C.
            scope: The level to search, named in full, one entry per level
                down to it, for example
                {"customer": "acme", "team": "payments", "user": "dana"}.
                Only needed if your scope ladder has a level between customer
                and user.
            mode: "preview" or "direct".

        Returns:
            ForgetResponse. `status` is one of "nothing_matches", "preview",
            "deleted" or "partial". "partial" means the memories are hidden
            but a store named in `still_present_in` may still hold the text,
            so do not tell an end user it was removed unless
            `status == "deleted"`. `failed_ids` lists memories that could not
            be deleted and `may_be_incomplete` is True when the match may not
            have covered everything.

        Raises:
            InvalidInputError: 400 (not scoped to one user) or 422
                (validation), or a bad `description` or `mode` caught before
                the request is sent.
            ServiceUnavailableError: 503. The matcher was unavailable and
                nothing was deleted, so it is safe to try again. The
                transport applies its normal retry policy first.
            RateLimitError: 429, as on the other routes.
        """
        self._sdk._check_customer_id(customer_id, where="memories.forget")
        self._sdk._ensure_initialized()

        if not isinstance(description, str) or not 3 <= len(description) <= 500:
            raise InvalidInputError(
                "description must be between 3 and 500 characters"
            )
        try:
            forget_mode = ForgetMode(mode)
        except ValueError:
            raise InvalidInputError(
                f"Invalid mode '{mode}'. Expected one of: "
                f"{', '.join(m.value for m in ForgetMode)}"
            ) from None

        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = time.time()

        body = {
            "description": description,
            "user_id": user_id,
            "customer_id": customer_id,
            "scope": scope,
            "mode": forget_mode.value,
        }

        emit_memory_event(
            self._sdk._telemetry_collector,
            TelemetryEventType.MEMORY_DELETE,
            correlation_id=correlation_id,
            status="started",
        )

        try:
            auth_context = await self._sdk._get_auth_context(correlation_id)

            result = await self._sdk._http_transport.post(
                path="/api/v1/memories/forget",
                auth_context=auth_context,
                json=body,
                correlation_id=correlation_id,
            )

            response = ForgetResponse(**result)

            # Something was deleted on the server: stop serving it from this
            # process's caches too. A preview deletes nothing, so it evicts nothing.
            # A forget can remove only copies (a profile entry, a chunk) and so return
            # no memory ids; those are served from the same caches.
            if response.deleted_ids or response.status in ("deleted", "partial"):
                self._sdk._evict_forgotten(user_id, customer_id)

            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_DELETE,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="success",
            )

            return response

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            emit_memory_event(
                self._sdk._telemetry_collector,
                TelemetryEventType.MEMORY_DELETE,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                status="error",
                error_code=type(e).__name__,
            )
            raise
