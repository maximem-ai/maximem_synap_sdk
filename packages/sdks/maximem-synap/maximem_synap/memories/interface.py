"""Interface for memory operations."""

import asyncio
import json as _json
import os
import time
from datetime import datetime
from typing import IO, List, Optional, Union
from uuid import UUID

from ..utils.correlation import generate_correlation_id
from ..utils.datetime_utils import parse_iso_datetime as _parse_dt
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
    """

    def __init__(self, sdk: "MaximemSynapSDK"):
        self._sdk = sdk

    async def create(
        self,
        document: str,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        document_type: str = "ai-chat-conversation",
        document_id: Optional[str] = None,
        document_created_at: Optional[datetime] = None,
        mode: str = "long-range",
        metadata: Optional[dict] = None,
        **kwargs,
    ) -> CreateMemoryResponse:
        """
        Create a memory from a document.

        This is an async operation - returns immediately with an ingestion_id.
        Use `status()` to check progress.

        Args:
            document: The content to memorize
            user_id: External user ID this memory is about. Optional — omit
                for customer- or client-scope ingestion.
            customer_id: External customer ID this memory belongs to. Optional —
                omit for client-scope ingestion. The server derives the
                effective scope from the IDs passed and the instance's
                user_context_isolation (B2C/B2B).
            document_type: Type of document (ai-chat-conversation, email, pdf, etc.)
            document_id: Optional client-provided ID
            document_created_at: When the document was originally created
            mode: "fast" or "long-range"
            metadata: Additional metadata

        Returns:
            CreateMemoryResponse with ingestion_id and status
        """
        self._sdk._ensure_initialized()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        start_time = time.time()

        request = CreateMemoryRequest(
            document=document,
            document_type=DocumentType(document_type),
            document_id=document_id,
            document_created_at=document_created_at,
            user_id=user_id,
            customer_id=customer_id,
            mode=IngestMode(mode),
            metadata=metadata or {},
        )

        emit_memory_event(
            self._sdk._telemetry_collector,
            TelemetryEventType.MEMORY_CREATE,
            correlation_id=correlation_id,
            status="started",
        )

        try:
            auth_context = await self._sdk._get_auth_context(correlation_id)

            result = await self._sdk._http_transport.post(
                path="/api/v1/memories/create",
                auth_context=auth_context,
                json=request.model_dump(mode="json"),
                correlation_id=correlation_id,
            )

            response = CreateMemoryResponse(
                ingestion_id=UUID(result["ingestion_id"]),
                document_id=result["document_id"],
                status=IngestStatus(result["status"]),
                queued_at=_parse_dt(result["queued_at"]),
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
                        queued_at=_parse_dt(r["queued_at"]),
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
                queued_at=_parse_dt(result["queued_at"]),
                started_at=_parse_dt(result.get("started_at")),
                completed_at=_parse_dt(result.get("completed_at")),
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
                created_at=_parse_dt(result["created_at"]),
                updated_at=_parse_dt(result.get("updated_at")),
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
        customer_id: str,
        relationship_type: str = "b2c",
        file_path: Optional[str] = None,
        file: Optional[IO[bytes]] = None,
        filename: Optional[str] = None,
        text: Optional[str] = None,
        document_type: Optional[str] = None,
        mode: str = "long-range",
        metadata: Optional[dict] = None,
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

        Returns:
            CreateMemoryResponse with ingestion_id and status
        """
        self._sdk._ensure_initialized()
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
                "customer_id": customer_id,
                "relationship_type": relationship_type,
                "mode": mode,
            }
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
            else:
                raise ValueError("One of file_path, file, or text must be provided.")

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
                queued_at=_parse_dt(result["queued_at"]),
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
                "deleted_at": _parse_dt(result["deleted_at"]),
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
