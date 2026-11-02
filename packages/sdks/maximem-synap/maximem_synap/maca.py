"""MACA plugin sub-interface for the Python SDK (Synap Core).

A Synap Core server can have Synap Cloud write its memory config from a short
description of your use case. The server holds the Synap Cloud key; this
interface only starts the job and reads it back.

Usage:
    sdk = MaximemSynapSDK()
    await sdk.initialize()

    job = await sdk.maca.generate(
        file_path="use-case.md",
        user_relationship="b2c",
    )
    job = await sdk.maca.wait_for_completion(job.job_id)
    print(job.status, job.config_file)

A run takes 15 to 20 minutes. The finished config is saved by the server as
``config/memory.generated.yml``; it never replaces ``memory.yml``.

This exists only on Synap Core. Synap Cloud has no such route and answers 404.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from .models.errors import InvalidInputError
from .utils.correlation import generate_correlation_id

if TYPE_CHECKING:
    from .sdk import MaximemSynapSDK

#: What a job can end as. Anything else is still running.
FINAL_STATUSES = ("succeeded", "invalid", "failed")

_RELATIONSHIPS = ("b2c", "b2b")

#: The server refuses a larger use-case file.
MAX_FILE_BYTES = 256 * 1024


@dataclass(slots=True)
class MacaJob:
    """One MACA plugin job on the local server."""

    job_id: str
    status: str
    message: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    removed_fields: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    config_file: Optional[str] = None
    #: The generated config, once the job has ended with ``succeeded`` or ``invalid``.
    yaml: Optional[str] = None


def _job(result: dict) -> MacaJob:
    return MacaJob(
        job_id=result["job_id"],
        status=result["status"],
        message=result.get("message") or "",
        created_at=result.get("created_at"),
        updated_at=result.get("updated_at"),
        removed_fields=list(result.get("removed_fields") or []),
        problems=list(result.get("problems") or []),
        config_file=result.get("config_file"),
        yaml=result.get("yaml"),
    )


class MacaInterface:
    """``sdk.maca.*``: generate a memory config for Synap Core with the MACA plugin."""

    def __init__(self, sdk: "MaximemSynapSDK"):
        self._sdk = sdk

    async def generate(
        self,
        *,
        file_path: "str | os.PathLike[str]",
        user_relationship: str,
    ) -> MacaJob:
        """Upload a use-case file and start a job.

        Args:
            file_path: The use-case file (UTF-8 text, at most 256 KB), uploaded as is.
            user_relationship: ``"b2c"`` when each user is their own customer,
                ``"b2b"`` when users sit under a customer. The config is built for
                that kind of app, and it cannot change once memories exist.

        Raises:
            InvalidInputError: The relationship is not ``b2c`` or ``b2b``, or the
                file is missing, empty or larger than 256 KB.
        """
        if user_relationship not in _RELATIONSHIPS:
            raise InvalidInputError("user_relationship must be 'b2c' or 'b2b'")
        try:
            size = os.path.getsize(file_path)
        except OSError:
            raise InvalidInputError(f"file_path does not point to a readable file: {file_path}") from None
        if size == 0:
            raise InvalidInputError("The use-case file is empty")
        if size > MAX_FILE_BYTES:
            raise InvalidInputError(f"The use-case file is larger than {MAX_FILE_BYTES // 1024} KB")
        with open(file_path, "rb") as fh:
            data = fh.read()

        self._sdk._ensure_initialized()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        auth_context = await self._sdk._get_auth_context(correlation_id)
        result = await self._sdk._http_transport.post_multipart(
            path="/v1/maca/generate",
            auth_context=auth_context,
            data={"user_relationship": user_relationship},
            files={"file": (os.path.basename(file_path), data)},
            correlation_id=correlation_id,
        )
        return _job(result)

    async def get_job(self, job_id: str) -> MacaJob:
        """Read a job: its status, and the config once it has ended."""
        if not job_id:
            raise InvalidInputError("job_id is required")
        self._sdk._ensure_initialized()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        auth_context = await self._sdk._get_auth_context(correlation_id)
        result = await self._sdk._http_transport.get(
            path=f"/v1/maca/jobs/{job_id}",
            auth_context=auth_context,
            correlation_id=correlation_id,
        )
        return _job(result)

    async def wait_for_completion(
        self,
        job_id: str,
        timeout_seconds: int = 3000,
        poll_interval_seconds: int = 15,
    ) -> MacaJob:
        """Wait until a job is ``succeeded``, ``invalid`` or ``failed`` and return it.

        Raises:
            TimeoutError: The job was still running after ``timeout_seconds``.
        """
        start = time.time()
        while time.time() - start < timeout_seconds:
            job = await self.get_job(job_id)
            if job.status in FINAL_STATUSES:
                return job
            await asyncio.sleep(poll_interval_seconds)
        raise TimeoutError(f"MACA job {job_id} did not finish within {timeout_seconds}s")
