"""Content parts: text and files sent alongside, or instead of, plain text.

`memories.create`, `conversation.record_message` and `instance.send_message`
all accept a list of parts in place of a string. This module is the one place
that turns what the caller passed into what goes on the wire, so the three
routes cannot drift apart.

The rule every helper here keeps: a plain string, and a list holding only
text, end up as exactly the string the caller would have sent before parts
existed. Only a list holding a `FilePart` takes a new path.
"""

import asyncio
import base64
import json as _json
import mimetypes
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

from ..models.errors import InvalidInputError

#: What a write is for. Free text is accepted too; these are the values the
#: server knows. Stored with the ingestion and changes nothing else yet.
OBJECTIVES: Tuple[str, ...] = (
    "conversation",
    "bootstrap",
    "knowledge-base",
    "record",
    "profile",
)
Objective = Literal["conversation", "bootstrap", "knowledge-base", "record", "profile"]

#: Largest single file the SDK will send. The server enforces the same limit
#: (413); checking here saves uploading 10 MB only to have it refused.
MAX_FILE_BYTES = 10 * 1024 * 1024

#: Largest total of raw image bytes in one `send_message` event. The gRPC
#: stream refuses messages over 4 MB, and base64 grows the bytes by a third.
MAX_STREAM_IMAGE_BYTES = int(2.5 * 1024 * 1024)

#: How long a `FilePart(url=...)` fetch may take. The SDK fetches the URL
#: itself; the server never fetches a caller's URL.
URL_FETCH_TIMEOUT_SECONDS = 30.0

#: The `send_message` metadata key carrying the parts as a JSON string.
STREAM_PARTS_METADATA_KEY = "synap_content_parts"

# Looked up before `mimetypes`, whose answer for some of these depends on the
# platform's own tables (webp is missing from several).
_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".pdf": "application/pdf",
}
_DEFAULT_MEDIA_TYPE = "application/octet-stream"


@dataclass(frozen=True)
class TextPart:
    """A piece of text in a list of parts."""

    text: str


@dataclass(frozen=True)
class FilePart:
    """A file in a list of parts.

    Give exactly one of `data` (the bytes), `path` (read from disk when the
    call is made) or `url` (fetched by the SDK when the call is made, 30 s
    timeout). `filename` defaults to the basename of `path` or `url`, else
    "file". `media_type` defaults to a guess from the filename, else
    "application/octet-stream".

    Only images (png, jpeg, webp) are accepted by the server today. Any other
    type is refused by the server, not here.
    """

    data: Optional[bytes] = None
    path: Optional[str] = None
    url: Optional[str] = None
    filename: Optional[str] = None
    media_type: Optional[str] = None

    def __post_init__(self) -> None:
        given = [n for n in ("data", "path", "url") if getattr(self, n) is not None]
        if len(given) != 1:
            raise InvalidInputError(
                "FilePart takes exactly one of data, path or url"
                + (f" (got {', '.join(given)})" if given else "")
            )
        if self.data is not None and not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise InvalidInputError("FilePart.data must be bytes")
        filename = self.filename or _default_filename(self.path, self.url)
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "media_type", self.media_type or guess_media_type(filename))


Part = Union[TextPart, FilePart]
PartsInput = Union[str, TextPart, FilePart, Sequence[Union[str, TextPart, FilePart]]]


@dataclass(frozen=True)
class ResolvedFile:
    """A `FilePart` whose bytes have been read."""

    filename: str
    media_type: str
    data: bytes


def _default_filename(path: Optional[str], url: Optional[str]) -> str:
    if path:
        name = os.path.basename(path)
        if name:
            return name
    if url:
        name = os.path.basename(urlparse(url).path)
        if name:
            return name
    return "file"


def guess_media_type(filename: Optional[str]) -> str:
    """Media type for a filename, or "application/octet-stream"."""
    if not filename:
        return _DEFAULT_MEDIA_TYPE
    ext = os.path.splitext(filename)[1].lower()
    if ext in _MEDIA_TYPES:
        return _MEDIA_TYPES[ext]
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or _DEFAULT_MEDIA_TYPE


def to_parts(value: PartsInput, *, what: str) -> List[Part]:
    """Turn a string, a part, or a list of them into a list of parts.

    Raises InvalidInputError for an empty list or an item that is not a
    string, `TextPart` or `FilePart`.
    """
    if isinstance(value, (str, TextPart, FilePart)):
        items: Sequence[Any] = [value]
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        raise InvalidInputError(
            f"{what} must be a string or a list of strings, TextPart and FilePart"
        )
    if not items:
        raise InvalidInputError(f"{what} is an empty list; pass at least one part")
    parts: List[Part] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            parts.append(TextPart(item))
        elif isinstance(item, (TextPart, FilePart)):
            parts.append(item)
        else:
            raise InvalidInputError(
                f"{what}[{i}] is a {type(item).__name__}; "
                "expected a string, TextPart or FilePart"
            )
    return parts


def is_parts(value: Any) -> bool:
    """Whether `content` is parts to unpack: a list, or a single part.

    Anything else (a string, or whatever a caller passed before parts
    existed) goes out exactly as it did.
    """
    return isinstance(value, (list, tuple, TextPart, FilePart))


def has_files(parts: Sequence[Part]) -> bool:
    return any(isinstance(p, FilePart) for p in parts)


def join_text(parts: Sequence[Part]) -> str:
    """The text of a list holding only text parts, joined with newlines.

    This is the string a caller would have passed before parts existed, so
    it goes down the old path byte for byte.
    """
    return "\n".join(p.text for p in parts if isinstance(p, TextPart))


def placeholder_text(parts: Sequence[Part]) -> str:
    """Text standing in for the parts where bytes cannot go.

    Text as-is, each file as ``[image: <filename>]``, joined with newlines.
    Used for the local short-term store and as the gRPC `content`, so an old
    server still records something meaningful.
    """
    out = []
    for p in parts:
        if isinstance(p, TextPart):
            out.append(p.text)
        else:
            out.append(f"[image: {p.filename or 'image'}]")
    return "\n".join(out)


async def resolve_file(part: FilePart, *, max_bytes: int = MAX_FILE_BYTES) -> ResolvedFile:
    """Read a `FilePart`'s bytes from wherever they are.

    Raises InvalidInputError when the file is over `max_bytes`, a path cannot
    be read, or a URL fetch fails or answers with a non-2xx status.
    """
    filename = part.filename or "file"
    media_type = part.media_type or _DEFAULT_MEDIA_TYPE

    if part.data is not None:
        data = bytes(part.data)
    elif part.path is not None:
        data = await asyncio.to_thread(_read_path, part.path, max_bytes)
    else:
        data, fetched_type = await _fetch_url(part.url, max_bytes)
        # The filename said nothing useful, so trust what the server said.
        if media_type == _DEFAULT_MEDIA_TYPE and fetched_type:
            media_type = fetched_type

    if len(data) > max_bytes:
        raise InvalidInputError(
            f"{filename} is {len(data)} bytes; the limit is {max_bytes} bytes per file"
        )
    return ResolvedFile(filename=filename, media_type=media_type, data=data)


def _read_path(path: str, max_bytes: int) -> bytes:
    try:
        size = os.path.getsize(path)
        if size > max_bytes:
            raise InvalidInputError(
                f"{os.path.basename(path)} is {size} bytes; "
                f"the limit is {max_bytes} bytes per file"
            )
        with open(path, "rb") as fh:
            return fh.read()
    except OSError as e:
        raise InvalidInputError(f"Could not read FilePart path {path!r}: {e}") from e


async def _fetch_url(url: str, max_bytes: int) -> Tuple[bytes, Optional[str]]:
    import httpx

    try:
        async with httpx.AsyncClient(
            timeout=URL_FETCH_TIMEOUT_SECONDS, follow_redirects=True,
        ) as client:
            async with client.stream("GET", url) as response:
                if not 200 <= response.status_code < 300:
                    raise InvalidInputError(
                        f"Fetching FilePart url {url!r} returned HTTP {response.status_code}"
                    )
                chunks = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise InvalidInputError(
                            f"FilePart url {url!r} is over the limit of "
                            f"{max_bytes} bytes per file"
                        )
                    chunks.append(chunk)
                content_type = response.headers.get("content-type")
    except httpx.HTTPError as e:
        raise InvalidInputError(f"Could not fetch FilePart url {url!r}: {e}") from e
    if content_type:
        content_type = content_type.split(";", 1)[0].strip().lower() or None
    return b"".join(chunks), content_type


async def resolve_files(
    parts: Sequence[Part], *, max_total_bytes: Optional[int] = None,
) -> List[Union[TextPart, ResolvedFile]]:
    """Resolve every file in `parts`, in order, keeping text parts as they are.

    With `max_total_bytes`, raises InvalidInputError as soon as the files
    together go over it, before fetching the rest.
    """
    out: List[Union[TextPart, ResolvedFile]] = []
    total = 0
    for p in parts:
        if isinstance(p, TextPart):
            out.append(p)
            continue
        resolved = await resolve_file(p)
        total += len(resolved.data)
        if max_total_bytes is not None and total > max_total_bytes:
            raise InvalidInputError(
                f"The images in this event come to more than {max_total_bytes} bytes, "
                "which is over what one streamed event can carry. Send them with "
                "memories.create or conversation.record_message instead."
            )
        out.append(resolved)
    return out


def inline_wire_parts(resolved: Sequence[Union[TextPart, ResolvedFile]]) -> List[Dict[str, str]]:
    """Wire parts with the image bytes inline as base64.

    The shape `conversation.record_message` sends as `content` and
    `send_message` sends under the metadata key.
    """
    out: List[Dict[str, str]] = []
    for p in resolved:
        if isinstance(p, TextPart):
            out.append({"type": "text", "text": p.text})
        else:
            out.append({
                "type": "image",
                "filename": p.filename,
                "media_type": p.media_type,
                "data": base64.b64encode(p.data).decode("ascii"),
            })
    return out


def multipart_wire_parts(
    resolved: Sequence[Union[TextPart, ResolvedFile]],
) -> Tuple[List[Dict[str, str]], Dict[str, Tuple[str, bytes, str]]]:
    """Manifest parts plus the form files they point at.

    Each file part names its form field, ``file_0``, ``file_1`` and so on in
    order of appearance, and the bytes travel in that field.
    """
    manifest: List[Dict[str, str]] = []
    files: Dict[str, Tuple[str, bytes, str]] = {}
    for p in resolved:
        if isinstance(p, TextPart):
            manifest.append({"type": "text", "text": p.text})
        else:
            field = f"file_{len(files)}"
            manifest.append({
                "type": "file",
                "field": field,
                "filename": p.filename,
                "media_type": p.media_type,
            })
            files[field] = (p.filename, p.data, p.media_type)
    return manifest, files


def compact_json(value: Any) -> str:
    """JSON the way JavaScript's ``JSON.stringify`` writes it, so both SDKs
    put the same bytes on the wire."""
    return _json.dumps(value, separators=(",", ":"), ensure_ascii=False)
