"""One clear warning when the generated protobuf stubs cannot be imported.

The generated ``synap_service_pb2`` modules import lazily, the first time the
stream opens. If the installed ``protobuf`` or ``grpcio`` is outside the range
the wheel was generated for, Python raises a bare ``VersionError`` /
``TypeError`` / ``ImportError`` deep in generated code and the customer sees
a traceback that names neither package. This module turns that into a single
log line that says what is installed, what is required and the command that
fixes it. It never swallows the error: the caller re-raises.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("synap.sdk.transport.grpc")

# Mirrors pyproject.toml. tests/test_protobuf_floor.py asserts they agree.
FALLBACK_REQUIREMENTS = {"protobuf": ">=4.25.0", "grpcio": ">=1.60.0"}

_warned = False


def _installed(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def _required(name: str) -> str:
    """The specifier this wheel declares for ``name`` (e.g. ``>=4.25.0``)."""
    try:
        from importlib.metadata import requires
        for line in requires("maximem-synap") or []:
            m = re.match(rf"^{name}\s*([^;\s]*)", line, re.IGNORECASE)
            if m and "extra" not in line:
                return m.group(1) or FALLBACK_REQUIREMENTS[name]
    except Exception:
        pass
    return FALLBACK_REQUIREMENTS[name]


def is_proto_runtime_error(exc: BaseException) -> bool:
    """True for the failures an incompatible protobuf/grpcio produces at import."""
    if type(exc).__name__ == "VersionError":  # google.protobuf.runtime_version
        return True
    if isinstance(exc, (TypeError, ImportError, RuntimeError)):
        text = str(exc).lower()
        return any(k in text for k in ("protobuf", "descriptor", "gencode", "grpc"))
    return False


def warn_incompatible_protobuf(exc: BaseException) -> Optional[str]:
    """Log ONE warning (per process) for a protobuf/grpcio import failure.

    Returns the fix command when a warning was logged on this call, else None.
    """
    global _warned
    if _warned or not is_proto_runtime_error(exc):
        return None
    _warned = True
    pb_req, grpc_req = _required("protobuf"), _required("grpcio")
    fix = f"pip install 'protobuf{pb_req}' 'grpcio{grpc_req}'"
    logger.warning(
        "maximem-synap could not load its gRPC stubs, so the live stream "
        "(anticipation and automatic memory) will not open. Installed: "
        "protobuf %s, grpcio %s. Required: protobuf%s, grpcio%s. Fix: %s "
        "(another package may pin an older protobuf; run `pip check`). "
        "Cause: %s: %s",
        _installed("protobuf"), _installed("grpcio"), pb_req, grpc_req, fix,
        type(exc).__name__, exc,
    )
    return fix
