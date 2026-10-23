"""The declared protobuf/grpcio floors must not be lower than the gencode needs.

Background: the generated stubs were once built by protoc 6.31.1. That gencode
calls ``ValidateProtobufRuntimeVersion(PUBLIC, 6, 31, 1, ...)`` on import, so
it refuses protobuf < 6.31.1, while pyproject still said ``protobuf>=4.25.0``.
Installed next to a library pinning protobuf<6, the live stream silently failed
to open. These tests read the generated headers and pyproject (no protobuf
runtime needed) and fail if the two ever disagree again.
"""
import logging
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "maximem_synap" / "transport" / "proto"
PYPROJECT = (ROOT / "pyproject.toml").read_text()


def _ver(s):
    return tuple(int(x) for x in re.findall(r"\d+", s)[:3])


def _floor(name):
    m = re.search(rf'"{name}\s*>=\s*([\d.]+)', PYPROJECT)
    assert m, f"pyproject.toml declares no lower bound for {name}"
    return _ver(m.group(1))


def _gencode_min_protobuf():
    src = (PROTO / "synap_service_pb2.py").read_text()
    call = re.search(
        r"ValidateProtobufRuntimeVersion\(\s*[\w.]+,\s*(\d+),\s*(\d+),\s*(\d+)", src)
    if call:  # protoc >= 5.26 embeds a hard runtime check
        return tuple(int(x) for x in call.groups())
    hdr = re.search(r"# Protobuf Python Version: ([\d.]+)", src)
    assert hdr, "synap_service_pb2.py has no 'Protobuf Python Version' header"
    return _ver(hdr.group(1))


def test_protobuf_floor_not_below_gencode_minimum():
    assert _floor("protobuf") >= _gencode_min_protobuf() or _gencode_has_no_runtime_check(), (
        f"pyproject protobuf>={_floor('protobuf')} is lower than the gencode's "
        f"minimum {_gencode_min_protobuf()}: the stub would raise VersionError. "
        "Regenerate with grpcio-tools==1.62.3 (synap/proto/compile.sh --sdk-only) "
        "or raise the floor."
    )


def _gencode_has_no_runtime_check():
    return "ValidateProtobufRuntimeVersion" not in (
        PROTO / "synap_service_pb2.py").read_text()


def test_gencode_has_no_hard_runtime_version_check():
    """Stricter, intentional: a hard check makes the wheel unusable beside
    libraries pinning older protobuf. If you must allow one, raise the pyproject
    floor to match and delete this test deliberately."""
    assert _gencode_has_no_runtime_check(), (
        "synap_service_pb2.py calls ValidateProtobufRuntimeVersion; regenerate "
        "with grpcio-tools==1.62.3 via synap/proto/compile.sh --sdk-only")


def test_grpcio_floor_not_below_grpc_gencode_minimum():
    src = (PROTO / "synap_service_pb2_grpc.py").read_text()
    m = re.search(r"GRPC_GENERATED_VERSION\s*=\s*'([\d.]+)'", src)
    if m:  # newer grpcio-tools raise RuntimeError on import if grpcio is older
        assert _floor("grpcio") >= _ver(m.group(1)), (
            f"pyproject grpcio>={_floor('grpcio')} is below the stub's "
            f"GRPC_GENERATED_VERSION {m.group(1)}")


def test_proto_compat_fallback_matches_pyproject():
    from maximem_synap.transport.proto_compat import FALLBACK_REQUIREMENTS
    for name in ("protobuf", "grpcio"):
        assert _ver(FALLBACK_REQUIREMENTS[name]) == _floor(name)


def test_incompatible_protobuf_logs_one_clear_warning(caplog):
    from maximem_synap.transport import proto_compat

    class VersionError(Exception):
        pass

    proto_compat._warned = False
    err = VersionError("Detected incompatible Protobuf Gencode/Runtime versions")
    with caplog.at_level(logging.WARNING, logger="synap.sdk.transport.grpc"):
        fix = proto_compat.warn_incompatible_protobuf(err)
        assert proto_compat.warn_incompatible_protobuf(err) is None  # only once
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "protobuf" in msgs[0] and "pip install 'protobuf>=" in msgs[0]
    assert fix and fix.startswith("pip install 'protobuf>=")
    proto_compat._warned = False


def test_unrelated_errors_are_not_reported_as_protobuf():
    from maximem_synap.transport import proto_compat
    proto_compat._warned = False
    assert proto_compat.warn_incompatible_protobuf(ValueError("boom")) is None
    assert proto_compat.warn_incompatible_protobuf(KeyError("x")) is None
