"""String enums print as their value, as the JS SDK's plain strings do.

`class X(str, Enum)` formats as "IngestStatus.QUEUED" on Python 3.11+, so the
Quickstart's `print(f"Status: {response.status}")` printed that where JS
printed "queued". They are StrEnum now.
"""
from enum import StrEnum

import pytest

from maximem_synap.memories import models as mm
from maximem_synap.models import enums as me
from maximem_synap.telemetry.models import TelemetryEventType
from maximem_synap.transport.grpc_client import StreamState

ENUMS = [mm.DocumentType, mm.IngestMode, mm.IngestStatus, mm.MergeStrategy, mm.ForgetMode,
         mm.ForgetStatus, me.ContextScope, me.ContextType, me.CompactionLevel, me.LogLevel,
         me.RetrievalMode, TelemetryEventType, StreamState]


@pytest.mark.parametrize("enum", ENUMS, ids=lambda e: e.__name__)
def test_every_member_prints_and_formats_as_its_value(enum):
    assert issubclass(enum, StrEnum)
    for member in enum:
        assert str(member) == member.value
        assert f"{member}" == member.value
        assert member == member.value, "still equal to the plain string"


def test_the_quickstart_line():
    assert f"Status: {mm.IngestStatus('queued')}" == "Status: queued"
