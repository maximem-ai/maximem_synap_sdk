"""Guards on the package's public surface.

A failure here means a published import path changed, which is a breaking
change even when every other test still passes.
"""

import pytest

import synap_harnessx

EXPECTED_EXPORTS = {
    "SYNAP_MEMORY_PROMPT",
    "SynapExtension",
    "make_synap_tools",
}


def test_all_matches_expected_exports():
    assert set(synap_harnessx.__all__) == EXPECTED_EXPORTS


def test_all_is_sorted():
    assert synap_harnessx.__all__ == sorted(synap_harnessx.__all__)


@pytest.mark.parametrize("name", sorted(EXPECTED_EXPORTS))
def test_every_export_is_importable(name):
    assert getattr(synap_harnessx, name) is not None
