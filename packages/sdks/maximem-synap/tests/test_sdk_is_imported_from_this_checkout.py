"""The SDK under test is the one in this checkout, not an installed copy.

A wheel from PyPI, or an editable install that points at another checkout,
makes every test in this folder and in synap/tests/sdk test code that is not
the code under review, while still passing. Run the suite with this checkout
first on the path (scripts/test_gate/gate.py does: PYTHONPATH=<repo>:<repo>/synap/sdk/python)
or with `pip install -e synap/sdk/python` from this checkout.
"""

from pathlib import Path

import maximem_synap

THIS_SDK = Path(__file__).resolve().parents[1] / "maximem_synap"


def test_maximem_synap_is_imported_from_this_checkout():
    imported = Path(maximem_synap.__file__).resolve().parent
    assert imported == THIS_SDK, (
        f"maximem_synap was imported from {imported}, not from this checkout's "
        f"{THIS_SDK}. The tests would be testing another copy of the SDK."
    )
