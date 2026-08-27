"""The B2C identifier contract, client side.

    B2C   send user_id, and nothing else. customer_id is not accepted.
    B2B   customer_id is required.

The server is authoritative and rejects independently. This exists so the
failure happens at the call site, with a message that says what to do, instead
of as a network round trip that returns an empty result.

That last part is the whole reason. Before this, sending a customer_id on a B2C
instance produced no exception anywhere: the write was filed under the customer,
the read asked for the user, and both returned success. One client ran 4,634
consecutive empty fetches across seven days without a single error to look at.
An SDK that stays silent about a misuse it can see is not being permissive, it
is hiding the bug.
"""

from __future__ import annotations

from typing import Optional

B2C_ISOLATION = "equals_customer"


class CustomerIdNotAcceptedError(ValueError):
    """Raised when a customer_id is passed on a B2C instance."""

    def __init__(self, where: str, customer_id: str, instance_id: Optional[str] = None):
        self.where = where
        self.customer_id = customer_id
        self.instance_id = instance_id
        super().__init__(
            f"{where}: customer_id is not accepted on this instance"
            f"{f' ({instance_id})' if instance_id else ''}, which is B2C "
            f"(user_context_isolation={B2C_ISOLATION!r}). Send user_id only; it "
            f"is the whole identity, and the server files and reads your data "
            f"under it. Received customer_id={customer_id!r}."
        )


def check_customer_id(
    isolation: Optional[str],
    customer_id: Optional[str],
    *,
    where: str,
    instance_id: Optional[str] = None,
) -> None:
    """Raise if this call breaks the contract.

    `isolation` is what `GET /api/v1/auth/whoami` reported. **None means the
    server did not tell us**, which is the case against any deployment older
    than this field, and it must mean "do nothing". Guessing B2C there would
    make this SDK refuse a B2B client's mandatory field against every server
    that has not been upgraded yet, which would be a far worse failure than the
    one it prevents.
    """
    if not customer_id:
        return
    if isolation != B2C_ISOLATION:
        return
    raise CustomerIdNotAcceptedError(where, customer_id, instance_id)
