"""Refund helpers for the Harbor Labs demo store."""

from __future__ import annotations


def refund_is_allowed(amount_cents: int, days_since_purchase: int) -> bool:
    """Return True when a refund is still inside the 30-day policy window.

    Refunds over 50_000 cents also need a manager flag, which this
    helper does not set. Callers must check that separately.
    """
    if amount_cents <= 0:
        return False
    return days_since_purchase <= 30
