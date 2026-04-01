"""
tools/refund_tools.py

LangChain tool definitions for refund and return operations.

These tools implement the automated refund flow that the OrderAgent can
execute on behalf of the user:
    1. Check eligibility (policy check before initiating refund).
    2. Create refund (if eligible).
    3. Query refund status (for follow-up inquiries).

Refund policy rules (implemented in MockOrderService):
    - Refund allowed within 7 days of delivery for non-perishable items.
    - Damaged/wrong items: refund allowed within 30 days with photo evidence.
    - Refund processing time: 3–7 business days after approval.

Production safety notes:
    - create_refund is a write operation: add idempotency key (order_id + timestamp)
      to prevent duplicate submissions.
    - Rate-limit refund creation per user (max 3 refunds per day) to prevent abuse.
    - Log all refund tool calls to an audit trail for compliance.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from services.mock_order_service import MockOrderService

logger = logging.getLogger(__name__)

_order_service = MockOrderService()


@tool
def check_refund_eligibility(order_id: str) -> dict:
    """
    Check whether an order qualifies for a refund under the current policy.

    Always call this tool BEFORE create_refund to confirm eligibility.
    Inform the user of the reason if they are not eligible.

    Args:
        order_id: Order ID to check.

    Returns:
        Dict with keys:
            order_id   (str):  Order ID.
            eligible   (bool): True if a refund can be initiated.
            reason     (str):  Human-readable eligibility explanation.
                               E.g. "Order delivered 5 days ago. Refund allowed."
            deadline   (str):  Last date by which a refund can be requested
                               (YYYY-MM-DD), or null if not eligible.
        On error: {"error": str, "order_id": str}.

    How to implement:
        return _order_service.check_refund_eligibility(order_id)
    """
    # TODO: implement
    pass


@tool
def create_refund(order_id: str, reason: str) -> dict:
    """
    Initiate a refund request for an eligible order.

    Only call this tool after check_refund_eligibility confirms eligible=True
    and the user has explicitly confirmed they want to proceed.

    Args:
        order_id: Order ID to refund.
        reason:   User-provided reason for the refund (e.g. "商品与描述不符").
                  Stored in the refund record for customer service review.

    Returns:
        Dict with keys:
            refund_id    (str):  Unique refund request ID (e.g. "REF-20240310-001").
            order_id     (str):  Associated order ID.
            status       (str):  Initial status: "pending_review".
            amount       (float): Refund amount in CNY.
            created_at   (str):  Timestamp of refund creation.
            estimated_completion (str): Expected completion date.
            message      (str):  Confirmation message for the user.
        On error: {"error": str, "order_id": str}.

    How to implement:
        return _order_service.create_refund(order_id, reason)
    """
    # TODO: implement
    pass


@tool
def get_refund_status(refund_id: str) -> dict:
    """
    Query the current status of an existing refund request.

    Use this tool when the user asks "What happened to my refund?" or
    "Has my refund been approved?".

    Args:
        refund_id: Refund request ID returned by create_refund
                   (e.g. "REF-20240310-001").

    Returns:
        Dict with keys:
            refund_id  (str):  Refund request ID.
            order_id   (str):  Associated order ID.
            status     (str):  Current status: "pending_review", "approved",
                               "processing", "completed", "rejected".
            amount     (float): Refund amount in CNY.
            updated_at (str):  Timestamp of the last status change.
            message    (str):  Status explanation for the user.
        On error: {"error": str, "refund_id": str}.

    How to implement:
        return _order_service.get_refund_status(refund_id)
    """
    # TODO: implement
    pass
