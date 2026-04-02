"""
tools/order_tools.py

LangChain tool definitions for order-related queries.

These functions are decorated with @tool so they can be:
    1. Passed to llm.bind_tools() for structured function calling.
    2. Discovered by the LLM via their docstrings (used as tool descriptions).
    3. Executed by OrderAgent.execute_tool() via tool.invoke(args).

Implementation note:
    The actual data retrieval is delegated to MockOrderService (services/).
    In production, replace the mock with real REST/gRPC calls to the order
    management system, adding auth headers, retries, and circuit-breakers.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from services.mock_order_service import MockOrderService

logger = logging.getLogger(__name__)

# Module-level service instance (singleton pattern for tools)
# In production: inject via dependency injection rather than module-level state.
_order_service = MockOrderService()


@tool
def get_order_status(order_id: str) -> dict:
    """
    Retrieve the current status of an order by its order ID.

    Use this tool when the user asks about the status of a specific order,
    such as "Is my order shipped?", "What is the status of ORD-001?", etc.

    Args:
        order_id: The unique order identifier string (e.g. "ORD-20240310-001").

    Returns:
        Dict with keys:
            order_id (str):  The queried order ID.
            status   (str):  Current status: "pending", "processing",
                             "shipped", "delivered", "cancelled", "refunded".
            updated_at (str): ISO-8601 timestamp of the last status update.
            message  (str):  Human-readable status description.
        On error: {"error": str, "order_id": str}.
    """
    return _order_service.get_order_status(order_id)


@tool
def get_order_detail(order_id: str) -> dict:
    """
    Retrieve full details of an order including items, prices, and address.

    Use this tool when the user needs comprehensive order information beyond
    just the status, e.g. "What did I order?", "What is the total amount?".

    Args:
        order_id: The unique order identifier string.

    Returns:
        Dict with keys:
            order_id    (str):  Order ID.
            status      (str):  Current order status.
            items       (list): List of {"product_name": str, "quantity": int,
                                "unit_price": float} dicts.
            total_amount(float): Total order amount in CNY.
            created_at  (str):  Order creation timestamp.
            shipping_address (str): Delivery address.
            payment_method   (str): Payment method used.
        On error: {"error": str, "order_id": str}.
    """
    return _order_service.get_order(order_id)
