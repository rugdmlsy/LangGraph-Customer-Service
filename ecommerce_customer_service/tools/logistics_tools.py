"""
tools/logistics_tools.py

LangChain tool definitions for shipment tracking queries.

These tools wrap MockLogisticsService to provide the OrderAgent with
real-time (mocked) shipment tracking information.

Logistics data model:
    Each logistics record has: carrier, tracking_number, current_location,
    status (in_transit / delivered / exception), estimated_delivery date,
    and a movement history list.

Production integration note:
    Replace MockLogisticsService with calls to carrier APIs (e.g. SF Express,
    JD Logistics, China Post) or a third-party logistics aggregator API.
    Add timeout handling (logistics APIs can be slow) and cache results in
    Redis for 5 minutes to reduce API call volume.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from services.mock_logistics_service import MockLogisticsService

logger = logging.getLogger(__name__)

_logistics_service = MockLogisticsService()


@tool
def query_logistics(order_id: str) -> dict:
    """
    Query shipment tracking information for an order.

    Use this tool when the user asks about delivery status, carrier info,
    or expected arrival date for a specific order.
    Examples: "Where is my package?", "When will order ORD-001 arrive?"

    Args:
        order_id: The order ID whose shipment you want to track.

    Returns:
        Dict with keys:
            order_id          (str):  Order ID.
            carrier           (str):  Carrier name (e.g. "顺丰速运").
            tracking_number   (str):  Carrier-issued tracking number.
            status            (str):  "in_transit", "delivered", or "exception".
            current_location  (str):  Last known location of the package.
            estimated_delivery(str):  Expected delivery date (YYYY-MM-DD).
            last_update       (str):  Timestamp of the most recent scan event.
        On error: {"error": str, "order_id": str}.
    """
    return _logistics_service.get_logistics(order_id)


@tool
def get_logistics_detail(tracking_number: str) -> dict:
    """
    Retrieve the full movement history for a tracking number.

    Use this tool when the user wants to see step-by-step logistics events,
    e.g. "Show me all tracking events for my shipment".

    Args:
        tracking_number: Carrier tracking number (e.g. "SF1234567890").

    Returns:
        Dict with keys:
            tracking_number (str):  The queried tracking number.
            carrier         (str):  Carrier name.
            events          (list): Chronological list of
                                    {"timestamp": str, "location": str,
                                     "description": str} dicts.
            status          (str):  Current shipment status.
        On error: {"error": str, "tracking_number": str}.
    """
    return _logistics_service.get_tracking(tracking_number)
