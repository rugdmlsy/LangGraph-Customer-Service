"""
tests/test_tools.py

Unit tests for the LangChain tool functions.

These tests exercise the tool wrappers and the underlying MockOrderService /
MockLogisticsService without any LLM calls.

Focus areas:
    - Happy path: valid inputs return expected dict structure.
    - Error path: invalid order/refund IDs return error dicts (not exceptions).
    - Refund eligibility: test time-window logic.
    - Tool decorator: verify tools have correct .name attribute.
"""

from __future__ import annotations

import pytest

# Import the module-level tool functions
from tools.order_tools import get_order_status, get_order_detail
from tools.logistics_tools import query_logistics, get_logistics_detail
from tools.refund_tools import check_refund_eligibility, create_refund, get_refund_status
from unittest.mock import patch
from datetime import datetime, timedelta
import services.mock_order_service


# --------------------------------------------------------------------------- #
# Order tool tests                                                             #
# --------------------------------------------------------------------------- #


class TestOrderTools:
    """Tests for get_order_status and get_order_detail."""

    def test_get_order_status_valid_order(self):
        """
        get_order_status should return a dict with 'status' key for a known order.
        """
        result = get_order_status.invoke({"order_id": "ORD-20240310-001"})
        assert "status" in result
        assert result["order_id"] == "ORD-20240310-001"
        assert isinstance(result["status"], str)

    def test_get_order_status_invalid_order(self):
        """
        get_order_status with an unknown order ID should return an error dict.
        """
        result = get_order_status.invoke({"order_id": "INVALID-999"})
        assert "error" in result
        assert "INVALID-999" in result.get("order_id", "") or "error" in result

    def test_get_order_detail_returns_items(self):
        """
        get_order_detail should return a dict with 'items' list.
        """
        result = get_order_detail.invoke({"order_id": "ORD-20240310-001"})
        assert "items" in result
        assert isinstance(result["items"], list)
        assert len(result["items"]) > 0
        assert "product_name" in result["items"][0]

    def test_get_order_status_tool_name(self):
        """
        Verify the @tool decorator set the correct name attribute.
        """
        assert get_order_status.name == "get_order_status"


# --------------------------------------------------------------------------- #
# Logistics tool tests                                                         #
# --------------------------------------------------------------------------- #


class TestLogisticsTools:
    """Tests for query_logistics and get_logistics_detail."""

    def test_query_logistics_valid_order(self):
        """
        query_logistics should return carrier and tracking info for a known order.
        """
        result = query_logistics.invoke({"order_id": "ORD-20240310-001"})
        assert "carrier" in result
        assert "tracking_number" in result
        assert "status" in result

    def test_query_logistics_invalid_order(self):
        """
        query_logistics with an unknown order ID should return an error dict.
        """
        result = query_logistics.invoke({"order_id": "INVALID-999"})
        assert "error" in result

    def test_get_logistics_detail_returns_events(self):
        """
        get_logistics_detail should return a list of tracking events.
        """
        result = get_logistics_detail.invoke({"tracking_number": "SF1001001001"})
        assert "events" in result
        assert isinstance(result["events"], list)
        assert len(result["events"]) > 0
        assert "timestamp" in result["events"][0]
        assert "location" in result["events"][0]


# --------------------------------------------------------------------------- #
# Refund tool tests                                                            #
# --------------------------------------------------------------------------- #


class TestRefundTools:
    """Tests for refund eligibility, creation, and status."""

    def test_check_refund_eligibility_returns_bool(self):
        """
        check_refund_eligibility should return a dict with bool 'eligible' field.
        """
        result = check_refund_eligibility.invoke({"order_id": "ORD-20240310-001"})
        assert "eligible" in result
        assert isinstance(result["eligible"], bool)
        assert "reason" in result

    def test_check_refund_eligibility_non_delivered_order(self):
        """
        Orders not in 'delivered' status should not be eligible.
        """
        result = check_refund_eligibility.invoke({"order_id": "ORD-20240308-002"})
        # ORD-20240308-002 is in "shipped" status
        assert result["eligible"] is False
        assert "reason" in result

    def test_create_refund_ineligible_order_returns_error(self):
        """
        create_refund on an ineligible order should return an error dict.
        """
        result = create_refund.invoke({
            "order_id": "ORD-20240308-002",  # shipped, not delivered
            "reason": "不想要了",
        })
        assert "error" in result

    def test_get_refund_status_invalid_id(self):
        """
        get_refund_status with an unknown refund ID should return an error dict.
        """
        result = get_refund_status.invoke({"refund_id": "REF-INVALID"})
        assert "error" in result

    def test_create_and_get_refund_flow(self):
        """
        Integration test: create a refund on a delivered order, then query its status.

        Note: This test depends on the delivered_at timestamp being recent enough
        to pass the 7-day eligibility window.  In a real test, mock datetime.now().

        How to implement:
            from unittest.mock import patch
            from datetime import datetime, timedelta

            # Mock datetime.now() so the delivered order is within the 7-day window
            recent_delivery = (datetime.now() - timedelta(days=2)).isoformat()
            with patch.object(
                MockOrderService instance,
                "orders",
                {... order with recent delivered_at ...}
            ):
                eligibility = check_refund_eligibility.invoke({"order_id": "ORD-20240310-001"})
                if eligibility["eligible"]:
                    refund = create_refund.invoke({"order_id": "ORD-20240310-001", "reason": "test"})
                    assert "refund_id" in refund
                    status = get_refund_status.invoke({"refund_id": refund["refund_id"]})
                    assert status["refund_id"] == refund["refund_id"]
        """
        recent_delivery = (datetime.now() - timedelta(days=2)).isoformat()
        with patch.dict("tools.refund_tools._order_service.orders", {
            "TEST-ORDER-001": {
                "order_id": "TEST-ORDER-001",
                "user_id": "user_test",
                "status": "delivered",
                "items": [
                    {"product_name": "测试商品", "quantity": 1, "unit_price": 100.00},
                ],
                "total_amount": 100.00,
                "created_at": (datetime.now() - timedelta(days=5)).isoformat(),
                "delivered_at": recent_delivery,
                "shipping_address": "测试地址",
                "payment_method": "测试支付",
                "tracking_number": "TESTTRACK001",
            },
        }, clear=True):
            eligibility = check_refund_eligibility.invoke({"order_id": "TEST-ORDER-001"})
            print("Eligibility result:", eligibility)
            assert eligibility["eligible"] is True
            refund = create_refund.invoke({"order_id": "TEST-ORDER-001", "reason": "test"})
            assert "refund_id" in refund
            status = get_refund_status.invoke({"refund_id": refund["refund_id"]})
            assert status["refund_id"] == refund["refund_id"]
        
