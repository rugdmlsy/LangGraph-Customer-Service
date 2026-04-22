"""
services/mock_order_service.py

Mock Order Service — simulates a real e-commerce order management backend.

Provides an in-memory fake database of orders and refunds so the agent
tools (tools/order_tools.py, tools/refund_tools.py) can be developed and
tested without a live backend.

Data model:
    Order: {
        order_id, user_id, status, items, total_amount,
        created_at, delivered_at, shipping_address, payment_method
    }
    Refund: {
        refund_id, order_id, status, amount, reason,
        created_at, completed_at
    }

Realistic status flow:
    pending → processing → shipped → delivered
                                   ↓
                                cancelled (before shipped)
                                refunded  (after delivered, within policy window)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Seed data — representative orders for testing all code paths                #
# --------------------------------------------------------------------------- #

_SEED_ORDERS: dict[str, dict] = {
    "ORD-20240310-001": {
        "order_id": "ORD-20240310-001",
        "user_id": "user_123",
        "status": "delivered",
        "items": [
            {"product_name": "无线蓝牙耳机", "quantity": 1, "unit_price": 299.00},
        ],
        "total_amount": 299.00,
        "created_at": "2024-03-10T10:00:00",
        "delivered_at": "2024-03-13T14:30:00",
        "shipping_address": "北京市朝阳区某某街道 100 号",
        "payment_method": "支付宝",
        "tracking_number": "SF1001001001",
    },
    "ORD-20240308-002": {
        "order_id": "ORD-20240308-002",
        "user_id": "user_456",
        "status": "shipped",
        "items": [
            {"product_name": "机械键盘", "quantity": 1, "unit_price": 599.00},
            {"product_name": "鼠标垫",   "quantity": 1, "unit_price": 49.00},
        ],
        "total_amount": 648.00,
        "created_at": "2024-03-08T09:00:00",
        "delivered_at": None,
        "shipping_address": "上海市浦东新区某某路 200 号",
        "payment_method": "微信支付",
        "tracking_number": "YT2002002002",
    },
    "ORD-20240301-003": {
        "order_id": "ORD-20240301-003",
        "user_id": "user_123",
        "status": "pending",
        "items": [
            {"product_name": "智能手表", "quantity": 1, "unit_price": 1299.00},
        ],
        "total_amount": 1299.00,
        "created_at": "2024-03-01T16:00:00",
        "delivered_at": None,
        "shipping_address": "广州市天河区某某大道 300 号",
        "payment_method": "信用卡",
        "tracking_number": None,
    },
}

_SEED_REFUNDS: dict[str, dict] = {}


class MockOrderService:
    """
    In-memory order management service for development and testing.

    The service stores orders and refunds in class-level dicts so state
    is shared across instances within the same process.  This mirrors
    how a singleton service client would work in production.

    Attributes:
        orders:  Dict[order_id → order dict].
        refunds: Dict[refund_id → refund dict].
    """
    orders: dict[str, dict] = {}
    refunds: dict[str, dict] = {}
    status_messages = {
        "pending":    "您的订单已提交，等待处理。",
        "processing": "您的订单正在处理中。",
        "shipped":    "您的订单已发货。",
        "delivered":  "您的订单已签收。",
        "cancelled":  "您的订单已取消。",
        "refunded":   "您的订单退款已完成。",
    }

    def __init__(self) -> None:
        self.orders = dict(_SEED_ORDERS)
        self.refunds = dict(_SEED_REFUNDS)
        self._load_from_redis()

    def _load_from_redis(self) -> None:
        """Load seeded orders and refunds from Redis (mock:order:* / mock:refund:*)."""
        try:
            import json
            import redis as _redis
            r = _redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
            for key in r.keys("mock:order:*"):
                order = json.loads(r.get(key))
                self.orders[order["order_id"]] = order
            for key in r.keys("mock:refund:*"):
                refund = json.loads(r.get(key))
                self.refunds[refund["refund_id"]] = refund
            extra = len(self.orders) - len(_SEED_ORDERS)
            logger.info("Loaded %d extra orders from Redis", extra)
        except Exception as exc:
            logger.debug("Redis seed load skipped: %s", exc)

    def get_order(self, order_id: str) -> dict:
        """
        Return full order details by order ID.

        Args:
            order_id: Order identifier.

        Returns:
            Order dict, or {"error": "Order not found", "order_id": order_id}.
        """
        order = self.orders.get(order_id)
        if not order:
            logger.warning("Order not found: %s", order_id)
            return {"error": "Order not found", "order_id": order_id}
        return dict(order)

    def get_order_status(self, order_id: str) -> dict:
        """
        Return a lightweight status-only response for an order.

        Args:
            order_id: Order identifier.

        Returns:
            {"order_id": str, "status": str, "updated_at": str, "message": str}
            or error dict.
        """
        order = self.get_order(order_id)
        if "error" in order:
            return order
        updated_at = order["delivered_at"] if order["delivered_at"] else order["created_at"]
        return {
            "order_id":   order_id,
            "status":     order["status"],
            "updated_at": updated_at,
            "message":    self.status_messages.get(order["status"], "状态未知。"),
        }

    def get_orders_by_user(self, user_id: str) -> list[dict]:
        """
        Return all orders belonging to a specific user.

        Args:
            user_id: User identifier.

        Returns:
            List of order dicts (may be empty if user has no orders).
        """
        return [o for o in self.orders.values() if o["user_id"] == user_id]

    def check_refund_eligibility(self, order_id: str) -> dict:
        """
        Apply the refund policy to determine if an order can be refunded.

        Policy:
            - Order must be in "delivered" status.
            - Delivery must have occurred within the past 7 days.

        Args:
            order_id: Order to check.

        Returns:
            {"order_id": str, "eligible": bool, "reason": str, "deadline": str|None}
        """
        order = self.get_order(order_id)
        if "error" in order:
            return {
                **order,
                "eligible": False,
                "reason": order.get("error", "订单不存在"),
                "deadline": None,
            }
        if order["status"] != "delivered":
            return {
                "order_id": order_id,
                "eligible": False,
                "reason": f"订单状态为 {order['status']}，不符合退款条件。",
                "deadline": None
            }
        delivered_at = datetime.fromisoformat(order["delivered_at"])
        deadline = delivered_at + timedelta(days=7)
        now = datetime.now()
        if now > deadline:
            return {
                "order_id": order_id,
                "eligible": False,
                "reason": "超过7天退款期限。",
                "deadline": deadline.date().isoformat()
            }
        return {
            "order_id": order_id,
            "eligible": True,
            "reason": f"订单于 {delivered_at.date()} 签收，在退款期限内。",
            "deadline": deadline.date().isoformat()
        }

    def create_refund(self, order_id: str, reason: str) -> dict:
        """
        Create a new refund request for an order.
        为内部refund字典添加一个新的退款请求，并将订单状态更新为"refunded"。

        Args:
            order_id: Order to refund.
            reason:   Refund reason provided by the user.

        Returns:
            New refund dict or error dict.
        """
        eligibility = self.check_refund_eligibility(order_id)
        if not eligibility.get("eligible"):
            return {"error": eligibility.get("reason", "不符合退款条件"), "order_id": order_id}
        order = self.get_order(order_id)
        refund_id = f"REFUND-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        refund = {
            "refund_id":   refund_id,
            "order_id":    order_id,
            "status":      "pending_review",
            "amount":      order["total_amount"],
            "reason":      reason,
            "created_at":  datetime.now().isoformat(),
            "estimated_completion": (datetime.now() + timedelta(days=5)).date().isoformat(),
            "message":     f"退款申请 {refund_id} 已提交，预计5个工作日内处理。",
        }
        self.refunds[refund_id] = refund
        self.orders[order_id]["status"] = "refunded"
        return refund
        
    def get_refund_status(self, refund_id: str) -> dict:
        """
        Return the current status of a refund request.

        Args:
            refund_id: Refund identifier.

        Returns:
            Refund dict or error dict.
        """
        refund = self.refunds.get(refund_id)
        if not refund:
            return {"error": "Refund not found", "refund_id": refund_id}
        return dict(refund)
