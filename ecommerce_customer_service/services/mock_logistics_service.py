"""
services/mock_logistics_service.py

Mock Logistics Service — simulates a shipment tracking backend.

Provides realistic fake tracking data for orders so the logistics tools
(tools/logistics_tools.py) can be exercised in development without calling
real carrier APIs.

Carriers simulated:
    - SF Express (顺丰速运)
    - YTO Express (圆通速递)
    - JD Logistics (京东物流)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Seed data                                                                    #
# --------------------------------------------------------------------------- #

_SEED_LOGISTICS: dict[str, dict] = {
    "ORD-20240310-001": {
        "order_id": "ORD-20240310-001",
        "carrier": "顺丰速运",
        "tracking_number": "SF1001001001",
        "status": "delivered",
        "current_location": "北京朝阳配送站",
        "estimated_delivery": "2024-03-13",
        "last_update": "2024-03-13T14:30:00",
    },
    "ORD-20240308-002": {
        "order_id": "ORD-20240308-002",
        "carrier": "圆通速递",
        "tracking_number": "YT2002002002",
        "status": "in_transit",
        "current_location": "上海转运中心",
        "estimated_delivery": "2024-03-11",
        "last_update": "2024-03-09T18:00:00",
    },
}

_SEED_TRACKING: dict[str, dict] = {
    "SF1001001001": {
        "tracking_number": "SF1001001001",
        "carrier": "顺丰速运",
        "status": "delivered",
        "events": [
            {"timestamp": "2024-03-10T11:00:00", "location": "广州揽收站",   "description": "已揽收"},
            {"timestamp": "2024-03-11T02:00:00", "location": "广州转运中心", "description": "已到达转运中心"},
            {"timestamp": "2024-03-12T06:00:00", "location": "北京转运中心", "description": "已到达目的地转运中心"},
            {"timestamp": "2024-03-13T09:00:00", "location": "北京朝阳配送站","description": "派件中"},
            {"timestamp": "2024-03-13T14:30:00", "location": "北京朝阳区",   "description": "已签收"},
        ],
    },
    "YT2002002002": {
        "tracking_number": "YT2002002002",
        "carrier": "圆通速递",
        "status": "in_transit",
        "events": [
            {"timestamp": "2024-03-08T10:00:00", "location": "深圳揽收站",   "description": "已揽收"},
            {"timestamp": "2024-03-09T03:00:00", "location": "深圳转运中心", "description": "已到达转运中心"},
            {"timestamp": "2024-03-09T18:00:00", "location": "上海转运中心", "description": "在途中"},
        ],
    },
}


class MockLogisticsService:
    """
    In-memory logistics service for development and testing.

    Attributes:
        logistics_db: Dict[order_id → logistics dict].
        tracking_db:  Dict[tracking_number → tracking detail dict].
    """

    def __init__(self) -> None:
        """
        Initialise with seed data.

        TODO:
            - self.logistics_db = dict(_SEED_LOGISTICS)
            - self.tracking_db  = dict(_SEED_TRACKING)
        """
        # TODO: implement
        pass

    def get_logistics(self, order_id: str) -> dict:
        """
        Return logistics summary for an order.

        Args:
            order_id: Order identifier.

        Returns:
            Logistics dict or {"error": "Logistics not found", "order_id": order_id}.

        How to implement:
            record = self.logistics_db.get(order_id)
            if not record:
                logger.warning("Logistics not found for order: %s", order_id)
                return {"error": "Logistics not found", "order_id": order_id}
            return dict(record)
        """
        # TODO: implement
        pass

    def get_tracking(self, tracking_number: str) -> dict:
        """
        Return full event history for a tracking number.

        Args:
            tracking_number: Carrier tracking number.

        Returns:
            Tracking detail dict or error dict.

        How to implement:
            record = self.tracking_db.get(tracking_number)
            if not record:
                return {"error": "Tracking number not found", "tracking_number": tracking_number}
            return dict(record)
        """
        # TODO: implement
        pass
