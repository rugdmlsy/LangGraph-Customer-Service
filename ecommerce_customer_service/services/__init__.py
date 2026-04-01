"""
services package

Mock backend services that simulate the real e-commerce platform APIs.
Used for development and testing without requiring live system access.

In production:
    Replace each mock service with a real HTTP/gRPC client that calls the
    actual order management system, logistics platform, and payment gateway.
"""

from services.mock_order_service import MockOrderService
from services.mock_logistics_service import MockLogisticsService

__all__ = ["MockOrderService", "MockLogisticsService"]
