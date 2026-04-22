#!/usr/bin/env python3
"""
scripts/seed_test_data.py

Auto-generate test orders, logistics, and refund records and store them
in Redis so the mock services can load them at startup.

Generates:
    - 15 orders across 5 users covering all status transitions
    - Includes orders eligible/ineligible for refund (relative to today)
    - Full logistics + tracking event history for shipped/delivered orders
    - 2 pre-created refund records

Usage:
    cd ecommerce_customer_service
    python scripts/seed_test_data.py [--clear]

Redis key layout:
    mock:order:{order_id}              → JSON order dict
    mock:user_orders:{user_id}         → Redis SET of order_ids
    mock:logistics:{order_id}          → JSON logistics dict
    mock:tracking:{tracking_number}    → JSON tracking dict
    mock:refund:{refund_id}            → JSON refund dict
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import os
from collections import Counter
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis

# ─────────────────────────────────────────────────────────────────────────────
# Reference data
# ─────────────────────────────────────────────────────────────────────────────

PRODUCTS = [
    ("扫地机器人X1 Pro",     1999.00),
    ("扫拖一体机器人S5",     2999.00),
    ("无线蓝牙耳机 Pro",      399.00),
    ("机械键盘 RGB 版",       599.00),
    ("智能手表 Series 3",    1299.00),
    ("电动牙刷套装",           199.00),
    ("空气净化器 400㎡",      899.00),
    ("便携充电宝 20000mAh",   149.00),
    ("鼠标垫 XL 超大号",       49.00),
    ("高清网络摄像头",         399.00),
    ("降噪耳机",               799.00),
    ("智能门锁",              1599.00),
]

CARRIERS = [
    ("顺丰速运", "SF"),
    ("圆通速递", "YT"),
    ("京东物流", "JD"),
    ("中通快递", "ZT"),
]

CITY_ADDRESSES = [
    ("北京", "北京市朝阳区建国路 1 号"),
    ("上海", "上海市浦东新区陆家嘴 88 号"),
    ("广州", "广州市天河区珠江新城 66 号"),
    ("深圳", "深圳市南山区科技园 100 号"),
    ("杭州", "杭州市西湖区文三路 100 号"),
]

PAYMENT_METHODS = ["支付宝", "微信支付", "信用卡", "储蓄卡"]

ORIGIN_CITIES = ["广州", "深圳", "上海", "杭州", "武汉"]

# (status, days_ago_created, days_ago_delivered, user_hint)
# days_ago_delivered=None means not yet delivered
SCENARIOS = [
    # user_001 — multiple orders, including one refund-eligible
    ("pending",    0,  None, "user_001"),
    ("shipped",    3,  None, "user_001"),
    ("delivered",  6,  3,    "user_001"),   # ✅ refund eligible (delivered 3 days ago)
    # user_002 — order just outside refund window
    ("delivered",  10, 8,    "user_002"),   # ❌ 8 days ago, just past window
    ("shipped",    2,  None, "user_002"),
    # user_003 — cancelled and a fresh pending
    ("cancelled",  4,  None, "user_003"),
    ("pending",    1,  None, "user_003"),
    ("delivered",  5,  2,    "user_003"),   # ✅ refund eligible
    # user_004 — processing + old delivered
    ("processing", 2,  None, "user_004"),
    ("delivered",  20, 17,   "user_004"),   # ❌ too old
    ("shipped",    4,  None, "user_004"),
    # user_005 — variety
    ("delivered",  8,  5,    "user_005"),   # ✅ refund eligible (delivered 5 days ago)
    ("delivered",  30, 27,   "user_005"),   # ❌ very old
    ("shipped",    1,  None, "user_005"),
    ("processing", 3,  None, "user_005"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────────────────────────────────────

def order_id(dt: datetime, idx: int) -> str:
    return f"ORD-{dt.strftime('%Y%m%d')}-{idx:03d}"


def tracking_number(prefix: str, idx: int) -> str:
    return f"{prefix}{(idx * 137_000_000 + 900_000_000):010d}"


def build_tracking_events(
    origin: str, dest: str, created_at: datetime, delivered: bool
) -> tuple[list[dict], datetime]:
    events = []
    t = created_at + timedelta(hours=2)

    events.append({"timestamp": t.isoformat(), "location": f"{origin}揽收站",   "description": "已揽收"})
    t += timedelta(hours=8)
    events.append({"timestamp": t.isoformat(), "location": f"{origin}转运中心", "description": "已到达转运中心"})
    if origin != dest:
        t += timedelta(hours=16)
        events.append({"timestamp": t.isoformat(), "location": f"{dest}转运中心", "description": "已到达目的地转运中心"})
    t += timedelta(hours=6)
    events.append({"timestamp": t.isoformat(), "location": f"{dest}配送站",     "description": "派件中"})
    if delivered:
        t += timedelta(hours=3)
        events.append({"timestamp": t.isoformat(), "location": dest,             "description": "已签收"})

    return events, t


def generate_all(now: datetime) -> tuple[dict, dict, dict, dict]:
    orders: dict[str, dict]    = {}
    logistics: dict[str, dict] = {}
    tracking: dict[str, dict]  = {}
    refunds: dict[str, dict]   = {}

    for idx, (status, days_created, days_delivered, user_id) in enumerate(SCENARIOS, start=20):
        created_at   = now - timedelta(days=days_created,   hours=random.randint(0, 10))
        delivered_at = (
            (now - timedelta(days=days_delivered, hours=random.randint(0, 8))).isoformat()
            if days_delivered is not None else None
        )

        num_items = random.randint(1, 2)
        items_raw = random.sample(PRODUCTS, num_items)
        items = [
            {"product_name": name, "quantity": random.randint(1, 2), "unit_price": price}
            for name, price in items_raw
        ]
        total = round(sum(i["quantity"] * i["unit_price"] for i in items), 2)

        carrier_name, carrier_prefix = random.choice(CARRIERS)
        oid    = order_id(created_at, idx)
        tnum   = tracking_number(carrier_prefix, idx) if status in ("shipped", "delivered") else None
        dest_city, address = random.choice(CITY_ADDRESSES)

        orders[oid] = {
            "order_id":        oid,
            "user_id":         user_id,
            "status":          status,
            "items":           items,
            "total_amount":    total,
            "created_at":      created_at.isoformat(),
            "delivered_at":    delivered_at,
            "shipping_address": address,
            "payment_method":  random.choice(PAYMENT_METHODS),
            "tracking_number": tnum,
        }

        if status in ("shipped", "delivered"):
            origin = random.choice(ORIGIN_CITIES)
            is_delivered = (status == "delivered")
            events, last_t = build_tracking_events(origin, dest_city, created_at, is_delivered)

            logistics[oid] = {
                "order_id":          oid,
                "carrier":           carrier_name,
                "tracking_number":   tnum,
                "status":            "delivered" if is_delivered else "in_transit",
                "current_location":  events[-1]["location"],
                "estimated_delivery": (created_at + timedelta(days=4)).date().isoformat(),
                "last_update":        last_t.isoformat(),
            }
            tracking[tnum] = {
                "tracking_number": tnum,
                "carrier":         carrier_name,
                "status":          "delivered" if is_delivered else "in_transit",
                "events":          events,
            }

    # Add 2 pre-existing refund records for the oldest delivered orders
    delivered_ids = [
        oid for oid, o in orders.items()
        if o["status"] == "delivered" and o.get("delivered_at")
    ]
    delivered_ids.sort(key=lambda oid: orders[oid]["delivered_at"] or "")

    for i, oid in enumerate(delivered_ids[:2]):
        refund_id = f"REFUND-{now.strftime('%Y%m%d')}-{(i + 1):03d}"
        orders[oid]["status"] = "refunded"
        reasons = ["质量问题", "收到商品与描述不符"]
        refunds[refund_id] = {
            "refund_id":   refund_id,
            "order_id":    oid,
            "status":      "completed" if i == 0 else "pending_review",
            "amount":      orders[oid]["total_amount"],
            "reason":      reasons[i],
            "created_at":  now.isoformat(),
            "estimated_completion": (now + timedelta(days=3)).date().isoformat(),
            "message":     f"退款 {refund_id} {'已完成' if i == 0 else '审核中'}。",
        }

    return orders, logistics, tracking, refunds


# ─────────────────────────────────────────────────────────────────────────────
# Redis I/O
# ─────────────────────────────────────────────────────────────────────────────

def flush_existing(r: redis.Redis) -> None:
    keys = r.keys("mock:*")
    if keys:
        r.delete(*keys)
        print(f"  🗑  Cleared {len(keys)} existing mock:* keys")


def write_to_redis(
    r: redis.Redis,
    orders: dict,
    logistics: dict,
    tracking: dict,
    refunds: dict,
) -> None:
    pipe = r.pipeline()

    for oid, order in orders.items():
        pipe.set(f"mock:order:{oid}", json.dumps(order, ensure_ascii=False))
        pipe.sadd(f"mock:user_orders:{order['user_id']}", oid)

    for oid, rec in logistics.items():
        pipe.set(f"mock:logistics:{oid}", json.dumps(rec, ensure_ascii=False))

    for tnum, rec in tracking.items():
        pipe.set(f"mock:tracking:{tnum}", json.dumps(rec, ensure_ascii=False))

    for rid, rec in refunds.items():
        pipe.set(f"mock:refund:{rid}", json.dumps(rec, ensure_ascii=False))

    pipe.execute()


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(orders: dict, logistics: dict, tracking: dict, refunds: dict) -> None:
    now = datetime.now()
    print("\n" + "─" * 60)
    print(f"  📦 Orders: {len(orders)}  |  🚚 Logistics: {len(logistics)}  "
          f"|  🔍 Tracking: {len(tracking)}  |  💰 Refunds: {len(refunds)}")
    print("─" * 60)

    status_counts = Counter(o["status"] for o in orders.values())
    for s, n in sorted(status_counts.items()):
        print(f"  {s:<15} {n} order(s)")

    print("\n  Refund-eligible orders (delivered ≤ 7 days ago):")
    for oid, o in sorted(orders.items()):
        if o["status"] == "delivered" and o.get("delivered_at"):
            dt = datetime.fromisoformat(o["delivered_at"])
            days_since = (now - dt).days
            eligible = days_since <= 7
            marker = "✅" if eligible else "❌"
            print(f"    {marker} {oid}  user={o['user_id']}  "
                  f"delivered {days_since}d ago  ¥{o['total_amount']:.0f}")

    print("\n  Sample orders per user:")
    seen_users: set[str] = set()
    for oid, o in sorted(orders.items()):
        uid = o["user_id"]
        if uid not in seen_users:
            seen_users.add(uid)
            tnum = o.get("tracking_number") or "—"
            print(f"    {uid}  {oid}  {o['status']:<12}  tracking={tnum}")

    print("\n  Refunds:")
    for rid, rf in refunds.items():
        print(f"    {rid}  order={rf['order_id']}  status={rf['status']}  ¥{rf['amount']:.0f}")

    print("─" * 60)
    print("  ✅ Done. Restart the API server to pick up new data.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed test data into Redis")
    parser.add_argument("--clear", action="store_true", help="Clear existing mock:* keys first")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    args = parser.parse_args()

    r = redis.Redis(host=args.host, port=args.port, db=0, decode_responses=True)
    try:
        r.ping()
    except redis.ConnectionError:
        print(f"❌ Cannot connect to Redis at {args.host}:{args.port}")
        sys.exit(1)

    if args.clear:
        flush_existing(r)

    now = datetime.now()
    random.seed(42)
    orders, logistics, tracking, refunds = generate_all(now)
    write_to_redis(r, orders, logistics, tracking, refunds)
    print_summary(orders, logistics, tracking, refunds)


if __name__ == "__main__":
    main()
