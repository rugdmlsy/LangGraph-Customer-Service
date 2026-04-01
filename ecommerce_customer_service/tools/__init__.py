"""
tools package

LangChain @tool-decorated functions exposed to the OrderAgent for tool calling.

Tool groups:
    order_tools    – get_order_status, get_order_detail
    logistics_tools– query_logistics, get_logistics_detail
    refund_tools   – check_refund_eligibility, create_refund, get_refund_status

Each function uses @tool from langchain_core.tools so it can be passed to
llm.bind_tools() and executed via the ReAct loop in OrderAgent.
"""

from tools.order_tools import get_order_status, get_order_detail
from tools.logistics_tools import query_logistics, get_logistics_detail
from tools.refund_tools import (
    check_refund_eligibility,
    create_refund,
    get_refund_status,
)

ALL_TOOLS = [
    get_order_status,
    get_order_detail,
    query_logistics,
    get_logistics_detail,
    check_refund_eligibility,
    create_refund,
    get_refund_status,
]

__all__ = [
    "get_order_status",
    "get_order_detail",
    "query_logistics",
    "get_logistics_detail",
    "check_refund_eligibility",
    "create_refund",
    "get_refund_status",
    "ALL_TOOLS",
]
