"""
agents/order_agent.py

Order Agent — handles all transactional queries via LangChain tool calling.

Covers three intent sub-types that all require API access rather than RAG:
    - ORDER:      query order status and details
    - LOGISTICS:  track shipment, carrier info, estimated delivery
    - REFUND:     check eligibility, create refund, query refund status

Design pattern — ReAct tool-calling loop:
    The agent iterates: Thought → Action (tool call) → Observation (tool result)
    until the LLM decides the question is fully answered (no more tool calls).
    LangGraph's ToolNode + create_react_agent handles this loop, but a manual
    implementation gives finer control over error handling and retry logic.

Why not use a single monolithic agent?
    Separating order logic from FAQ avoids polluting the retrieval context with
    structured API data, and keeps each agent's prompt shorter and more focused.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class OrderAgent:
    """
    Tool-calling agent for order, logistics, and refund operations.

    Attributes:
        llm:        LangChain chat model bound with tools via .bind_tools().
        tools:      List of LangChain @tool functions available to the agent.
        tool_map:   Dict[tool_name → callable] for fast dispatch.
        max_iterations: Safety limit on the ReAct loop (default 5) to prevent
                    runaway tool-call chains.

    LangGraph integration:
        graph.add_node("order_agent", order_agent.run)
        graph.add_edge("order_agent", "response_agent")
    """

    def __init__(self, llm: Any, tools: list[Any], max_iterations: int = 5) -> None:
        """
        Initialise the order agent.

        Args:
            llm:            Chat model.  Will be bound with tools via
                            llm.bind_tools(tools) to enable structured
                            function-calling output.
            tools:          List of @tool decorated functions from tools/*.py.
                            Typically: [get_order_status, get_order_detail,
                            query_logistics, get_logistics_detail,
                            check_refund_eligibility, create_refund,
                            get_refund_status].
            max_iterations: Upper bound on ReAct loop iterations.

        TODO:
            - self.llm_with_tools = llm.bind_tools(tools)
            - self.tool_map = {t.name: t for t in tools}
            - self.max_iterations = max_iterations
            - Build a system prompt that lists available tools and their purpose.
        """
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = llm.bind_tools(tools)
        self.tool_map = {t.name: t for t in tools}
        self.max_iterations = max_iterations
        tool_descriptions = "\n".join([f"{t.name}: {t.description}" for t in tools])
        self.system_prompt = f"You are an assistant for handling customer service queries \
                                related to orders, logistics, and refunds. You have access \
                                to the following tools:\n{tool_descriptions}\nGiven a user \
                                query and classified intent, decide which tools to call and \
                                in what order to retrieve the necessary information to answer \
                                the query. Always call the most specific tool available for \
                                the intent. If the query is ambiguous, use your best judgement \
                                to choose the most relevant tool. If a tool call returns an error, \
                                acknowledge it and suggest contacting support."

    # ---------------------------------------------------------------------- #
    # Tool selection                                                           #
    # ---------------------------------------------------------------------- #

    def select_tool(self, query: str, intent: str) -> str:
        """
        Given the query and classified intent, decide which tool to invoke first.

        Args:
            query:  Rewritten user query.
            intent: IntentType value string ("order", "logistics", "refund").

        Returns:
            Tool name string (must be a key in self.tool_map), e.g.
            "get_order_status", "query_logistics", "check_refund_eligibility".

        How to implement (two approaches):

        Approach A — Rule-based (fast, 0 ms overhead):
            Map intent → default tool:
            {
                "order":     "get_order_status",
                "logistics": "query_logistics",
                "refund":    "check_refund_eligibility",
            }
            Then use keyword matching to pick a more specific tool if the
            query contains signals like "详情" → get_order_detail.

        Approach B — LLM-driven (flexible for complex queries):
            Ask the LLM to pick from the tool list given the query.
            More reliable when the query is ambiguous across tools.

        In practice: use Approach A as default; fall back to LLM when intent
        is UNKNOWN or the first tool call returns an error.
        """
        if intent == "order":
            if "详情" in query or "detail" in query:
                return "get_order_detail"
            return "get_order_status"
        elif intent == "logistics":
            if "carrier" in query or "承运商" in query:
                return "get_logistics_detail"
            return "query_logistics"
        elif intent == "refund":
            if "创建" in query or "create" in query:
                return "create_refund"
            elif "状态" in query or "status" in query:
                return "get_refund_status"
            return "check_refund_eligibility"
        else:
            # Intent is UNKNOWN; ask LLM to choose the most relevant tool
            prompt = f"""Given the user query: "{query}", and the following tools:
                        {', '.join(self.tool_map.keys())}
                        Which tool is most relevant to answer the query? Respond with only the tool name."""
            response = self.llm.invoke([("system", prompt)])
            chosen_tool = response.content.strip()
            if chosen_tool in self.tool_map:
                return chosen_tool
            else:
                logger.warning(f"LLM selected unknown tool: {chosen_tool}. Defaulting to 'get_order_status'.")
                return "get_order_status"

    # ---------------------------------------------------------------------- #
    # Tool execution                                                           #
    # ---------------------------------------------------------------------- #

    def execute_tool(self, tool_name: str, params: dict) -> dict:
        """
        Execute a tool by name with the given parameters.

        Args:
            tool_name: Name of the tool to invoke (key in self.tool_map).
            params:    Parameter dict extracted from the LLM's tool-call output.

        Returns:
            Dict with the tool's return value.  Wraps errors in a standardised
            {"error": str, "tool": tool_name} dict so the LLM can reason about
            failures.

        How to implement:
            1. Look up tool_fn = self.tool_map.get(tool_name).
            2. If not found, return {"error": f"unknown tool: {tool_name}"}.
            3. Call tool_fn.invoke(params) (LangChain tools use .invoke()).
            4. Return the result dict.
            5. Catch all exceptions; return {"error": repr(e), "tool": tool_name}.
            6. Log every tool call (name, params, result snippet) at INFO level
               for debugging and for computing tool-call success rate metrics.
        """
        tool_fn = self.tool_map.get(tool_name)
        if not tool_fn:
            error_msg = f"unknown tool: {tool_name}"
            logger.error(error_msg)
            return {"error": error_msg, "tool": tool_name}
        try:
            result = tool_fn.invoke(params)
            logger.info(f"Executed tool {tool_name} with params {params}. Result snippet: {str(result)[:100]}")
            return result
        except Exception as e:
            error_msg = f"error executing tool {tool_name}: {repr(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "tool": tool_name}

    # ---------------------------------------------------------------------- #
    # LangGraph node — ReAct loop                                             #
    # ---------------------------------------------------------------------- #

    def run(self, state: dict) -> dict:
        """
        LangGraph node entry point.  Runs the ReAct tool-calling loop until the
        LLM stops requesting tool calls or max_iterations is reached.

        Args:
            state: AgentState dict.
                   Reads:  "rewritten_query", "intent", "history", "user_id".
                   Writes: "tool_results" (list of tool call records),
                           "final_answer" (draft answer from tool results).

        Returns:
            Partial state dict.

        How to implement (ReAct loop):
            1. Build initial messages: [system_prompt, *history, HumanMessage(query)].
            2. Loop up to max_iterations:
               a. Call self.llm_with_tools.invoke(messages) → ai_message.
               b. If ai_message.tool_calls is empty → LLM is done; break.
               c. For each tool_call in ai_message.tool_calls:
                  - Extract tool_name and args.
                  - result = self.execute_tool(tool_name, args)
                  - Append ToolMessage(content=str(result), tool_call_id=...).
               d. Append ai_message + tool messages to messages.
            3. Extract final text from the last ai_message without tool calls.
            4. Collect all tool call records into tool_results list.
            5. Return {"tool_results": tool_results, "final_answer": final_text}.
            6. On LLM error, return a graceful error message and empty tool_results.
        """
        import json
        from langchain_core.messages import ToolMessage

        query = state.get("rewritten_query") or state["query"]
        history = state.get("history", [])
        messages: list = [("system", self.system_prompt)] + history + [("user", query)]
        tool_results = []
        try:
            ai_message = None
            for _ in range(self.max_iterations):
                ai_message = self.llm_with_tools.invoke(messages)
                if not ai_message.tool_calls:
                    break
                messages.append(ai_message)
                for tc in ai_message.tool_calls:
                    result = self.execute_tool(tc["name"], tc["args"])
                    tool_results.append({"tool": tc["name"], "args": tc["args"], "result": result})
                    messages.append(ToolMessage(
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=tc["id"],
                    ))
            final_answer = (ai_message.content or "").strip() if ai_message else ""
            if final_answer and not final_answer[-1] in ".?!。？！":
                final_answer += "。"
            return {"tool_results": tool_results, "final_answer": final_answer}
        except Exception as e:
            logger.error("Error during ReAct loop: %s", repr(e))
            return {"tool_results": [], "final_answer": "抱歉，处理您的请求时出现了问题，请联系人工客服。"}
