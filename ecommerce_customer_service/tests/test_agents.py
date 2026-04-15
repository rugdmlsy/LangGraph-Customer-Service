"""
tests/test_agents.py

Unit tests for the four agent modules.

All external dependencies (LLM, retriever) are mocked using unittest.mock
so tests run fast and deterministically without a running LLM server.

Test coverage targets:
    - RouterAgent: intent classification for all IntentType values.
    - FAQAgent:    context retrieval and answer generation paths.
    - OrderAgent:  tool selection, execution, and ReAct loop.
    - ResponseAgent: synthesis with various combinations of inputs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.router_agent import IntentType, RouterAgent
from agents.faq_agent import FAQAgent
from agents.order_agent import OrderAgent
from agents.response_agent import ResponseAgent


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_llm():
    """Return a MagicMock simulating a LangChain chat model."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="mocked response")
    llm.bind_tools.return_value = llm  # bind_tools returns self
    return llm


@pytest.fixture
def mock_retriever():
    """Return a MagicMock simulating a Retriever instance."""
    retriever = MagicMock()
    retriever.hybrid_search.return_value = [
        {"id": 1, "text": "退款政策文本", "score": 0.92},
        {"id": 2, "text": "退款流程说明", "score": 0.85},
    ]
    retriever.rerank.return_value = [
        {"id": 1, "text": "退款政策文本", "rerank_score": 0.95},
    ]
    return retriever


@pytest.fixture
def mock_tools():
    """Return a list of mock tool objects."""
    tool = MagicMock()
    tool.name = "get_order_status"
    tool.invoke.return_value = {"status": "shipped"}
    return [tool]


# --------------------------------------------------------------------------- #
# RouterAgent tests                                                            #
# --------------------------------------------------------------------------- #


class TestRouterAgent:
    """Tests for RouterAgent intent classification and routing."""

    def test_router_intent_classification_faq(self, mock_llm):
        """
        Test that FAQ-type queries are classified as IntentType.FAQ.

        How to implement:
            1. Instantiate RouterAgent(mock_llm).
            2. Mock mock_llm.invoke to return a response containing "faq".
            3. Call router.classify_intent("退款政策是什么？").
            4. Assert result == IntentType.FAQ.

        TODO: implement test body
        """
        agent = RouterAgent(mock_llm)
        mock_llm.invoke.return_value = MagicMock(content="FAQ")
        result = agent.classify_intent("退款政策是什么？")
        assert result == IntentType.FAQ


    def test_router_intent_classification_order(self, mock_llm):
        """
        Test that order-related queries are classified as IntentType.ORDER.

        How to implement:
            Mock LLM response to contain "order", then assert classification.

        TODO: implement test body
        """
        agent = RouterAgent(mock_llm)
        mock_llm.invoke.return_value = MagicMock(content="ORDER")
        result = agent.classify_intent("我的订单在哪里？")
        assert result == IntentType.ORDER

    def test_router_intent_classification_logistics(self, mock_llm):
        """
        Test logistics queries → IntentType.LOGISTICS.

        TODO: implement test body
        """
        agent = RouterAgent(mock_llm)
        mock_llm.invoke.return_value = MagicMock(content="LOGISTICS")
        result = agent.classify_intent("快递什么时候到？")
        assert result == IntentType.LOGISTICS

    def test_router_intent_classification_refund(self, mock_llm):
        """
        Test refund queries → IntentType.REFUND.

        TODO: implement test body
        """
        agent = RouterAgent(mock_llm)
        mock_llm.invoke.return_value = MagicMock(content="REFUND")
        result = agent.classify_intent("我想申请退款。")
        assert result == IntentType.REFUND

    def test_router_intent_classification_unknown(self, mock_llm):
        """
        Test that ambiguous queries return IntentType.UNKNOWN rather than crashing.

        TODO: implement test body
        """
        agent = RouterAgent(mock_llm)
        mock_llm.invoke.return_value = MagicMock(content="UNKNOWN")
        result = agent.classify_intent("今天天气怎么样？")
        assert result == IntentType.UNKNOWN

    def test_router_rewrite_query(self, mock_llm):
        """
        Test that rewrite_query returns the LLM's output string.

        How to implement:
            mock_llm.invoke.return_value = MagicMock(content="rewritten query")
            router = RouterAgent(mock_llm)
            result = router.rewrite_query("快递呢", [])
            assert result == "rewritten query"

        TODO: implement test body
        """
        agent = RouterAgent(mock_llm)
        mock_llm.invoke.return_value = MagicMock(content="rewritten query")
        result = agent.rewrite_query("快递呢", [])
        assert result == "rewritten query"

    def test_router_rewrite_falls_back_on_llm_error(self, mock_llm):
        """
        Test that rewrite_query returns the original query when the LLM raises.

        How to implement:
            mock_llm.invoke.side_effect = Exception("LLM unavailable")
            router = RouterAgent(mock_llm)
            result = router.rewrite_query("original query", [])
            assert result == "original query"

        TODO: implement test body
        """
        agent = RouterAgent(mock_llm)
        mock_llm.invoke.side_effect = Exception("LLM unavailable")
        result = agent.rewrite_query("original query", [])
        assert result == "original query"

    def test_route_updates_state(self, mock_llm):
        """
        Test that route() updates state with rewritten_query and intent.

        How to implement:
            router = RouterAgent(mock_llm)
            # Patch rewrite_query and classify_intent
            router.rewrite_query = lambda q, h: "rewritten"
            router.classify_intent = lambda q: IntentType.FAQ
            state = {"query": "original", "history": []}
            result = router.route(state)
            assert state.get("rewritten_query") == "rewritten"
            assert state.get("intent") == IntentType.FAQ.value
            assert result == "faq_agent"

        TODO: implement test body
        """
        router = RouterAgent(mock_llm)
        # Patch rewrite_query and classify_intent
        router.rewrite_query = lambda query, history: "rewritten"
        router.classify_intent = lambda query: IntentType.FAQ
        state = {"query": "original", "history": []}
        result = router.route(state)
        assert state.get("rewritten_query") == "rewritten"
        assert state.get("intent") == IntentType.FAQ.value
        assert result == "faq_agent"


# --------------------------------------------------------------------------- #
# FAQAgent tests                                                               #
# --------------------------------------------------------------------------- #


class TestFAQAgent:
    """Tests for FAQAgent retrieval and generation."""

    def test_faq_agent_retrieval(self, mock_llm, mock_retriever):
        """
        Test that retrieve_context calls hybrid_search and rerank.

        How to implement:
            agent = FAQAgent(mock_llm, mock_retriever)
            docs  = agent.retrieve_context("退款政策", top_k=3)
            mock_retriever.hybrid_search.assert_called_once()
            mock_retriever.rerank.assert_called_once()
            assert isinstance(docs, list)

        TODO: implement test body
        """
        agent = FAQAgent(mock_llm, mock_retriever)
        docs  = agent.retrieve_context("退款政策", top_k=3)
        mock_retriever.hybrid_search.assert_called_once()
        mock_retriever.rerank.assert_called_once()
        assert isinstance(docs, list)

    def test_faq_agent_empty_retrieval_returns_no_info_message(self, mock_llm, mock_retriever):
        """
        Test that generate_answer with empty context returns a graceful message
        without calling the LLM.

        How to implement:
            mock_retriever.hybrid_search.return_value = []
            mock_retriever.rerank.return_value = []
            agent = FAQAgent(mock_llm, mock_retriever)
            answer = agent.generate_answer("question", [], [])
            # Should NOT call the LLM when context is empty
            mock_llm.invoke.assert_not_called()
            assert "没有" in answer or "无法" in answer or len(answer) > 0

        TODO: implement test body
        """
        mock_retriever.hybrid_search.return_value = []
        mock_retriever.rerank.return_value = []
        agent = FAQAgent(mock_llm, mock_retriever)
        answer = agent.generate_answer("question", [], [])
        # Should NOT call the LLM when context is empty
        mock_llm.invoke.assert_not_called()
        assert "没有" in answer or "无法" in answer or len(answer) > 0

    def test_faq_agent_run_updates_state(self, mock_llm, mock_retriever):
        """
        Test that run() updates state with rag_results and final_answer.

        How to implement:
            agent = FAQAgent(mock_llm, mock_retriever)
            state = {"rewritten_query": "退款政策", "history": []}
            result = agent.run(state)
            assert "rag_results" in result
            assert "final_answer" in result

        TODO: implement test body
        """
        agent = FAQAgent(mock_llm, mock_retriever)
        state = {"rewritten_query": "退款政策", "history": []}
        result = agent.run(state)
        assert "rag_results" in result
        assert "final_answer" in result


# --------------------------------------------------------------------------- #
# OrderAgent tests                                                             #
# --------------------------------------------------------------------------- #


class TestOrderAgent:
    """Tests for OrderAgent tool selection and execution."""

    def test_order_agent_tool_call(self, mock_llm, mock_tools):
        """
        Test that execute_tool calls the correct tool and returns its result.

        How to implement:
            agent = OrderAgent(mock_llm, mock_tools)
            result = agent.execute_tool("get_order_status", {"order_id": "ORD-001"})
            mock_tools[0].invoke.assert_called_once_with({"order_id": "ORD-001"})
            assert result == {"status": "shipped"}

        TODO: implement test body
        """
        agent = OrderAgent(mock_llm, mock_tools)
        result = agent.execute_tool("get_order_status", {"order_id": "ORD-001"})
        mock_tools[0].invoke.assert_called_once_with({"order_id": "ORD-001"})
        assert result == {"status": "shipped"}

    def test_order_agent_unknown_tool_returns_error(self, mock_llm, mock_tools):
        """
        Test that execute_tool returns an error dict for an unknown tool name.

        How to implement:
            agent = OrderAgent(mock_llm, mock_tools)
            result = agent.execute_tool("nonexistent_tool", {})
            assert "error" in result

        TODO: implement test body
        """
        agent = OrderAgent(mock_llm, mock_tools)
        result = agent.execute_tool("nonexistent_tool", {})
        assert "error" in result

    def test_order_agent_select_tool_order_intent(self, mock_llm, mock_tools):
        """
        Test that select_tool returns get_order_status for ORDER intent.

        TODO: implement test body
        """
        agent = OrderAgent(mock_llm, mock_tools)
        mock_llm.invoke.return_value = MagicMock(content="This query is about order")
        result = agent.select_tool("Where is my order?", IntentType.ORDER)
        assert result == "get_order_status"

    def test_order_agent_run_updates_state(self, mock_llm, mock_tools):
        """
        Test that run() returns tool_results and final_answer in the state dict.

        How to implement:
            Mock the LLM to return a response with no tool_calls (end of ReAct loop).
            Verify state keys are present.

        TODO: implement test body
        """
        agent = OrderAgent(mock_llm, mock_tools)
        mock_llm.invoke.return_value = MagicMock(content="No more tool calls")
        state = {"rewritten_query": "Where is my order?", "intent": IntentType.ORDER.value, "history": []}
        result = agent.run(state)
        assert "tool_results" in result
        assert "final_answer" in result