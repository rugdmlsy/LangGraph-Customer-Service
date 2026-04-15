"""
agents/response_agent.py

Response Agent — final answer synthesis.

This agent is the last node before the graph terminates.  It receives all
intermediate results (RAG context, tool call outputs, conversation history)
and produces a single coherent, customer-friendly answer.

Responsibilities:
    1. Merge partial answers from FAQAgent and/or OrderAgent.
    2. Apply tone and style guidelines (polite, concise, in the user's language).
    3. Enrich the answer with follow-up suggestions when appropriate.
    4. Handle the edge case where all upstream agents returned empty results.

Why a separate ResponseAgent?
    - In complex queries (e.g. "Can I return this item and what's the policy?")
      both FAQAgent (policy info) and OrderAgent (refund eligibility) may run.
      A dedicated synthesis step avoids awkward concatenation artefacts.
    - Centralising style / tone enforcement in one agent makes A/B testing
      different response styles trivial.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ResponseAgent:
    """
    Answer synthesis agent — terminal node of the LangGraph workflow.

    Attributes:
        llm: Chat model for synthesis.  Can be the same model instance shared
             across all agents or a smaller, faster model (e.g. Qwen1.5-1.8B)
             if latency is critical, since synthesis is a simpler task.

    LangGraph integration:
        graph.add_node("response_agent", response_agent.run)
        graph.set_finish_point("response_agent")
    """

    def __init__(self, llm: Any) -> None:
        """
        Initialise the response agent.

        Args:
            llm: Chat model instance.
        """
        self.llm = llm
        self.synthesis_prompt = [
            ("system", "You are a helpful assistant. Your task is as follows: \
                * Merge information from multiple sources without repetition. \
                * Answer in the same language as the user. \
                * Be concise (≤ 3 paragraphs for most queries). \
                * Suggest a relevant follow-up action when applicable. \
                * Never fabricate information not present in the provided inputs.")
        ]


    # ---------------------------------------------------------------------- #
    # Core synthesis                                                          #
    # ---------------------------------------------------------------------- #

    def synthesize(
        self,
        query: str,
        tool_results: list[dict],
        rag_results: list[dict],
        history: list[dict],
    ) -> str:
        """
        Merge tool call outputs and RAG snippets into a final customer response.

        Args:
            query:        Rewritten user query (for grounding the response).
            tool_results: List of tool call records from OrderAgent, each dict:
                          {"tool": str, "params": dict, "result": dict}.
                          May be empty if routing went to FAQAgent only.
            rag_results:  List of retrieved document dicts from FAQAgent.
                          May be empty if routing went to OrderAgent only.
            history:      Conversation history for coreference context.

        Returns:
            Final answer string suitable for returning directly to the user.

        Edge cases:
            - Tool call returned an error → acknowledge the failure politely
              and suggest the user contact human support.
            TODO: - RAG returned irrelevant docs (low scores) → omit them from the
              synthesis prompt to avoid confusing the LLM.
        """
        has_tool = len(tool_results) > 0
        has_rag = len(rag_results) > 0
        if not has_tool and not has_rag:
            return "I'm sorry, I couldn't find relevant information to answer \
                    your question. Please contact our support team for further assistance."
        if "error" in [res.get("result", {}).get("status") for res in tool_results]:
            return "I'm sorry, there was an issue retrieving some information. \
                    Please contact our support team for further assistance."
        tool_results_str = "None"
        if has_tool:
            tool_summaries = []
            for res in tool_results:
                tool_name = res["tool"]
                result_summary = res["result"].get("summary") or str(res["result"])
                tool_summaries.append(f"{tool_name} output: {result_summary}")
            tool_results_str = "\n".join(tool_summaries)
        rag_results_str = "None"
        if has_rag:
            rag_summaries = []
            for doc in rag_results:
                title = doc.get("metadata", {}).get("title", "Untitled")
                snippet = doc.get("snippet", "")
                rag_summaries.append(f"{title}: {snippet}")
            rag_results_str = "\n".join(rag_summaries)
        self.synthesis_prompt.append(
            ("user", f"User query: {query}\n\n"
                     f"Tool results: {tool_results_str if has_tool else 'None'}\n\n"
                     f"RAG results: {rag_results_str if has_rag else 'None'}\n\n"
                     "Please synthesise a single coherent answer based on the above information."))
        response = self.llm.invoke(self.synthesis_prompt)
        final_answer = response.content.strip()
        if not final_answer.endswith(('.', '?')):
            final_answer += '.'
        return final_answer
    
    # ---------------------------------------------------------------------- #
    # LangGraph node                                                          #
    # ---------------------------------------------------------------------- #

    def run(self, state: dict) -> dict:
        """
        LangGraph node entry point for the response agent.

        Args:
            state: AgentState dict.
                   Reads:  "rewritten_query" or "query", "tool_results",
                           "rag_results", "history".
                   Writes: "final_answer" (overrides any draft set by earlier agents).

        Returns:
            Partial state dict: {"final_answer": str}.

        How to implement:
            1. query        = state.get("rewritten_query") or state["query"]
            2. tool_results = state.get("tool_results", [])
            3. rag_results  = state.get("rag_results", [])
            4. history      = state.get("history", [])
            5. If the graph only ran one agent and it already set a good
               final_answer, optionally skip the LLM call and return as-is
               (check a "needs_synthesis" flag in state to control this).
            6. answer = self.synthesize(query, tool_results, rag_results, history)
            7. Return {"final_answer": answer}.
        """
        query = state.get("rewritten_query") or state["query"]
        tool_results = state.get("tool_results", [])
        rag_results = state.get("rag_results", [])
        history = state.get("history", [])
        if state.get("needs_synthesis") is False and "final_answer" in state:
            return {"final_answer": state["final_answer"]}
        answer = self.synthesize(query, tool_results, rag_results, history)
        return {"final_answer": answer}